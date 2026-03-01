"""
Joint Audio-Video DiT

Main model combining Video DiT (Wan2.x) and Audio DiT via TA-CrossAttn
and UniTemp-RoPE.

Architecture:
  Dual Stream Phase (M=16 blocks):
    - Video tokens [B, T_v, D_v=4096] processed by Wan2.x DiT blocks
    - Audio tokens [B, T_a, D_a=1536] processed by AudioDiT blocks
    - TA-CrossAttn connects both paths at each block

  Single Stream Phase (N=40 blocks):
    - Audio tokens projected from D_a → D_v (1536 → 4096)
    - Both token sequences concatenated: [B, T_v+T_a, D_v=4096]
    - Processed jointly through standard transformer blocks
    - Split back into video [B, T_v, D_v] and audio [B, T_a, D_v]

  Output:
    - Video: [B, T_v, C_v_out=16]   — predicted velocity in video latent space
    - Audio: [B, T_a, C_a_out=32]   — predicted velocity in audio latent space
"""

import math
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ta_cross_attn import TACrossAttnBlock
from .unitime_rope import UniTempRoPE
from ..audio_dit.audio_dit import AudioDiTBlock


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding helper
# ---------------------------------------------------------------------------

def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal positional encoding for scalar timesteps.

    Args:
        t: Timesteps [B], values in (0, 1).
        dim: Embedding dimension (must be even).

    Returns:
        emb: [B, dim] sinusoidal embeddings.
    """
    assert dim % 2 == 0
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )  # [half]
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
    return torch.cat([args.sin(), args.cos()], dim=-1)  # [B, dim]


# ---------------------------------------------------------------------------
# Single Stream Block (joint processing of video+audio tokens)
# ---------------------------------------------------------------------------

class SingleStreamBlock(nn.Module):
    """
    Transformer block for the Single Stream Phase.

    Processes concatenated video+audio token sequences at a uniform dimension
    (D_v = 4096). Uses AdaLayerNormZero for timestep conditioning.

    Structure per block:
      AdaLN-Zero → Self-Attention → residual
      AdaLN-Zero → FFN           → residual
    """

    def __init__(
        self,
        dim: int = 4096,
        num_heads: int = 32,
        head_dim: int = 128,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        """
        Args:
            dim: Token hidden dimension.
            num_heads: Number of attention heads.
            head_dim: Dimension per head.
            mlp_ratio: FFN hidden / dim ratio.
            dropout: Dropout rate.
        """
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim

        # AdaLN-Zero: produces 6 modulation params from timestep embedding
        # Outputs: shift1, scale1, gate1 (attn), shift2, scale2, gate2 (FFN)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada_linear = nn.Linear(dim, dim * 6)

        # Fused QKV projection for efficiency
        self.qkv_proj = nn.Linear(dim, inner_dim * 3, bias=True)
        self.out_proj = nn.Linear(inner_dim, dim, bias=True)
        self.attn_dropout = nn.Dropout(dropout)

        # FFN: dim → mlp_ratio*dim → dim
        mlp_hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        # Zero-init the AdaLN modulation outputs so identity at start
        nn.init.zeros_(self.ada_linear.weight)
        nn.init.zeros_(self.ada_linear.bias)

        # Standard Xavier for QKV / output / FFN
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.zeros_(self.qkv_proj.bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        timestep_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Token sequence [B, T, dim].
            timestep_embed: Timestep embedding [B, dim].

        Returns:
            x: Updated token sequence [B, T, dim].
        """
        B, T, D = x.shape

        # AdaLN-Zero modulation parameters
        ada_params = self.ada_linear(timestep_embed).unsqueeze(1)  # [B, 1, D*6]
        shift1, scale1, gate1, shift2, scale2, gate2 = ada_params.chunk(6, dim=-1)

        # --- Self-Attention ---
        x_norm1 = self.norm1(x) * (1.0 + scale1) + shift1  # [B, T, D]

        # Fused QKV → reshape for multi-head attention
        qkv = self.qkv_proj(x_norm1)  # [B, T, 3 * inner_dim]
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each [B, T, num_heads, head_dim]
        q = q.transpose(1, 2)  # [B, heads, T, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        # RoPE for self-attention within single stream uses absolute token index
        scale = self.head_dim ** -0.5
        attn_w = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, heads, T, T]
        attn_w = F.softmax(attn_w, dim=-1)
        attn_w = self.attn_dropout(attn_w)
        attn_out = torch.matmul(attn_w, v)  # [B, heads, T, head_dim]

        attn_out = attn_out.transpose(1, 2).reshape(B, T, -1)  # [B, T, inner_dim]
        attn_out = self.out_proj(attn_out)  # [B, T, D]
        x = x + gate1 * attn_out

        # --- FFN ---
        x_norm2 = self.norm2(x) * (1.0 + scale2) + shift2
        x = x + gate2 * self.ffn(x_norm2)

        return x


# ---------------------------------------------------------------------------
# Dual Stream Block
# ---------------------------------------------------------------------------

class JointAVDiTBlock(nn.Module):
    """
    One block of the Dual Stream Phase.

    Each block runs:
      1. Video self-attention + text cross-attention  (Wan2.x block — injected externally)
      2. Audio DiT block  (fully implemented here)
      3. Bidirectional TA-CrossAttn between the two streams
    """

    def __init__(
        self,
        video_dim: int = 4096,
        audio_dim: int = 1536,
        video_num_heads: int = 32,
        audio_num_heads: int = 24,
        head_dim: int = 128,
        speech_text_dim: int = 4096,
        style_text_dim: int = 4096,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.video_dim = video_dim
        self.audio_dim = audio_dim

        # Audio DiT block (fully implemented)
        self.audio_block = AudioDiTBlock(
            dim=audio_dim,
            num_heads=audio_num_heads,
            head_dim=head_dim,
            speech_text_dim=speech_text_dim,
            style_text_dim=style_text_dim,
            dropout=dropout,
        )

        # Video DiT block — set to None here; populated by WanDiTAdapter at load time.
        # When video_block is None, video_hidden passes through unchanged.
        # See models/video_dit/wan2x_adapter.py for injection logic.
        self.video_block = None

        # Bidirectional TA-CrossAttn
        self.ta_cross_attn = TACrossAttnBlock(
            video_dim=video_dim,
            audio_dim=audio_dim,
            video_num_heads=video_num_heads,
            audio_num_heads=audio_num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )

    def forward(
        self,
        video_hidden: torch.Tensor,
        audio_hidden: torch.Tensor,
        timestep_embed: torch.Tensor,
        text_embeds: Optional[torch.Tensor] = None,
        speech_text: Optional[torch.Tensor] = None,
        style_text: Optional[torch.Tensor] = None,
        speech_mask: Optional[torch.Tensor] = None,
        style_mask: Optional[torch.Tensor] = None,
        video_positions: Optional[torch.Tensor] = None,
        audio_positions: Optional[torch.Tensor] = None,
        unitime_rope: Optional[UniTempRoPE] = None,
        audio_mask: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            video_hidden: [B, T_v, D_v]
            audio_hidden: [B, T_a, D_a]
            timestep_embed: [B, D_v]  — sinusoidal + MLP embedding
            text_embeds: [B, N_text, D_text]  — visual text for video path
            speech_text: [B, N_s, D_s]  — speech text for audio path
            style_text: [B, N_style, D_style]  — style text for audio path
            speech_mask: [B, N_s]
            style_mask: [B, N_style]
            video_positions: [T_v] physical time positions (seconds)
            audio_positions: [T_a] physical time positions (seconds)
            unitime_rope: UniTempRoPE module
            audio_mask: [B, T_a] padding mask
            video_mask: [B, T_v] padding mask

        Returns:
            video_hidden: [B, T_v, D_v]
            audio_hidden: [B, T_a, D_a]
        """
        # 1. Video self-attention (Wan2.x block, injected externally)
        if self.video_block is not None:
            video_hidden = self.video_block(video_hidden, timestep_embed, text_embeds)

        # 2. Audio DiT block
        audio_hidden = self.audio_block(
            audio_hidden,
            timestep_embed=timestep_embed,
            speech_text=speech_text,
            style_text=style_text,
            speech_mask=speech_mask,
            style_mask=style_mask,
        )

        # 3. Bidirectional TA-CrossAttn
        video_hidden, audio_hidden = self.ta_cross_attn(
            video_hidden,
            audio_hidden,
            video_positions=video_positions,
            audio_positions=audio_positions,
            unitime_rope=unitime_rope,
            audio_mask=audio_mask,
            video_mask=video_mask,
        )

        return video_hidden, audio_hidden


# ---------------------------------------------------------------------------
# Main JointAVDiT Model
# ---------------------------------------------------------------------------

class JointAVDiT(nn.Module):
    """
    Joint Audio-Video DiT Model.

    Supports:
    - Wan2.1-14B or Wan2.2-A14B as Video DiT base (loaded separately)
    - Audio DiT with dual conditioning (speech + style)
    - UniTemp-RoPE for AV time alignment
    - Multi-CFG inference
    """

    def __init__(
        self,
        # Video settings
        video_dim: int = 4096,
        video_num_heads: int = 32,
        # Audio settings
        audio_dim: int = 1536,
        audio_num_heads: int = 24,
        # Joint settings
        head_dim: int = 128,
        dual_stream_blocks: int = 16,
        single_stream_blocks: int = 40,
        single_stream_num_heads: int = 32,
        # Text encoder
        text_dim: int = 4096,
        speech_text_dim: int = 4096,
        style_text_dim: int = 4096,
        # RoPE
        rope_dim: int = 64,
        video_fps: int = 24,
        video_temporal_compress: int = 4,
        audio_sr: int = 16000,
        audio_temporal_compress: int = 320,
        # Input / output channels (latent space)
        video_input_dim: int = 16,
        audio_input_dim: int = 32,
        video_output_dim: int = 16,
        audio_output_dim: int = 32,
        # Other
        dropout: float = 0.0,
    ):
        """
        Initialize Joint AV DiT.

        Args:
            video_dim: Hidden dimension of the video stream (Wan2.x: 4096).
            video_num_heads: Number of attention heads in the video stream.
            audio_dim: Hidden dimension of the audio stream (1536).
            audio_num_heads: Number of attention heads in the audio stream.
            head_dim: Per-head dimension (128).
            dual_stream_blocks: Number of dual-stream blocks (M=16).
            single_stream_blocks: Number of single-stream blocks (N=40).
            single_stream_num_heads: Attention heads in single stream (32).
            text_dim: Dimension of the primary text embeddings (Qwen3: 4096).
            speech_text_dim: Dimension of speech text embeddings.
            style_text_dim: Dimension of style text embeddings.
            rope_dim: RoPE embedding dimension (64 per head pair).
            video_fps: Video frame rate (24).
            video_temporal_compress: Video VAE temporal compression (4).
            audio_sr: Audio sample rate (16000).
            audio_temporal_compress: Audio VAE temporal compression (320).
            video_input_dim: Video VAE latent channels (16).
            audio_input_dim: Audio VAE latent channels (32).
            video_output_dim: Output channels for video velocity prediction (16).
            audio_output_dim: Output channels for audio velocity prediction (32).
            dropout: Dropout rate.
        """
        super().__init__()

        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.dual_stream_blocks = dual_stream_blocks
        self.single_stream_blocks = single_stream_blocks

        # ------------------------------------------------------------------ #
        # UniTemp-RoPE for physical time alignment
        # ------------------------------------------------------------------ #
        self.unitime_rope = UniTempRoPE(
            dim=rope_dim,
            video_fps=video_fps,
            video_temporal_compress=video_temporal_compress,
            audio_sr=audio_sr,
            audio_temporal_compress=audio_temporal_compress,
        )

        # ------------------------------------------------------------------ #
        # Timestep embedding  (sinusoidal → MLP → [B, video_dim])
        # ------------------------------------------------------------------ #
        self.time_embed = nn.Sequential(
            nn.Linear(video_dim, video_dim * 4),
            nn.SiLU(),
            nn.Linear(video_dim * 4, video_dim),
        )

        # ------------------------------------------------------------------ #
        # Input projections: latent channels → hidden dim
        # ------------------------------------------------------------------ #
        self.video_input_proj = nn.Linear(video_input_dim, video_dim)
        self.audio_input_proj = nn.Linear(audio_input_dim, audio_dim)

        # ------------------------------------------------------------------ #
        # Dual Stream Phase (M blocks)
        # ------------------------------------------------------------------ #
        self.dual_stream = nn.ModuleList([
            JointAVDiTBlock(
                video_dim=video_dim,
                audio_dim=audio_dim,
                video_num_heads=video_num_heads,
                audio_num_heads=audio_num_heads,
                head_dim=head_dim,
                speech_text_dim=speech_text_dim,
                style_text_dim=style_text_dim,
                dropout=dropout,
            )
            for _ in range(dual_stream_blocks)
        ])

        # ------------------------------------------------------------------ #
        # Single Stream Phase (N blocks)
        # ------------------------------------------------------------------ #
        # Project audio tokens to joint dimension before concatenation
        self.audio_to_joint_proj = nn.Linear(audio_dim, video_dim)

        # Learned modality-type embeddings added before single stream
        # so the model can distinguish video tokens from audio tokens
        self.video_modality_embed = nn.Parameter(torch.zeros(1, 1, video_dim))
        self.audio_modality_embed = nn.Parameter(torch.zeros(1, 1, video_dim))

        self.single_stream = nn.ModuleList([
            SingleStreamBlock(
                dim=video_dim,
                num_heads=single_stream_num_heads,
                head_dim=head_dim,
                dropout=dropout,
            )
            for _ in range(single_stream_blocks)
        ])

        # ------------------------------------------------------------------ #
        # Output projections: hidden dim → latent channels
        # After single stream, both video and audio tokens are at video_dim.
        # ------------------------------------------------------------------ #
        self.video_norm_out = nn.LayerNorm(video_dim)
        self.audio_norm_out = nn.LayerNorm(video_dim)
        self.video_output_proj = nn.Linear(video_dim, video_output_dim)
        self.audio_output_proj = nn.Linear(video_dim, audio_output_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with standard practices for DiT models."""
        # Modality embeddings — small random init
        nn.init.normal_(self.video_modality_embed, std=0.02)
        nn.init.normal_(self.audio_modality_embed, std=0.02)

        # Time-embed MLP
        for m in self.time_embed:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        # Input projections
        nn.init.xavier_uniform_(self.video_input_proj.weight)
        nn.init.zeros_(self.video_input_proj.bias)
        nn.init.xavier_uniform_(self.audio_input_proj.weight)
        nn.init.zeros_(self.audio_input_proj.bias)

        # Audio-to-joint projection
        nn.init.xavier_uniform_(self.audio_to_joint_proj.weight)
        nn.init.zeros_(self.audio_to_joint_proj.bias)

        # Output projections — zero-init for stable training start
        nn.init.zeros_(self.video_output_proj.weight)
        nn.init.zeros_(self.video_output_proj.bias)
        nn.init.zeros_(self.audio_output_proj.weight)
        nn.init.zeros_(self.audio_output_proj.bias)

    def _compute_timestep_embed(self, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert scalar timestep [B] → embedding [B, video_dim].

        Uses sinusoidal encoding followed by a two-layer MLP.
        """
        sin_emb = sinusoidal_timestep_embedding(timestep, self.video_dim)  # [B, D]
        return self.time_embed(sin_emb.to(self.video_input_proj.weight.dtype))

    def forward(
        self,
        video_latent: torch.Tensor,
        audio_latent: torch.Tensor,
        text_embeds: torch.Tensor,
        timestep: torch.Tensor,
        speech_text: Optional[torch.Tensor] = None,
        style_text: Optional[torch.Tensor] = None,
        speech_mask: Optional[torch.Tensor] = None,
        style_mask: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Forward pass through Joint AV DiT.

        Args:
            video_latent: Video latent, either
                [B, C_v, T_v, H, W]  (raw VAE output, will be flattened), or
                [B, T_v, C_v]        (already sequence format).
            audio_latent: Audio latent, either
                [B, C_a, T_a]        (raw VAE output, will be transposed), or
                [B, T_a, C_a]        (already sequence format).
            text_embeds: Visual text embeddings [B, N_text, D_text].
            timestep: Diffusion timestep in (0, 1) [B].
            speech_text: Speech text embeddings [B, N_s, D_s] (optional).
            style_text: Style text embeddings [B, N_style, D_style] (optional).
            speech_mask: Speech text padding mask [B, N_s] (optional).
            style_mask: Style text padding mask [B, N_style] (optional).
            return_intermediates: If True, return intermediate hidden states.

        Returns:
            v_pred_video: Predicted video velocity.
                Shape matches input video_latent.
            v_pred_audio: Predicted audio velocity.
                Shape matches input audio_latent.
            intermediates: Dict of intermediate tensors, or None.
        """
        # ------------------------------------------------------------------ #
        # 1. Normalise input shapes to sequence format [B, T, C]
        # ------------------------------------------------------------------ #
        video_5d = video_latent.ndim == 5  # [B, C, T, H, W]
        audio_bct = (audio_latent.ndim == 3 and
                     audio_latent.shape[1] == self.audio_input_proj.in_features)

        if video_5d:
            B, C_v, T_v, H, W = video_latent.shape
            # Flatten spatial dims: [B, C, T, H, W] → [B, T*H*W, C]
            video_seq = video_latent.permute(0, 2, 3, 4, 1).reshape(B, T_v * H * W, C_v)
        else:
            video_seq = video_latent  # [B, T_v, C_v]
            B, T_v_flat, C_v = video_seq.shape
            T_v = T_v_flat

        if audio_bct:
            # [B, C_a, T_a] → [B, T_a, C_a]
            audio_seq = audio_latent.transpose(1, 2)
        else:
            audio_seq = audio_latent  # [B, T_a, C_a]

        T_a = audio_seq.shape[1]

        # ------------------------------------------------------------------ #
        # 2. Input projections: channels → hidden dims
        # ------------------------------------------------------------------ #
        video_hidden = self.video_input_proj(video_seq)   # [B, T_v, D_v]
        audio_hidden = self.audio_input_proj(audio_seq)   # [B, T_a, D_a]

        # ------------------------------------------------------------------ #
        # 3. Compute timestep embedding
        # ------------------------------------------------------------------ #
        t_emb = self._compute_timestep_embed(timestep)    # [B, D_v]

        # ------------------------------------------------------------------ #
        # 4. Physical time positions for UniTemp-RoPE
        # ------------------------------------------------------------------ #
        video_positions = self.unitime_rope.get_video_physical_positions(
            T_v, device=video_hidden.device, dtype=torch.float32
        )
        audio_positions = self.unitime_rope.get_audio_physical_positions(
            T_a, device=audio_hidden.device, dtype=torch.float32
        )

        # ------------------------------------------------------------------ #
        # 5. Dual Stream Phase
        # ------------------------------------------------------------------ #
        intermediates: Optional[Dict] = {} if return_intermediates else None

        for i, block in enumerate(self.dual_stream):
            video_hidden, audio_hidden = block(
                video_hidden=video_hidden,
                audio_hidden=audio_hidden,
                timestep_embed=t_emb,
                text_embeds=text_embeds,
                speech_text=speech_text,
                style_text=style_text,
                speech_mask=speech_mask,
                style_mask=style_mask,
                video_positions=video_positions,
                audio_positions=audio_positions,
                unitime_rope=self.unitime_rope,
            )
            if return_intermediates:
                intermediates[f"dual_{i}"] = {
                    "video": video_hidden.detach(),
                    "audio": audio_hidden.detach(),
                }

        # ------------------------------------------------------------------ #
        # 6. Single Stream Phase
        # ------------------------------------------------------------------ #
        # Project audio to joint (video) dimension
        audio_joint = self.audio_to_joint_proj(audio_hidden)  # [B, T_a, D_v]

        # Add modality-type embeddings so the model knows which tokens are video/audio
        video_joint = video_hidden + self.video_modality_embed  # [B, T_v, D_v]
        audio_joint = audio_joint + self.audio_modality_embed   # [B, T_a, D_v]

        # Concatenate along sequence dimension: [B, T_v+T_a, D_v]
        joint = torch.cat([video_joint, audio_joint], dim=1)

        for block in self.single_stream:
            joint = block(joint, t_emb)

        # Split back into video and audio portions
        video_joint_out = joint[:, :T_v, :]   # [B, T_v, D_v]
        audio_joint_out = joint[:, T_v:, :]   # [B, T_a, D_v]

        # ------------------------------------------------------------------ #
        # 7. Output projections: hidden dim → latent channels
        # ------------------------------------------------------------------ #
        v_pred_video = self.video_output_proj(self.video_norm_out(video_joint_out))  # [B, T_v, C_v_out]
        v_pred_audio = self.audio_output_proj(self.audio_norm_out(audio_joint_out))  # [B, T_a, C_a_out]

        # ------------------------------------------------------------------ #
        # 8. Restore original tensor layout
        # ------------------------------------------------------------------ #
        if video_5d:
            # [B, T_v*H*W, C_v_out] → [B, C_v_out, T_v, H, W]
            v_pred_video = v_pred_video.reshape(B, T_v, H, W, -1).permute(0, 4, 1, 2, 3)

        if audio_bct:
            # [B, T_a, C_a_out] → [B, C_a_out, T_a]
            v_pred_audio = v_pred_audio.transpose(1, 2)

        return v_pred_video, v_pred_audio, intermediates

    def load_pretrained(self, config: Dict[str, Any]) -> None:
        """
        Load pre-trained weights into the model.

        This method delegates to the WanModelLoader for the video path and
        loads audio DiT weights separately.

        Args:
            config: Configuration dict with keys:
                wan_path (str): Path to Wan2.x checkpoint directory.
                audio_dit_path (str, optional): Path to Audio DiT checkpoint.
        """
        from ..video_dit.wan2x_loader import WanModelLoader

        wan_path = config.get("wan_path")
        if wan_path:
            loader = WanModelLoader(
                model_path=wan_path,
                model_type=config.get("wan_model_type", "wan2.1"),
            )
            # Inject Wan2.x blocks into dual_stream blocks
            loader.inject_into_joint_model(self)

        audio_dit_path = config.get("audio_dit_path")
        if audio_dit_path:
            state = torch.load(audio_dit_path, map_location="cpu")
            # Load audio DiT weights for each dual-stream block
            missing, unexpected = self.load_state_dict(state, strict=False)
            if missing:
                print(f"[JointAVDiT] Missing keys when loading audio DiT: {missing[:5]}")


def create_joint_av_dit(config: Dict[str, Any]) -> JointAVDiT:
    """
    Factory function to create Joint AV DiT from config dict.

    Args:
        config: Configuration dictionary. Supported keys mirror JointAVDiT.__init__.

    Returns:
        JointAVDiT model instance.
    """
    return JointAVDiT(
        video_dim=config.get("video_dim", 4096),
        video_num_heads=config.get("video_num_heads", 32),
        audio_dim=config.get("audio_dim", 1536),
        audio_num_heads=config.get("audio_num_heads", 24),
        head_dim=config.get("head_dim", 128),
        dual_stream_blocks=config.get("dual_stream_blocks", 16),
        single_stream_blocks=config.get("single_stream_blocks", 40),
        single_stream_num_heads=config.get("single_stream_num_heads", 32),
        text_dim=config.get("text_dim", 4096),
        speech_text_dim=config.get("speech_text_dim", 4096),
        style_text_dim=config.get("style_text_dim", 4096),
        rope_dim=config.get("rope_dim", 64),
        video_fps=config.get("video_fps", 24),
        video_temporal_compress=config.get("video_temporal_compress", 4),
        audio_sr=config.get("audio_sr", 16000),
        audio_temporal_compress=config.get("audio_temporal_compress", 320),
        video_input_dim=config.get("video_input_dim", 16),
        audio_input_dim=config.get("audio_input_dim", 32),
        video_output_dim=config.get("video_output_dim", 16),
        audio_output_dim=config.get("audio_output_dim", 32),
        dropout=config.get("dropout", 0.0),
    )
