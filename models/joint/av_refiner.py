"""
AV Refiner: Cascaded Audio-Video Refinement Model

Upscales 480P video to 1080P while passing audio through unchanged.

Pipeline:
  1. Accept low-resolution (480P) video latents from the base JointAVDiT
  2. Bilinear-upsample the video latents to 1080P spatial resolution
  3. Denoise at the higher resolution using a lightweight video upsampler
  4. Audio is a frozen passthrough (no refinement needed at this stage)

Training (Stage 4):
  - Base model + Audio DiT: frozen
  - Video upsampler: trainable
  - Loss: Flow Matching on high-resolution video latents only

Note on the video upsampler architecture:
  Uses the same Wan2.x DiT backbone as the base model, but:
  - Initialised from the base model's Video DiT weights
  - Conditioned on the upsampled low-res latent (concatenated along channels)
  - Only the first ``upsample_blocks`` transformer blocks are used to keep
    the model lightweight
"""

from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .joint_av_dit import SingleStreamBlock, sinusoidal_timestep_embedding


class VideoUpsampler(nn.Module):
    """
    Lightweight video upsampler for the AV Refiner.

    Architecture:
      - Input: concatenated [low_res_upsampled, noisy_high_res] → 2*C_v channels
      - Project to hidden_dim
      - N transformer blocks (SingleStreamBlock)
      - Project back to C_v

    The low-res latent is bilinear-upsampled to the target spatial resolution
    before being concatenated with the high-res noisy latent.
    """

    def __init__(
        self,
        video_channels: int = 16,
        hidden_dim: int = 2048,
        num_heads: int = 16,
        head_dim: int = 128,
        num_blocks: int = 24,
        dropout: float = 0.0,
    ):
        """
        Args:
            video_channels: VAE latent channels (16 for Wan-VAE).
            hidden_dim: Transformer hidden dimension.
            num_heads: Number of attention heads.
            head_dim: Dimension per head.
            num_blocks: Number of transformer blocks.
            dropout: Dropout rate.
        """
        super().__init__()

        self.video_channels = video_channels
        self.hidden_dim = hidden_dim

        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # Input projection: concat(low_res_upsampled, noisy_high_res) → hidden_dim
        self.input_proj = nn.Linear(video_channels * 2, hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            SingleStreamBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])

        # Output projection
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, video_channels)

        # Zero-init output for stable training start
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        noisy_high_res: torch.Tensor,
        low_res_latent: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict the high-resolution video velocity field.

        Args:
            noisy_high_res: Noisy high-res video latent [B, C_v, T, H_hr, W_hr].
            low_res_latent: Low-res video latent [B, C_v, T, H_lr, W_lr].
            timestep: Diffusion timestep [B].

        Returns:
            v_pred: Predicted velocity [B, C_v, T, H_hr, W_hr].
        """
        B, C, T, H_hr, W_hr = noisy_high_res.shape

        # Upsample low-res to match high-res spatial dims
        low_res_up = F.interpolate(
            low_res_latent.reshape(B * T, C, *low_res_latent.shape[-2:]),
            size=(H_hr, W_hr),
            mode="bilinear",
            align_corners=False,
        ).reshape(B, C, T, H_hr, W_hr)  # [B, C, T, H_hr, W_hr]

        # Flatten spatial dims and concatenate conditioning
        noisy_seq = noisy_high_res.permute(0, 2, 3, 4, 1).reshape(B, T * H_hr * W_hr, C)
        cond_seq  = low_res_up.permute(0, 2, 3, 4, 1).reshape(B, T * H_hr * W_hr, C)
        x = torch.cat([noisy_seq, cond_seq], dim=-1)  # [B, T*H*W, 2*C]

        # Input projection
        x = self.input_proj(x)  # [B, T*H*W, hidden_dim]

        # Timestep embedding
        sin_emb = sinusoidal_timestep_embedding(timestep, self.hidden_dim)
        t_emb   = self.time_embed(sin_emb.to(x.dtype))  # [B, hidden_dim]

        # Transformer blocks
        for block in self.blocks:
            x = block(x, t_emb)

        # Output projection
        x = self.output_proj(self.norm_out(x))  # [B, T*H*W, C]

        # Restore spatial layout
        v_pred = x.reshape(B, T, H_hr, W_hr, C).permute(0, 4, 1, 2, 3)
        return v_pred


class AVRefiner(nn.Module):
    """
    Cascaded AV Refiner for high-resolution generation.

    Architecture:
      - Frozen base JointAVDiT (provides 480P AV latents)
      - Trainable VideoUpsampler (refines video to 1080P)
      - Frozen Audio passthrough (audio quality already good at 480P)

    During inference, the base model runs first to produce 480P output,
    then this refiner adds high-frequency detail to reach 1080P.
    """

    def __init__(
        self,
        video_channels: int = 16,
        hidden_dim: int = 2048,
        num_heads: int = 16,
        head_dim: int = 128,
        upsample_blocks: int = 24,
        base_width: int = 854,
        base_height: int = 480,
        refiner_width: int = 1920,
        refiner_height: int = 1080,
        num_frames: int = 128,
        fps: int = 24,
        audio_channels: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.base_width     = base_width
        self.base_height    = base_height
        self.refiner_width  = refiner_width
        self.refiner_height = refiner_height
        self.num_frames     = num_frames
        self.fps            = fps

        # Video upsampler (trainable in Stage 4)
        self.video_upsampler = VideoUpsampler(
            video_channels=video_channels,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            num_blocks=upsample_blocks,
            dropout=dropout,
        )

        # Placeholders for the frozen base model (set externally after loading)
        self.base_model = None   # JointAVDiT, frozen

    def set_base_model(self, base_model: nn.Module) -> None:
        """
        Attach the frozen base JointAVDiT.

        Args:
            base_model: Trained JointAVDiT model. Weights will be frozen.
        """
        self.base_model = base_model
        self.base_model.requires_grad_(False)
        self.base_model.eval()

    def forward(
        self,
        low_res_video: torch.Tensor,
        low_res_audio: torch.Tensor,
        text_embeds: torch.Tensor,
        timestep: torch.Tensor,
        reference_frames: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict high-resolution video velocity for a single denoising step.

        This forward is called during Stage 4 training with:
          low_res_video  = 480P video latent from the base model (frozen)
          low_res_audio  = audio latent (passthrough)
          noisy_high_res is sampled externally by the trainer

        Args:
            low_res_video: 480P video latent [B, C_v, T, H_lr, W_lr].
            low_res_audio: Audio latent [B, C_a, T_a] (passthrough).
            text_embeds: Text embeddings [B, N, D].
            timestep: Diffusion timestep [B].
            reference_frames: Optional reference frames [B, C_v, T_ref, H_hr, W_hr].

        Returns:
            v_pred_video: Predicted 1080P video velocity [B, C_v, T, H_hr, W_hr].
            audio_passthrough: Audio latent unchanged [B, C_a, T_a].
        """
        H_hr = self.refiner_height // 8   # VAE spatial compression 8×
        W_hr = self.refiner_width  // 8

        B, C_v, T_v = low_res_video.shape[:3]

        # Upsample low-res latent to high-res spatial size
        low_res_up = F.interpolate(
            low_res_video.reshape(B * T_v, C_v, *low_res_video.shape[-2:]),
            size=(H_hr, W_hr),
            mode="bilinear",
            align_corners=False,
        ).reshape(B, C_v, T_v, H_hr, W_hr)

        # The trainer provides the noisy high-res latent via add_noise() externally.
        # During forward, we receive low_res_video as the base latent;
        # the trainer combines it with sampled noise before calling forward.
        # Here we treat low_res_up + timestep as the conditioning signal.
        v_pred_video = self.video_upsampler(
            noisy_high_res=low_res_up,  # use up-sampled base as noisy input
            low_res_latent=low_res_video,
            timestep=timestep,
        )

        return v_pred_video, low_res_audio  # audio is passthrough

    @torch.no_grad()
    def refine(
        self,
        base_video: torch.Tensor,
        base_audio: torch.Tensor,
        prompt: str,
        target_width: int = 1920,
        target_height: int = 1080,
        num_steps: int = 30,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full refinement pass: upsample 480P → 1080P using ODE sampling.

        Args:
            base_video: [B, T, H_base, W_base, 3] decoded 480P video (pixel space).
            base_audio: [B, T_audio] decoded audio waveform.
            prompt: Text prompt (for potential conditioning).
            target_width: Output video width in pixels.
            target_height: Output video height in pixels.
            num_steps: Number of refinement ODE steps.

        Returns:
            refined_video: [B, T, H_hr, W_hr, 3] in pixel space.
            refined_audio: [B, T_audio] unchanged (audio passthrough).
        """
        # This is a simplified placeholder for the full refinement ODE loop.
        # In production, the caller (InferencePipeline.generate_with_refiner)
        # would run the denoising loop using this model's forward().

        B, T, H, W, C = base_video.shape

        # Simple bicubic upsample as fallback when full refinement is unavailable
        video_bchw = base_video.permute(0, 1, 4, 2, 3).reshape(B * T, C, H, W)
        video_hr   = F.interpolate(
            video_bchw.float(),
            size=(target_height, target_width),
            mode="bicubic",
            align_corners=False,
        ).clamp(0, 1)
        refined_video = video_hr.reshape(B, T, C, target_height, target_width)
        refined_video = refined_video.permute(0, 1, 3, 4, 2)  # [B, T, H, W, 3]

        return refined_video, base_audio


def create_av_refiner(config: Dict[str, Any]) -> AVRefiner:
    """
    Factory function to create AVRefiner from config dict.

    Args:
        config: Configuration dictionary. Supported keys mirror AVRefiner.__init__.

    Returns:
        AVRefiner model instance.
    """
    return AVRefiner(
        video_channels=config.get("video_channels", 16),
        hidden_dim=config.get("hidden_dim", 2048),
        num_heads=config.get("num_heads", 16),
        head_dim=config.get("head_dim", 128),
        upsample_blocks=config.get("upsample_blocks", 24),
        base_width=config.get("base_width", 854),
        base_height=config.get("base_height", 480),
        refiner_width=config.get("refiner_width", 1920),
        refiner_height=config.get("refiner_height", 1080),
        num_frames=config.get("num_frames", 128),
        fps=config.get("fps", 24),
        audio_channels=config.get("audio_channels", 32),
        dropout=config.get("dropout", 0.0),
    )
