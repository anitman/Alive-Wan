"""
Audio DiT Implementation

Core Audio DiT model for Alive-Wan2X.
Based on Alive architecture with dual conditioning for speech and style.
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioDiTBlock(nn.Module):
    """
    Single Audio DiT Block with dual conditioning.
    
    Structure:
    - AdaLayerNormZero (timestep conditioning)
    - Self-Attention
    - Cross-Attention → Speech Text (what is said)
    - Cross-Attention → Descriptive Prompt (style/environment)
    - FFN
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        speech_text_dim: int = 4096,
        style_text_dim: int = 4096,
        dropout: float = 0.0
    ):
        """
        Initialize Audio DiT Block.
        
        Args:
            dim: Hidden dimension
            num_heads: Number of attention heads
            head_dim: Dimension per head
            speech_text_dim: Speech text embedding dimension
            style_text_dim: Style prompt embedding dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        
        # AdaLayerNormZero for timestep conditioning
        # TODO: Implement AdaLayerNormZero with proper modulation
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ada_norm_linear = nn.Linear(dim, dim * 6)  # scale, shift for each norm
        
        # Self-Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            kdim=dim,
            vdim=dim,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-Attention for Speech Text (controls what is said)
        self.speech_cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            kdim=speech_text_dim,
            vdim=speech_text_dim,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-Attention for Descriptive Prompt (controls style)
        self.style_cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            kdim=style_text_dim,
            vdim=style_text_dim,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-Forward Network (GELU activation, 4× expansion)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

        # Layer norms for cross-attention sub-layers
        # norm2: applied before speech cross-attention
        # norm3: applied before style cross-attention
        # (FFN is conditioned via AdaLN, so uses norm1 pathway only)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
    
    def forward(
        self,
        x: torch.Tensor,
        timestep_embed: torch.Tensor,
        speech_text: Optional[torch.Tensor] = None,
        style_text: Optional[torch.Tensor] = None,
        speech_mask: Optional[torch.Tensor] = None,
        style_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through Audio DiT Block.
        
        Args:
            x: Input [B, T, D]
            timestep_embed: Timestep embedding [B, D]
            speech_text: Speech text embeddings [B, N_s, D_s]
            style_text: Style text embeddings [B, N_style, D_style]
            speech_mask: Speech text mask [B, N_s]
            style_mask: Style text mask [B, N_style]
            
        Returns:
            Output [B, T, D]
        """
        # TODO: Implement proper forward with:
        # 1. AdaLayerNormZero modulation
        # 2. Self-Attention
        # 3. Speech text cross-attention
        # 4. Style text cross-attention
        # 5. FFN
        
        residual = x

        # AdaLayerNormZero: compute scale/shift/gate from timestep
        ada_params = self.ada_norm_linear(timestep_embed)  # [B, D*6]
        if ada_params.ndim == 2:
            ada_params = ada_params.unsqueeze(1)  # [B, 1, D*6]
        shift1, scale1, gate1, shift2, scale2, gate2 = ada_params.chunk(6, dim=-1)

        # Modulated self-attention
        x_norm = self.norm1(x) * (1 + scale1) + shift1
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + gate1 * attn_out
        
        # Speech text cross-attention (if provided)
        if speech_text is not None:
            speech_out, _ = self.speech_cross_attn(
                self.norm2(x), speech_text, speech_text,
                key_padding_mask=speech_mask
            )
            x = x + speech_out

        # Style text cross-attention (if provided)
        if style_text is not None:
            style_out, _ = self.style_cross_attn(
                self.norm3(x), style_text, style_text,
                key_padding_mask=style_mask
            )
            x = x + style_out

        # FFN with AdaLN modulation
        x_norm_ffn = self.norm1(x) * (1 + scale2) + shift2
        x = x + gate2 * self.ffn(x_norm_ffn)
        
        return x


class AudioDiT(nn.Module):
    """
    Audio DiT Model for Alive-Wan2X.
    
    Architecture:
    - 32 Transformer blocks
    - 1536 hidden dim
    - 24 heads × 64 head_dim
    - Dual conditioning: Speech + Style
    """
    
    def __init__(
        self,
        num_blocks: int = 32,
        hidden_dim: int = 1536,
        num_heads: int = 24,
        head_dim: int = 64,
        input_dim: int = 32,
        output_dim: int = 32,
        speech_text_dim: int = 4096,
        style_text_dim: int = 4096,
        max_audio_length: int = 256,
        dropout: float = 0.0
    ):
        """
        Initialize Audio DiT.
        
        Args:
            num_blocks: Number of Transformer blocks
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            head_dim: Dimension per head
            input_dim: Input latent channels
            output_dim: Output latent channels
            speech_text_dim: Speech text embedding dimension
            style_text_dim: Style text embedding dimension
            max_audio_length: Maximum audio sequence length
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Input projection
        # TODO: Implement proper input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional embedding
        # TODO: Implement UniTemp-RoPE for audio
        self.pos_embed = nn.Parameter(torch.zeros(1, max_audio_length, hidden_dim))
        
        # Timestep embedding
        # TODO: Implement proper timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Audio DiT blocks
        self.blocks = nn.ModuleList([
            AudioDiTBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                speech_text_dim=speech_text_dim,
                style_text_dim=style_text_dim,
                dropout=dropout
            )
            for _ in range(num_blocks)
        ])
        
        # Output projection
        # TODO: Implement proper output normalization
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """
        Initialize model weights.

        - Linear layers: Xavier uniform
        - AdaLN linear: zero-init (identity modulation at start)
        - Output projection: zero-init (stable training start)
        - Positional embedding: small normal init
        """
        import math

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Zero-init output proj for stable convergence
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        # Small init for positional embedding
        nn.init.normal_(self.pos_embed, std=0.02)

        # Zero-init all ada_norm_linear layers so initial modulation is identity
        for block in self.blocks:
            nn.init.zeros_(block.ada_norm_linear.weight)
            nn.init.zeros_(block.ada_norm_linear.bias)
    
    def forward(
        self,
        audio_latent: torch.Tensor,
        timestep: torch.Tensor,
        speech_text: Optional[torch.Tensor] = None,
        style_text: Optional[torch.Tensor] = None,
        speech_mask: Optional[torch.Tensor] = None,
        style_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through Audio DiT.
        
        Args:
            audio_latent: Audio latent [B, C, T] or [B, T, C]
            timestep: Diffusion timestep [B]
            speech_text: Speech text embeddings [B, N_s, D_s]
            style_text: Style text embeddings [B, N_style, D_style]
            speech_mask: Speech text mask [B, N_s]
            style_mask: Style text mask [B, N_style]
            
        Returns:
            noise_pred: Predicted noise [B, C, T] or [B, T, C]
        """
        # TODO: Implement full forward pass

        # Handle input shape: [B, C, T] → [B, T, C] for Linear projection
        if audio_latent.ndim == 3 and audio_latent.shape[1] == self.input_proj.in_features:
            audio_latent = audio_latent.transpose(1, 2)

        # Input projection
        x = self.input_proj(audio_latent)
        
        # Add positional embedding
        # TODO: Apply UniTemp-RoPE
        
        # Timestep embedding: sinusoidal encoding then MLP
        # timestep is [B], needs to be embedded to [B, hidden_dim] first
        half_dim = self.hidden_dim // 2
        emb = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=timestep.device, dtype=timestep.dtype) / half_dim
        )
        t_emb = timestep.unsqueeze(1) * emb.unsqueeze(0)  # [B, hidden_dim//2]
        t_emb = torch.cat([t_emb.sin(), t_emb.cos()], dim=-1)  # [B, hidden_dim]
        timestep_embed = self.time_embed(t_emb)
        
        # Pass through blocks
        for block in self.blocks:
            x = block(
                x,
                timestep_embed=timestep_embed,
                speech_text=speech_text,
                style_text=style_text,
                speech_mask=speech_mask,
                style_mask=style_mask
            )
        
        # Output projection
        x = self.norm_out(x)
        noise_pred = self.output_proj(x)
        
        return noise_pred
    
    def load_pretrained(self, checkpoint_path: str) -> None:
        """Load pre-trained weights."""
        # TODO: Implement checkpoint loading
        pass


def create_audio_dit(config: dict) -> AudioDiT:
    """
    Factory function to create Audio DiT from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        AudioDiT model
    """
    return AudioDiT(
        num_blocks=config.get("num_blocks", 32),
        hidden_dim=config.get("hidden_dim", 1536),
        num_heads=config.get("num_heads", 24),
        head_dim=config.get("head_dim", 64),
        input_dim=config.get("input_dim", 32),
        output_dim=config.get("output_dim", 32),
        speech_text_dim=config.get("speech_text_dim", 4096),
        style_text_dim=config.get("style_text_dim", 4096),
        dropout=config.get("dropout", 0.0)
    )