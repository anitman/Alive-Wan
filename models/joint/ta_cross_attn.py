"""
TA-CrossAttn: Time-Aligned Cross Attention

Core module for audio-visual cross-modal attention with time alignment.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TACrossAttention(nn.Module):
    """
    Time-Aligned Cross Attention module.
    
    Enables bidirectional attention between video and audio with
    UniTemp-RoPE for physical time alignment.
    
    Key features:
    - Bidirectional: audio→video and video→audio
    - Time-aligned via UniTemp-RoPE
    - Supports variable-length sequences
    """
    
    def __init__(
        self,
        query_dim: int,
        cross_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        qkv_bias: bool = True
    ):
        """
        Initialize TA Cross Attention.
        
        Args:
            query_dim: Query dimension (receiving modality)
            cross_dim: Cross dimension (attended modality)
            num_heads: Number of attention heads
            head_dim: Dimension per head
            dropout: Dropout rate
            qkv_bias: Use bias in QKV projections
        """
        super().__init__()
        
        self.query_dim = query_dim
        self.cross_dim = cross_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        
        # Q projection for query modality
        self.q_proj = nn.Linear(query_dim, self.inner_dim, bias=qkv_bias)
        
        # K, V projections for cross modality
        self.k_proj = nn.Linear(cross_dim, self.inner_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(cross_dim, self.inner_dim, bias=qkv_bias)
        
        # Output projection
        self.out_proj = nn.Linear(self.inner_dim, query_dim, bias=qkv_bias)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Zero-init the output projection so TA-CrossAttn starts as identity
        (no cross-modal signal at the beginning of training, which is stable
        when only CrossAttn is unfrozen in Stage 1).
        """
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        cross: torch.Tensor,
        video_positions: Optional[torch.Tensor] = None,
        audio_positions: Optional[torch.Tensor] = None,
        unitime_rope: Optional["UniTempRoPE"] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through TA Cross Attention.
        
        Args:
            query: Query [B, T_q, D_q]
            cross: Cross modality [B, T_c, D_c]
            video_positions: Video token positions [T_video] (physical time in seconds)
            audio_positions: Audio token positions [T_audio] (physical time in seconds)
            unitime_rope: UniTemp-RoPE module for time alignment
            key_padding_mask: Mask for padding [B, T_c]
            
        Returns:
            output: [B, T_q, D_q]
        """
        B, T_q, _ = query.shape
        B, T_c, _ = cross.shape
        
        # Project Q, K, V
        q = self.q_proj(query)  # [B, T_q, inner_dim]
        k = self.k_proj(cross)  # [B, T_c, inner_dim]
        v = self.v_proj(cross)  # [B, T_c, inner_dim]
        
        # Reshape for multi-head attention
        q = q.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T_c, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T_c, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply UniTemp-RoPE if provided
        # TODO: Implement RoPE application
        if unitime_rope is not None and video_positions is not None and audio_positions is not None:
            q, k = unitime_rope.apply_rope(q, k, video_positions, audio_positions)
        
        # Scaled dot-product attention
        # attention_scores = (q @ k.transpose(-2, -1)) / sqrt(head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply padding mask if provided
        if key_padding_mask is not None:
            # key_padding_mask: [B, T_c]
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # [B, heads, T_q, head_dim]
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous()  # [B, T_q, heads, head_dim]
        output = output.view(B, T_q, self.inner_dim)
        output = self.out_proj(output)  # [B, T_q, query_dim]
        
        return output


class TACrossAttnBlock(nn.Module):
    """
    TA-CrossAttn Block for Dual Stream phase.
    
    Implements bidirectional cross-attention between video and audio:
    - Audio attends to Video (for visual context)
    - Video attends to Audio (for sound signal)
    """
    
    def __init__(
        self,
        video_dim: int,
        audio_dim: int,
        video_num_heads: int,
        audio_num_heads: int,
        head_dim: int,
        dropout: float = 0.0
    ):
        """
        Initialize TA-CrossAttn Block.
        
        Args:
            video_dim: Video hidden dimension
            audio_dim: Audio hidden dimension
            video_num_heads: Video attention heads
            audio_num_heads: Audio attention heads
            head_dim: Head dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        # Audio → Video: audio queries, video keys/values
        self.audio_to_video = TACrossAttention(
            query_dim=audio_dim,
            cross_dim=video_dim,
            num_heads=audio_num_heads,
            head_dim=head_dim,
            dropout=dropout
        )
        
        # Video → Audio: video queries, audio keys/values
        self.video_to_audio = TACrossAttention(
            query_dim=video_dim,
            cross_dim=audio_dim,
            num_heads=video_num_heads,
            head_dim=head_dim,
            dropout=dropout
        )
        
        # Layer normalization
        self.norm_a2v = nn.LayerNorm(audio_dim)
        self.norm_v2a = nn.LayerNorm(video_dim)
    
    def forward(
        self,
        video_hidden: torch.Tensor,
        audio_hidden: torch.Tensor,
        video_positions: Optional[torch.Tensor] = None,
        audio_positions: Optional[torch.Tensor] = None,
        unitime_rope: Optional["UniTempRoPE"] = None,
        audio_mask: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through bidirectional TA-CrossAttn.
        
        Args:
            video_hidden: Video hidden states [B, T_v, D_v]
            audio_hidden: Audio hidden states [B, T_a, D_a]
            video_positions: Video positions [T_v]
            audio_positions: Audio positions [T_a]
            unitime_rope: UniTemp-RoPE module
            audio_mask: Audio padding mask [B, T_a]
            video_mask: Video padding mask [B, T_v]
            
        Returns:
            video_hidden: Updated video hidden [B, T_v, D_v]
            audio_hidden: Updated audio hidden [B, T_a, D_a]
        """
        # Audio attends to Video (with residual)
        a2v_attn = self.audio_to_video(
            query=self.norm_a2v(audio_hidden),
            cross=video_hidden,
            video_positions=video_positions,
            audio_positions=audio_positions,
            unitime_rope=unitime_rope,
            key_padding_mask=video_mask
        )
        audio_hidden = audio_hidden + a2v_attn
        
        # Video attends to Audio (with residual)
        v2a_attn = self.video_to_audio(
            query=self.norm_v2a(video_hidden),
            cross=audio_hidden,
            video_positions=video_positions,
            audio_positions=audio_positions,
            unitime_rope=unitime_rope,
            key_padding_mask=audio_mask
        )
        video_hidden = video_hidden + v2a_attn
        
        return video_hidden, audio_hidden


class ProjectionLayer(nn.Module):
    """
    Projection layer for modality dimension alignment.
    
    Used when video_dim != audio_dim for cross-attention.
    """
    
    def __init__(self, input_dim: int, output_dim: int, use_layer_norm: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.layer_norm = nn.LayerNorm(input_dim) if use_layer_norm else nn.Identity()
        self.projection = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        return self.projection(x)