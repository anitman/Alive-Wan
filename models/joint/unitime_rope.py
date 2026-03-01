"""
UniTemp-RoPE: Unified Time Rotary Positional Encoding

Maps video and audio token indices to shared physical time coordinate system
for proper temporal alignment in cross-modal attention.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class UniTempRoPE(nn.Module):
    """
    Unified Time Rotary Positional Encoding.
    
    Key concept:
    - Video and audio have different temporal resolutions
    - Map both to shared physical time (seconds)
    - Apply RoPE based on physical time for proper alignment
    
    Example:
    - Video: 24fps, VAE 4:1 compression → each token = 4/24 = 1/6 second
    - Audio: 16kHz, VAE 320:1 compression → each token = 320/16000 = 1/50 second
    
    In attention, tokens at the same physical time naturally get higher alignment.
    """
    
    def __init__(
        self,
        dim: int,
        video_fps: int = 24,
        video_temporal_compress: int = 4,
        audio_sr: int = 16000,
        audio_temporal_compress: int = 320,
        max_period: int = 10000
    ):
        """
        Initialize UniTemp-RoPE.
        
        Args:
            dim: RoPE dimension (typically head_dim)
            video_fps: Video frame rate
            video_temporal_compress: Video VAE temporal compression ratio
            audio_sr: Audio sample rate
            audio_temporal_compress: Audio VAE temporal compression ratio
            max_period: Maximum period for RoPE frequencies
        """
        super().__init__()
        
        self.dim = dim
        self.video_fps = video_fps
        self.video_temporal_compress = video_temporal_compress
        self.audio_sr = audio_sr
        self.audio_temporal_compress = audio_temporal_compress
        self.max_period = max_period
        
        # Calculate time delta per token (in seconds)
        # Video: each token represents video_temporal_compress frames
        self.video_dt = video_temporal_compress / video_fps  # e.g., 4/24 = 1/6 sec
        
        # Audio: each token represents audio_temporal_compress samples
        self.audio_dt = audio_temporal_compress / audio_sr  # e.g., 320/16000 = 1/50 sec
        
        # RoPE frequencies
        # freqs = 1 / (10000^(2i/d)) for i in [0, d/2)
        freqs = 1.0 / (max_period ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        self.register_buffer("freqs", freqs, persistent=False)
        
        # Cache for positions (computed on first use)
        self._video_pos_cache = None
        self._audio_pos_cache = None
        self._cache_max_len = 0
    
    def get_video_physical_positions(
        self,
        num_tokens: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Get physical time positions for video tokens.
        
        Args:
            num_tokens: Number of video tokens
            device: Device for positions
            dtype: Data type
            
        Returns:
            positions: [num_tokens] physical time in seconds
        """
        device = device or self.freqs.device
        
        # Use cache if available
        if num_tokens <= self._cache_max_len and self._video_pos_cache is not None:
            return self._video_pos_cache[:num_tokens]
        
        # Compute positions: t_i = i * video_dt (seconds)
        indices = torch.arange(num_tokens, device=device, dtype=dtype)
        positions = indices * self.video_dt  # [T_video]
        
        # Update cache
        if num_tokens > self._cache_max_len:
            self._cache_max_len = num_tokens
            self._video_pos_cache = positions.clone()
        
        return positions
    
    def get_audio_physical_positions(
        self,
        num_tokens: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Get physical time positions for audio tokens.
        
        Args:
            num_tokens: Number of audio tokens
            device: Device for positions
            dtype: Data type
            
        Returns:
            positions: [num_tokens] physical time in seconds
        """
        device = device or self.freqs.device
        
        # Use cache if available
        if num_tokens <= self._cache_max_len and self._audio_pos_cache is not None:
            return self._audio_pos_cache[:num_tokens]
        
        # Compute positions: t_i = i * audio_dt (seconds)
        indices = torch.arange(num_tokens, device=device, dtype=dtype)
        positions = indices * self.audio_dt  # [T_audio]
        
        # Update cache
        if num_tokens > self._cache_max_len:
            self._cache_max_len = num_tokens
            self._audio_pos_cache = positions.clone()
        
        return positions
    
    def apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        video_positions: Optional[torch.Tensor] = None,
        audio_positions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query and key tensors.
        
        For cross-attention between modalities:
        - Apply video positions to video tokens
        - Apply audio positions to audio tokens
        
        Args:
            q: Query [B, heads, T, head_dim]
            k: Key [B, heads, T, head_dim]
            video_positions: Video positions [T_video] in seconds
            audio_positions: Audio positions [T_audio] in seconds
            
        Returns:
            q_rotated: RoPE-applied query
            k_rotated: RoPE-applied key
        """
        # Determine which positions to use
        # For self-attention, use positions based on input shape
        # For cross-attention, positions are explicitly provided
        
        if video_positions is not None:
            # Cross-attention: different positions for q and k
            q_pos = video_positions  # Query modality positions
            k_pos = audio_positions  # Key modality positions
        else:
            # Self-attention: same positions for both
            # Infer from tensor shape (assume video if q and k have same length)
            T = q.shape[2]
            q_pos = self.get_video_physical_positions(T, q.device, q.dtype)
            k_pos = q_pos
        
        # Reshape positions for broadcasting
        # q_pos: [T_q], k_pos: [T_k]
        # freqs: [dim/2]
        
        # Compute angles: angles = position * freqs
        # q_angles: [T_q, dim/2], k_angles: [T_k, dim/2]
        q_angles = torch.outer(q_pos, self.freqs.to(q_pos.dtype))  # [T_q, dim/2]
        k_angles = torch.outer(k_pos, self.freqs.to(k_pos.dtype))  # [T_k, dim/2]
        
        # Compute cos/sin for RoPE: [T, dim/2]
        q_cos = q_angles.cos()  # [T_q, dim/2]
        q_sin = q_angles.sin()  # [T_q, dim/2]
        k_cos = k_angles.cos()  # [T_k, dim/2]
        k_sin = k_angles.sin()  # [T_k, dim/2]

        # Reshape q/k for paired rotation
        # q: [B, heads, T, dim] → [B, heads, T, dim/2, 2]
        q_reshape = q.reshape(*q.shape[:3], -1, 2)
        k_reshape = k.reshape(*k.shape[:3], -1, 2)

        # Extract even/odd components: [B, heads, T, dim/2]
        q_even = q_reshape[..., 0]
        q_odd = q_reshape[..., 1]

        # Apply RoPE rotation with proper broadcasting
        # q_cos/q_sin are [T_q, dim/2], need to broadcast to [1, 1, T_q, dim/2]
        q_rotated_even = q_even * q_cos - q_odd * q_sin
        q_rotated_odd = q_even * q_sin + q_odd * q_cos
        q_rotated = torch.stack([q_rotated_even, q_rotated_odd], dim=-1).flatten(-2)

        k_even = k_reshape[..., 0]
        k_odd = k_reshape[..., 1]

        k_rotated_even = k_even * k_cos - k_odd * k_sin
        k_rotated_odd = k_even * k_sin + k_odd * k_cos
        k_rotated = torch.stack([k_rotated_even, k_rotated_odd], dim=-1).flatten(-2)
        
        return q_rotated.to(q.dtype), k_rotated.to(k.dtype)
    
    def get_time_alignment_matrix(
        self,
        num_video_tokens: int,
        num_audio_tokens: int,
        tolerance: float = 0.1
    ) -> torch.Tensor:
        """
        Get time alignment matrix showing which tokens are temporally aligned.
        
        Args:
            num_video_tokens: Number of video tokens
            num_audio_tokens: Number of audio tokens
            tolerance: Time tolerance in seconds for alignment
            
        Returns:
            alignment: [T_video, T_audio] binary alignment matrix
        """
        video_pos = self.get_video_physical_positions(num_video_tokens)
        audio_pos = self.get_audio_physical_positions(num_audio_tokens)
        
        # Compute time difference matrix
        time_diff = torch.abs(
            video_pos.unsqueeze(1) - audio_pos.unsqueeze(0)
        )  # [T_v, T_a]
        
        # Binary alignment (1 if within tolerance)
        alignment = (time_diff <= tolerance).float()
        
        return alignment


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> torch.Tensor:
    """
    Apply rotary position embedding.
    
    Args:
        x: Input tensor [B, heads, T, dim]
        cos: Cosine embeddings [T, dim/2] or broadcastable
        sin: Sine embeddings [T, dim/2] or broadcastable
        
    Returns:
        x_rotated: RoPE-applied tensor
    """
    # Split into even and odd components
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    
    # Apply rotation
    # x_rotated_even = x_even * cos - x_odd * sin
    # x_rotated_odd = x_even * sin + x_odd * cos
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos
    
    # Concatenate
    x_rotated = torch.stack([rotated_even, rotated_odd], dim=-1).flatten(-2)
    
    return x_rotated