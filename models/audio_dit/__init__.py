"""
Audio DiT Module

Audio DiT implementation for Alive-Wan2X.
"""

from .audio_dit import AudioDiT, AudioDiTBlock, create_audio_dit
from .audio_vae import (
    AudioVAEBase,
    WavVAE,
    StableAudioVAE,
    DACVAE,
    create_audio_vae
)
from .dual_conditioning import DualConditioning

__all__ = [
    # Audio DiT
    "AudioDiT",
    "AudioDiTBlock",
    "create_audio_dit",
    
    # Audio VAE
    "AudioVAEBase",
    "WavVAE",
    "StableAudioVAE",
    "DACVAE",
    "create_audio_vae",
    
    # Conditioning
    "DualConditioning",
]