"""
Audio VAE Wrappers for Alive-Wan2X

Provides a unified AudioVAEBase interface for three backend implementations:

  WavVAE       — Custom waveform VAE trained jointly with Audio DiT.
                 Weights are released alongside this project.
                 32 latent channels, 320:1 temporal compression at 16 kHz.

  StableAudioVAE — Wraps the VAE from Stability AI's stable-audio-open-1.0.
                   Install: pip install diffusers accelerate
                   Load:    from diffusers import StableAudioPipeline

  DACVAE       — Wraps the Descript Audio Codec.
                   Install: pip install descript-audio-codec
                   GitHub:  https://github.com/descriptinc/descript-audio-codec

All implementations follow the [B, C, T_latent] latent convention.
Waveforms are [B, 1, T_audio] (mono, float32, in [-1, 1]).
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class AudioVAEBase(ABC, nn.Module):
    """
    Unified interface for Audio VAE backends.

    All subclasses must implement ``encode`` and ``decode``.

    Latent convention:
        encode input:   [B, 1, T_audio]  — mono waveform
        encode output:  [B, C, T_latent] — compressed latent
        decode input:   [B, C, T_latent]
        decode output:  [B, 1, T_audio]
    """

    def __init__(self, temporal_compress: int, channels: int, sample_rate: int = 16000):
        """
        Args:
            temporal_compress: Temporal compression ratio (e.g. 320).
            channels: Number of latent channels (e.g. 32).
            sample_rate: Audio sample rate in Hz (e.g. 16000).
        """
        super().__init__()
        self._temporal_compress = temporal_compress
        self._channels = channels
        self._sample_rate = sample_rate

    @property
    def temporal_compression_ratio(self) -> int:
        """Temporal compression ratio (samples → latent frames)."""
        return self._temporal_compress

    @property
    def channels(self) -> int:
        """Number of latent channels."""
        return self._channels

    @property
    def sample_rate(self) -> int:
        """Audio sample rate the VAE was trained with."""
        return self._sample_rate

    def samples_to_latent_length(self, num_samples: int) -> int:
        """Compute latent sequence length from raw sample count."""
        return num_samples // self._temporal_compress

    @abstractmethod
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Encode waveform to latent.

        Args:
            waveform: [B, 1, T_audio] float32 in [-1, 1].

        Returns:
            latent: [B, C, T_latent].
        """

    @abstractmethod
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to waveform.

        Args:
            latent: [B, C, T_latent].

        Returns:
            waveform: [B, 1, T_audio] float32 in [-1, 1].
        """

    def forward(self, x: torch.Tensor, mode: str = "encode") -> torch.Tensor:
        """Unified forward: mode='encode' or 'decode'."""
        if mode == "encode":
            return self.encode(x)
        elif mode == "decode":
            return self.decode(x)
        raise ValueError(f"Unknown mode '{mode}'. Expected 'encode' or 'decode'.")

    @torch.no_grad()
    def encode_no_grad(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode without gradient tracking (for inference)."""
        return self.encode(waveform)

    @torch.no_grad()
    def decode_no_grad(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode without gradient tracking (for inference)."""
        return self.decode(latent)


# ---------------------------------------------------------------------------
# WavVAE — custom waveform VAE
# ---------------------------------------------------------------------------

class WavVAE(AudioVAEBase):
    """
    Custom Waveform VAE used in the Alive architecture.

    Architecture overview:
      Encoder: 1D conv stack (strided) + residual blocks → mean/logvar
      Decoder: Transposed conv stack (upsampling) + residual blocks
      Compression: 320× at 16 kHz  →  50 latent frames per second

    Pretrained weights:
      Download from the project's HuggingFace repository and pass the
      path as ``pretrained_path``.

    Args:
        pretrained_path: Path to a .pt checkpoint with keys
            "encoder", "decoder" (or full state dict).
        temporal_compress: Temporal compression ratio (default 320).
        channels: Number of latent channels (default 32).
        sample_rate: Audio sample rate (default 16000).
        hidden_channels: Internal channel width (default 256).
        num_res_blocks: Number of residual blocks per stage (default 4).
    """

    def __init__(
        self,
        pretrained_path: Optional[str] = None,
        temporal_compress: int = 320,
        channels: int = 32,
        sample_rate: int = 16000,
        hidden_channels: int = 256,
        num_res_blocks: int = 4,
    ):
        super().__init__(temporal_compress, channels, sample_rate)

        # Build encoder and decoder networks
        self.encoder = _build_audio_encoder(
            in_channels=1,
            latent_channels=channels,
            hidden_channels=hidden_channels,
            temporal_compress=temporal_compress,
            num_res_blocks=num_res_blocks,
        )
        self.decoder = _build_audio_decoder(
            latent_channels=channels,
            out_channels=1,
            hidden_channels=hidden_channels,
            temporal_compress=temporal_compress,
            num_res_blocks=num_res_blocks,
        )

        if pretrained_path is not None:
            self._load_weights(pretrained_path)

    def _load_weights(self, path: str) -> None:
        state = torch.load(path, map_location="cpu")
        # Support both full state dict and sub-keyed dicts
        if "encoder" in state and "decoder" in state:
            self.encoder.load_state_dict(state["encoder"])
            self.decoder.load_state_dict(state["decoder"])
        else:
            self.load_state_dict(state, strict=False)

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [B, 1, T_audio]

        Returns:
            latent: [B, C, T_latent]
        """
        # Encoder outputs mean (we use the mean as the deterministic latent
        # during training; full VAE sampling can be added if needed)
        return self.encoder(waveform)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: [B, C, T_latent]

        Returns:
            waveform: [B, 1, T_audio]
        """
        return torch.tanh(self.decoder(latent))


# ---------------------------------------------------------------------------
# StableAudioVAE — Stability AI wrapper
# ---------------------------------------------------------------------------

class StableAudioVAE(AudioVAEBase):
    """
    Wrapper for the VAE from Stability AI's ``stable-audio-open-1.0``.

    Installation:
        pip install diffusers accelerate

    Download:
        The VAE is loaded automatically from HuggingFace when you supply the
        ``pretrained_path="stabilityai/stable-audio-open-1.0"`` (or a local
        clone of that repo).

    Note:
        Stable Audio VAE has 64 latent channels and operates at 44.1 kHz.
        If your model was designed for 32 channels / 16 kHz, use WavVAE instead.
    """

    def __init__(
        self,
        pretrained_path: str = "stabilityai/stable-audio-open-1.0",
        temporal_compress: int = 512,
        channels: int = 64,
        sample_rate: int = 44100,
    ):
        super().__init__(temporal_compress, channels, sample_rate)
        self._vae = None
        self._path = pretrained_path

    def load_model(self) -> None:
        """Load the VAE from HuggingFace (call once before using encode/decode)."""
        try:
            from diffusers import StableAudioPipeline
        except ImportError as e:
            raise ImportError(
                "StableAudioVAE requires diffusers. "
                "Install with: pip install diffusers accelerate"
            ) from e

        pipe = StableAudioPipeline.from_pretrained(self._path, torch_dtype=torch.float32)
        self._vae = pipe.vae
        self._vae.requires_grad_(False)
        self._vae.eval()

    def _check_loaded(self) -> None:
        if self._vae is None:
            raise RuntimeError(
                "StableAudioVAE.load_model() must be called before encode/decode."
            )

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [B, 1, T_audio] at self.sample_rate (44100 Hz).

        Returns:
            latent: [B, C, T_latent]
        """
        self._check_loaded()
        # Stable Audio VAE expects [B, 1, T]
        posterior = self._vae.encode(waveform)
        return posterior.latent_dist.mean  # deterministic encode

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: [B, C, T_latent]

        Returns:
            waveform: [B, 1, T_audio]
        """
        self._check_loaded()
        return self._vae.decode(latent).sample


# ---------------------------------------------------------------------------
# DACVAE — Descript Audio Codec wrapper
# ---------------------------------------------------------------------------

class DACVAE(AudioVAEBase):
    """
    Wrapper for the Descript Audio Codec (DAC).

    DAC provides a high-quality neural audio codec that can be used as a
    VAE-like encoder/decoder for latent diffusion.

    Installation:
        pip install descript-audio-codec

    Usage:
        vae = DACVAE(model_type="44khz")
        vae.load_model()
    """

    def __init__(
        self,
        pretrained_path: Optional[str] = None,
        model_type: str = "16khz",
        temporal_compress: int = 320,
        channels: int = 64,
        sample_rate: int = 16000,
    ):
        """
        Args:
            pretrained_path: Local path to a DAC .pth checkpoint,
                             or None to auto-download from HuggingFace.
            model_type: DAC model type: "16khz", "24khz", or "44khz".
            temporal_compress: Temporal compression ratio of the chosen model.
                16khz: 320×,  24khz: 320×,  44khz: 512×
            channels: Number of latent channels produced by the codec encoder.
            sample_rate: Audio sample rate matching the model type.
        """
        super().__init__(temporal_compress, channels, sample_rate)
        self._path = pretrained_path
        self._model_type = model_type
        self._model = None

    def load_model(self) -> None:
        """Load the DAC model (call once before encode/decode)."""
        try:
            import dac
        except ImportError as e:
            raise ImportError(
                "DACVAE requires descript-audio-codec. "
                "Install with: pip install descript-audio-codec"
            ) from e

        if self._path is not None:
            self._model = dac.DAC.load(self._path)
        else:
            self._model = dac.pretrained.get_model(self._model_type)

        self._model.requires_grad_(False)
        self._model.eval()

    def _check_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError(
                "DACVAE.load_model() must be called before encode/decode."
            )

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [B, 1, T_audio]

        Returns:
            latent: [B, C, T_latent]
                    DAC returns continuous embeddings from the encoder projection.
        """
        self._check_loaded()
        # DAC encode pipeline: preprocess → encoder → quantiser
        # We take the continuous encoder output (pre-quantisation) as the latent
        x = self._model.preprocess(waveform, self._sample_rate)
        z, _, _, _, _ = self._model.encode(x)
        return z

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: [B, C, T_latent]

        Returns:
            waveform: [B, 1, T_audio]
        """
        self._check_loaded()
        return self._model.decode(latent)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_audio_vae(
    vae_type: str = "wavvae",
    **kwargs,
) -> AudioVAEBase:
    """
    Factory function to create an AudioVAE.

    Args:
        vae_type: Backend type — "wavvae", "stable_audio", or "dac".
        **kwargs: Passed to the backend constructor.

    Returns:
        AudioVAEBase instance.

    Raises:
        ValueError: If ``vae_type`` is not recognised.
    """
    registry = {
        "wavvae":       WavVAE,
        "stable_audio": StableAudioVAE,
        "dac":          DACVAE,
    }

    if vae_type not in registry:
        raise ValueError(
            f"Unknown VAE type '{vae_type}'. Supported: {list(registry.keys())}"
        )

    return registry[vae_type](**kwargs)


# ---------------------------------------------------------------------------
# Internal network builders for WavVAE
# ---------------------------------------------------------------------------

class _ResBlock1d(nn.Module):
    """1-D residual block with two dilated convolutions."""

    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv1d(channels, channels, kernel_size=3,
                      padding=dilation, dilation=dilation),
            nn.LeakyReLU(0.1),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def _build_audio_encoder(
    in_channels: int,
    latent_channels: int,
    hidden_channels: int,
    temporal_compress: int,
    num_res_blocks: int,
) -> nn.Sequential:
    """
    Build a strided 1-D convolutional encoder.

    Compresses [B, 1, T] → [B, latent_channels, T // temporal_compress].

    Factorises ``temporal_compress`` as a product of stride-2 and stride-4
    downsampling convolutions until the desired ratio is achieved.
    """
    layers: list = [nn.Conv1d(in_channels, hidden_channels, kernel_size=7, padding=3)]

    # Compute downsampling strides (powers of 2 summing to log2(compress))
    compress = temporal_compress
    strides = []
    while compress > 1:
        if compress % 4 == 0:
            strides.append(4)
            compress //= 4
        else:
            strides.append(2)
            compress //= 2

    channels = hidden_channels
    for stride in strides:
        layers.append(nn.LeakyReLU(0.1))
        layers.append(nn.Conv1d(channels, channels * 2,
                                kernel_size=stride * 2,
                                stride=stride,
                                padding=stride // 2))
        channels *= 2
        for i in range(num_res_blocks):
            layers.append(_ResBlock1d(channels, dilation=2 ** i))

    layers.append(nn.LeakyReLU(0.1))
    layers.append(nn.Conv1d(channels, latent_channels, kernel_size=3, padding=1))
    return nn.Sequential(*layers)


def _build_audio_decoder(
    latent_channels: int,
    out_channels: int,
    hidden_channels: int,
    temporal_compress: int,
    num_res_blocks: int,
) -> nn.Sequential:
    """
    Build a transposed 1-D convolutional decoder (mirror of encoder).

    Upsamples [B, latent_channels, T_latent] → [B, 1, T_audio].
    """
    # Derive upsampling strides (same sequence as encoder strides, reversed)
    compress = temporal_compress
    strides = []
    while compress > 1:
        if compress % 4 == 0:
            strides.append(4)
            compress //= 4
        else:
            strides.append(2)
            compress //= 2
    strides = strides[::-1]

    channels = hidden_channels * (2 ** len(strides))
    layers: list = [nn.Conv1d(latent_channels, channels, kernel_size=3, padding=1)]

    for stride in strides:
        layers.append(nn.LeakyReLU(0.1))
        layers.append(nn.ConvTranspose1d(channels, channels // 2,
                                         kernel_size=stride * 2,
                                         stride=stride,
                                         padding=stride // 2))
        channels //= 2
        for i in range(num_res_blocks):
            layers.append(_ResBlock1d(channels, dilation=2 ** i))

    layers.append(nn.LeakyReLU(0.1))
    layers.append(nn.Conv1d(channels, out_channels, kernel_size=7, padding=3))
    return nn.Sequential(*layers)
