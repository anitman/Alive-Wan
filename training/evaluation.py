"""
Evaluation Metrics for Alive-Wan2X

Implements the following metrics:

  FVD  (Fréchet Video Distance)   — video quality vs. real videos
  FAD  (Fréchet Audio Distance)   — audio quality vs. real audio
  AV-Sync Score                   — audio-visual temporal alignment

External dependencies (install separately):
  FVD:  pip install pytorch-i3d   (or torchmetrics[video])
  FAD:  pip install fadtk          (https://github.com/LAION-AI/CLAP)
  Sync: pip install synchformer    (or SyncNet weights)
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Fréchet distance computation (shared)
# ---------------------------------------------------------------------------

def _frechet_distance(mu1: torch.Tensor, sigma1: torch.Tensor,
                      mu2: torch.Tensor, sigma2: torch.Tensor) -> float:
    """
    Compute Fréchet distance between two multivariate Gaussians.

    FD = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2 * sqrt(sigma1 @ sigma2))

    Args:
        mu1, mu2: Mean vectors [D].
        sigma1, sigma2: Covariance matrices [D, D].

    Returns:
        FD score (float, lower is better).
    """
    diff = mu1 - mu2
    term1 = diff.dot(diff).item()

    # Numerically stable matrix square root via eigendecomposition
    product = sigma1 @ sigma2
    eigvals, eigvecs = torch.linalg.eigh(product.clamp(min=0))
    sqrt_product = eigvecs @ torch.diag(eigvals.clamp(min=0).sqrt()) @ eigvecs.T

    term2 = (sigma1 + sigma2 - 2.0 * sqrt_product).trace().item()
    return term1 + term2


def _feature_stats(features: torch.Tensor):
    """
    Compute mean and covariance of a feature matrix.

    Args:
        features: [N, D] feature matrix.

    Returns:
        mu: [D]
        sigma: [D, D]
    """
    mu    = features.mean(dim=0)
    diff  = features - mu
    sigma = (diff.T @ diff) / (features.shape[0] - 1)
    return mu, sigma


# ---------------------------------------------------------------------------
# FVD — Fréchet Video Distance
# ---------------------------------------------------------------------------

class FVDEvaluator:
    """
    Fréchet Video Distance using an I3D feature extractor.

    Lower FVD = better video quality / diversity.

    Requirements:
        pip install torch-fidelity   (includes I3D)
        -- or --
        pip install torchmetrics[video]
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._i3d: Optional[nn.Module] = None

    def _load_i3d(self) -> nn.Module:
        """Lazy-load I3D network for video feature extraction."""
        if self._i3d is not None:
            return self._i3d
        try:
            from torchmetrics.image.fid import NoTrainInceptionV3  # type: ignore
            # FVD typically uses I3D; fall back to a mean-frame FID proxy if unavailable
            raise ImportError("use I3D path")
        except (ImportError, Exception):
            pass
        try:
            import torch.hub
            i3d = torch.hub.load("hassony2/kinetics_i3d_pytorch",
                                  "i3d_flow_imagenet", pretrained=True)
            i3d.to(self.device).eval().requires_grad_(False)
            self._i3d = i3d
            return i3d
        except Exception:
            # Fallback: dummy extractor (returns random features for structural testing)
            import warnings
            warnings.warn(
                "I3D feature extractor not available. FVD will not be meaningful. "
                "Install with: pip install torch-hub",
                RuntimeWarning,
            )
            self._i3d = _DummyFeatureExtractor(output_dim=400)
            return self._i3d

    @torch.no_grad()
    def extract_features(self, videos: torch.Tensor) -> torch.Tensor:
        """
        Extract I3D features from a batch of videos.

        Args:
            videos: [B, T, H, W, 3] float32 in [0, 1].

        Returns:
            features: [B, D] feature vectors.
        """
        i3d = self._load_i3d()
        # I3D expects [B, C, T, H, W] float in [0, 1]
        x = videos.permute(0, 4, 1, 2, 3).to(self.device)
        # Resize to I3D input size (224×224)
        B, C, T, H, W = x.shape
        if H != 224 or W != 224:
            x = x.reshape(B * C, T, H, W)
            x = F.interpolate(x.unsqueeze(0), size=(T, 224, 224),
                              mode="trilinear", align_corners=False).squeeze(0)
            x = x.reshape(B, C, T, 224, 224)
        feats = i3d(x)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        return feats.reshape(B, -1)

    def compute(
        self,
        generated_videos: torch.Tensor,
        real_videos: torch.Tensor,
    ) -> float:
        """
        Compute FVD between generated and real video sets.

        Args:
            generated_videos: [N, T, H, W, 3].
            real_videos: [N, T, H, W, 3].

        Returns:
            FVD score (float).
        """
        gen_feats  = self.extract_features(generated_videos)
        real_feats = self.extract_features(real_videos)

        mu_gen,  sigma_gen  = _feature_stats(gen_feats.float())
        mu_real, sigma_real = _feature_stats(real_feats.float())

        return _frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)


# ---------------------------------------------------------------------------
# FAD — Fréchet Audio Distance
# ---------------------------------------------------------------------------

class FADEvaluator:
    """
    Fréchet Audio Distance using VGGish or CLAP audio features.

    Lower FAD = better audio quality / diversity.

    Requirements (choose one):
        pip install fadtk          (LAION-CLAP based, recommended)
        pip install torchaudio     (VGGish fallback)
    """

    def __init__(self, device: str = "cuda", sample_rate: int = 16000):
        self.device = device
        self.sample_rate = sample_rate
        self._model: Optional[nn.Module] = None

    def _load_model(self) -> nn.Module:
        """Lazy-load audio feature extractor."""
        if self._model is not None:
            return self._model
        try:
            # CLAP-based FAD (preferred)
            from fadtk import FAD  # type: ignore
            self._model = FAD(ml="clap-2023", audio_load_worker=0)
            return self._model
        except ImportError:
            pass
        try:
            # VGGish fallback
            import torchaudio.transforms as T
            model = torch.hub.load("harritaylor/torchvggish", "vggish")
            model.to(self.device).eval().requires_grad_(False)
            self._model = model
            return model
        except Exception:
            import warnings
            warnings.warn(
                "Audio feature extractor not available. FAD will not be meaningful.",
                RuntimeWarning,
            )
            self._model = _DummyFeatureExtractor(output_dim=128)
            return self._model

    @torch.no_grad()
    def extract_features(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Extract audio features.

        Args:
            waveforms: [B, T_audio] float32 in [-1, 1].

        Returns:
            features: [B, D].
        """
        model = self._load_model()
        # Add channel dim for VGGish: [B, 1, T]
        x = waveforms.unsqueeze(1).to(self.device)
        feats = model(x)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        return feats.reshape(waveforms.shape[0], -1)

    def compute(
        self,
        generated_audio: torch.Tensor,
        real_audio: torch.Tensor,
    ) -> float:
        """
        Compute FAD between generated and real audio sets.

        Args:
            generated_audio: [N, T_audio].
            real_audio: [N, T_audio].

        Returns:
            FAD score (float).
        """
        gen_feats  = self.extract_features(generated_audio)
        real_feats = self.extract_features(real_audio)

        mu_gen,  sigma_gen  = _feature_stats(gen_feats.float())
        mu_real, sigma_real = _feature_stats(real_feats.float())

        return _frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)


# ---------------------------------------------------------------------------
# AV-Sync Score
# ---------------------------------------------------------------------------

class AVSyncEvaluator:
    """
    Audio-Visual synchronisation score.

    Uses cross-modal cosine similarity between audio and video features
    at aligned time steps. Higher score = better sync.

    A production implementation would use SyncNet or Synchformer.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

    @torch.no_grad()
    def compute(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        video_fps: int = 24,
        audio_sr: int = 16000,
    ) -> float:
        """
        Compute average cosine similarity between temporally aligned AV features.

        Args:
            video: [B, T_v, H, W, 3] video frames.
            audio: [B, T_audio] waveform.
            video_fps: Video frame rate.
            audio_sr: Audio sample rate.

        Returns:
            sync_score: Average cosine similarity in [-1, 1] (higher = better sync).
        """
        B, T_v, H, W, C = video.shape
        T_a = audio.shape[-1]

        # Compute simple mean features per time window
        # Each video frame corresponds to audio_sr/video_fps audio samples
        samples_per_frame = audio_sr // video_fps

        scores = []
        for t in range(min(T_v, T_a // samples_per_frame)):
            frame = video[:, t]               # [B, H, W, 3]
            frame_feat = frame.mean(dim=(1, 2))   # [B, 3] — trivial; replace with CNN

            start = t * samples_per_frame
            end   = start + samples_per_frame
            audio_chunk = audio[:, start:end]          # [B, samples_per_frame]
            audio_feat  = audio_chunk.abs().mean(dim=1, keepdim=True)  # [B, 1]

            # Pad frame_feat to match dim
            audio_feat = audio_feat.expand(-1, frame_feat.shape[-1])
            sim = F.cosine_similarity(frame_feat, audio_feat, dim=-1)
            scores.append(sim.mean().item())

        return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Combined evaluation runner
# ---------------------------------------------------------------------------

class EvaluationMetrics:
    """
    Combined evaluation runner for all AV generation metrics.

    Usage:
        metrics = EvaluationMetrics(device="cuda")
        scores = metrics.evaluate(generated, real)
        # scores: {"video_fvd": ..., "audio_fad": ..., "av_sync": ...}
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.fvd_eval  = FVDEvaluator(device=device)
        self.fad_eval  = FADEvaluator(device=device)
        self.sync_eval = AVSyncEvaluator(device=device)

    def compute_video_fvd(
        self,
        generated_videos: torch.Tensor,
        real_videos: torch.Tensor,
    ) -> float:
        """Compute Video FVD. Returns float (lower = better)."""
        return self.fvd_eval.compute(generated_videos, real_videos)

    def compute_audio_fad(
        self,
        generated_audio: torch.Tensor,
        real_audio: torch.Tensor,
    ) -> float:
        """Compute Audio FAD. Returns float (lower = better)."""
        return self.fad_eval.compute(generated_audio, real_audio)

    def compute_sync_score(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
    ) -> float:
        """Compute AV-Sync score. Returns float (higher = better)."""
        return self.sync_eval.compute(video, audio)

    def evaluate(
        self,
        generated: Dict[str, torch.Tensor],
        real: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics.

        Args:
            generated: Dict with keys "video" [N,T,H,W,3] and "audio" [N,T_a].
            real: Dict with the same keys, for real samples.

        Returns:
            Dict of metric names → float scores.
        """
        results: Dict[str, float] = {}

        if "video" in generated and "video" in real:
            results["video_fvd"] = self.compute_video_fvd(
                generated["video"], real["video"]
            )

        if "audio" in generated and "audio" in real:
            results["audio_fad"] = self.compute_audio_fad(
                generated["audio"], real["audio"]
            )

        if "video" in generated and "audio" in generated:
            results["av_sync"] = self.compute_sync_score(
                generated["video"], generated["audio"]
            )

        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _DummyFeatureExtractor(nn.Module):
    """Returns random features when real extractors are unavailable (for CI/testing)."""

    def __init__(self, output_dim: int = 400):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return torch.randn(B, self.output_dim, device=x.device)
