"""
Audio-Video Dataset for Alive-Wan2X

Loads paired audio-video clips with text captions for joint AV training.

Expected annotation format (JSON Lines, one sample per line):
    {
        "video_path": "path/to/video.mp4",
        "audio_path": "path/to/audio.wav",    (optional; extracted from video if absent)
        "caption": "A cat playing a piano.",
        "speech_transcription": "Hello world", (optional)
        "audio_description": "Soft piano music" (optional)
    }

Alternatively, a single JSON array file is also supported.

The dataset returns pre-processed tensors ready for the trainer (no VAE encoding).
VAE encoding happens inside the training loop to support online augmentation.
"""

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Video / Audio loading helpers
# ---------------------------------------------------------------------------

def _load_video_frames(
    path: str,
    num_frames: int,
    fps: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Load and uniformly sample ``num_frames`` frames from a video file.

    Uses ``decord`` when available (faster), falls back to
    ``torchvision.io.read_video``.

    Args:
        path: Path to video file.
        num_frames: Number of frames to sample.
        fps: Target frame rate (used for time-based sampling).
        height: Target frame height (pixels).
        width: Target frame width (pixels).

    Returns:
        frames: [T, H, W, 3] float32 tensor, values in [0, 1].
    """
    try:
        import decord
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(path, ctx=decord.cpu(0), num_threads=1)
        total = len(vr)
        # Uniform frame indices
        indices = _uniform_indices(total, num_frames)
        frames = vr.get_batch(indices).float() / 255.0  # [T, H_orig, W_orig, 3]
    except ImportError:
        import torchvision
        frames_raw, _, _ = torchvision.io.read_video(path, pts_unit="sec")
        # frames_raw: [T, H, W, 3] uint8
        total = frames_raw.shape[0]
        indices = _uniform_indices(total, num_frames)
        frames = frames_raw[indices].float() / 255.0

    # Resize to target resolution
    if frames.shape[1] != height or frames.shape[2] != width:
        # [T, H, W, 3] → [T, 3, H, W] for interpolate → [T, H, W, 3]
        frames = frames.permute(0, 3, 1, 2)
        frames = F.interpolate(frames, size=(height, width), mode="bilinear",
                               align_corners=False)
        frames = frames.permute(0, 2, 3, 1)

    return frames  # [T, H, W, 3]


def _load_audio_waveform(
    path: str,
    target_sr: int,
    duration: float,
) -> torch.Tensor:
    """
    Load and normalise an audio waveform.

    Args:
        path: Path to audio file (.wav / .mp3 / .flac / etc.).
        target_sr: Target sample rate (Hz).
        duration: Target duration in seconds; waveform is padded or trimmed.

    Returns:
        waveform: [T_audio] float32 mono waveform, values in [-1, 1].
    """
    import torchaudio

    waveform, sr = torchaudio.load(path)  # [C, T_orig]

    # Down-mix to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # [1, T]

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    target_samples = int(duration * target_sr)
    T = waveform.shape[-1]

    if T >= target_samples:
        waveform = waveform[..., :target_samples]
    else:
        pad = target_samples - T
        waveform = F.pad(waveform, (0, pad))

    return waveform.squeeze(0)  # [T_audio]


def _extract_audio_from_video(video_path: str, target_sr: int, duration: float) -> torch.Tensor:
    """
    Extract and return the audio track embedded in a video file.

    Args:
        video_path: Path to the video file.
        target_sr: Target sample rate.
        duration: Target duration in seconds.

    Returns:
        waveform: [T_audio] float32 mono waveform.
    """
    import torchaudio

    waveform, sr = torchaudio.load(video_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)

    target_samples = int(duration * target_sr)
    T = waveform.shape[-1]
    if T >= target_samples:
        waveform = waveform[..., :target_samples]
    else:
        waveform = F.pad(waveform, (0, target_samples - T))

    return waveform.squeeze(0)


def _uniform_indices(total: int, num_frames: int) -> List[int]:
    """Return ``num_frames`` uniformly spaced integer indices in [0, total)."""
    if total <= num_frames:
        indices = list(range(total)) + [total - 1] * (num_frames - total)
    else:
        step = total / num_frames
        indices = [int(i * step) for i in range(num_frames)]
    return indices


# ---------------------------------------------------------------------------
# AV Dataset
# ---------------------------------------------------------------------------

class AVDataset(Dataset):
    """
    Paired Audio-Video Dataset for Alive-Wan2X.

    Returns dicts with raw pixel/waveform tensors; the trainer (or a
    collation function) is responsible for VAE encoding.

    Supported data sources (via annotation JSON):
        VGGSound, AudioSet-AV, AudioCaps, LRS3, Custom

    Args:
        data_root: Root directory. Paths in the annotation file may be
                   relative to this directory or absolute.
        annotation_file: Path to the annotation file. If None, the dataset
                         searches for ``data_root/annotations.json`` or
                         ``data_root/annotations.jsonl``.
        max_video_frames: Number of frames to sample per clip.
        video_height: Frame height in pixels.
        video_width: Frame width in pixels.
        fps: Video frame rate.
        audio_duration: Audio clip duration in seconds.
        sample_rate: Audio sample rate in Hz.
        transform: Optional callable applied to the video tensor [T, H, W, 3].
        audio_transform: Optional callable applied to the audio waveform [T_a].
    """

    def __init__(
        self,
        data_root: str,
        annotation_file: Optional[str] = None,
        max_video_frames: int = 128,
        video_height: int = 480,
        video_width: int = 854,
        fps: int = 24,
        audio_duration: float = 5.0,
        sample_rate: int = 16000,
        transform: Optional[Callable] = None,
        audio_transform: Optional[Callable] = None,
    ):
        super().__init__()

        self.data_root = data_root
        self.max_video_frames = max_video_frames
        self.video_height = video_height
        self.video_width = video_width
        self.fps = fps
        self.audio_duration = audio_duration
        self.sample_rate = sample_rate
        self.transform = transform
        self.audio_transform = audio_transform

        # Resolve annotation file
        if annotation_file is None:
            for candidate in ["annotations.json", "annotations.jsonl"]:
                candidate_path = os.path.join(data_root, candidate)
                if os.path.isfile(candidate_path):
                    annotation_file = candidate_path
                    break

        self.samples: List[Dict[str, Any]] = []
        if annotation_file is not None and os.path.isfile(annotation_file):
            self.samples = self._load_samples(annotation_file)
        else:
            # Fallback: auto-discover .mp4 files under data_root
            self.samples = self._discover_videos(data_root)

    # ---------------------------------------------------------------------- #
    # Annotation loading
    # ---------------------------------------------------------------------- #

    def _load_samples(self, annotation_file: str) -> List[Dict[str, Any]]:
        """
        Load sample metadata from a JSON or JSON-Lines annotation file.

        Each entry must have at least ``video_path`` and ``caption``.
        ``audio_path`` is optional; if absent, audio is extracted from the video.

        Returns:
            List of sample dicts.
        """
        samples = []
        with open(annotation_file, "r", encoding="utf-8") as f:
            # Support both JSON array and JSON-Lines formats
            content = f.read().strip()
            if content.startswith("["):
                data = json.loads(content)
            else:
                data = [json.loads(line) for line in content.splitlines() if line.strip()]

        for entry in data:
            # Resolve relative paths
            video_path = entry.get("video_path", "")
            if not os.path.isabs(video_path):
                video_path = os.path.join(self.data_root, video_path)

            audio_path = entry.get("audio_path", None)
            if audio_path and not os.path.isabs(audio_path):
                audio_path = os.path.join(self.data_root, audio_path)

            if not os.path.isfile(video_path):
                continue  # skip missing files

            samples.append({
                "video_path": video_path,
                "audio_path": audio_path,
                "caption": entry.get("caption", ""),
                "speech_transcription": entry.get("speech_transcription", None),
                "audio_description": entry.get("audio_description", None),
            })

        return samples

    def _discover_videos(self, root: str) -> List[Dict[str, Any]]:
        """
        Fallback: discover all .mp4 files under ``root`` and create minimal
        sample dicts (caption defaults to filename stem).
        """
        samples = []
        for path in Path(root).rglob("*.mp4"):
            samples.append({
                "video_path": str(path),
                "audio_path": None,
                "caption": path.stem.replace("_", " "),
                "speech_transcription": None,
                "audio_description": None,
            })
        return samples

    # ---------------------------------------------------------------------- #
    # Dataset interface
    # ---------------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Return one training sample.

        Returns
        -------
        Dict with keys:
            video       : [T, H, W, 3] float32, values in [0, 1]
            audio       : [T_audio] float32, values in [-1, 1]
            text        : str — visual scene caption
            speech_text : str or None — speech transcript
            audio_prompt: str or None — audio style/environment description
        """
        sample = self.samples[idx]

        try:
            video = _load_video_frames(
                sample["video_path"],
                num_frames=self.max_video_frames,
                fps=self.fps,
                height=self.video_height,
                width=self.video_width,
            )
        except Exception:
            # Return a black frame sequence on loading failure
            video = torch.zeros(self.max_video_frames, self.video_height,
                                self.video_width, 3)

        try:
            audio_path = sample["audio_path"] or sample["video_path"]
            audio = _load_audio_waveform(
                audio_path,
                target_sr=self.sample_rate,
                duration=self.audio_duration,
            )
        except Exception:
            audio = torch.zeros(int(self.audio_duration * self.sample_rate))

        # Apply optional transforms (augmentation, normalisation, etc.)
        # Note: do NOT apply pitch shift or time stretch (breaks AV sync)
        if self.transform is not None:
            video = self.transform(video)
        if self.audio_transform is not None:
            audio = self.audio_transform(audio)

        return {
            "video": video,                                   # [T, H, W, 3]
            "audio": audio,                                   # [T_audio]
            "text": sample["caption"],
            "speech_text": sample["speech_transcription"],
            "audio_prompt": sample["audio_description"],
        }


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def av_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for DataLoader.

    Stacks tensors and collects string fields into lists.

    Args:
        batch: List of dicts from AVDataset.__getitem__.

    Returns:
        Collated batch dict.
    """
    return {
        "video": torch.stack([b["video"] for b in batch]),        # [B, T, H, W, 3]
        "audio": torch.stack([b["audio"] for b in batch]),        # [B, T_audio]
        "text": [b["text"] for b in batch],
        "speech_text": [b["speech_text"] for b in batch],
        "audio_prompt": [b["audio_prompt"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Audio-only Dataset (for Audio DiT pre-training)
# ---------------------------------------------------------------------------

class AudioDataset(Dataset):
    """
    Pure audio dataset for Audio DiT pre-training or fine-tuning.

    Supports music, speech, and environmental sounds.

    Annotation format (JSON Lines):
        {"audio_path": "...", "caption": "...", "speech": "..."}
    """

    def __init__(
        self,
        data_root: str,
        annotation_file: Optional[str] = None,
        audio_duration: float = 5.0,
        sample_rate: int = 16000,
        audio_transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.audio_duration = audio_duration
        self.sample_rate = sample_rate
        self.audio_transform = audio_transform

        self.samples: List[Dict[str, Any]] = []
        if annotation_file and os.path.isfile(annotation_file):
            with open(annotation_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.samples.append(json.loads(line))
        else:
            # Auto-discover audio files
            for ext in ("*.wav", "*.flac", "*.mp3"):
                for path in Path(data_root).rglob(ext):
                    self.samples.append({
                        "audio_path": str(path),
                        "caption": path.stem.replace("_", " "),
                        "speech": None,
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        audio_path = sample["audio_path"]
        if not os.path.isabs(audio_path):
            audio_path = os.path.join(self.data_root, audio_path)

        try:
            audio = _load_audio_waveform(audio_path, self.sample_rate, self.audio_duration)
        except Exception:
            audio = torch.zeros(int(self.audio_duration * self.sample_rate))

        if self.audio_transform is not None:
            audio = self.audio_transform(audio)

        return {
            "audio": audio,
            "text": sample.get("caption", ""),
            "speech_text": sample.get("speech", None),
        }
