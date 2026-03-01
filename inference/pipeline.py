"""
Inference Pipeline for Alive-Wan2X

Implements Euler-method ODE sampling for Flow Matching (Rectified Flow):

  dx/dt = v_θ(x_t, t)

Starting from pure noise at t=1 and integrating to t=0:

  x_{t - Δt} = x_t - Δt * v_θ(x_t, t)

Multi-CFG guidance is applied at each step (three model evaluations):
  1. v_uncond  — no text, TA-CrossAttn disabled
  2. v_text    — text conditioning, TA-CrossAttn disabled
  3. v_cond    — full conditioning (text + TA-CrossAttn)

  v_guided = v_uncond
           + cfg_text   * (v_text - v_uncond)
           + cfg_mutual * (v_cond  - v_text)
"""

import argparse
import math
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
from tqdm import tqdm

from .multi_cfg import MultiCFG


# ---------------------------------------------------------------------------
# Euler ODE Sampler for Flow Matching
# ---------------------------------------------------------------------------

class EulerSampler:
    """
    Euler-method ODE sampler for rectified flow.

    Solves  dx/dt = v(x_t, t)  from t=1 (noise) to t=0 (data) using
    uniform step sizes.
    """

    def __init__(self, num_steps: int = 50):
        """
        Args:
            num_steps: Number of denoising steps.
        """
        self.num_steps = num_steps

    def get_timesteps(self, device: torch.device) -> torch.Tensor:
        """
        Return evenly spaced timesteps from 1 to 1/num_steps.

        Returns:
            timesteps: [num_steps] descending, e.g. [1.0, 0.98, ..., 0.02]
        """
        return torch.linspace(1.0, 1.0 / self.num_steps, self.num_steps, device=device)

    def step(
        self,
        x_t: torch.Tensor,
        v_t: torch.Tensor,
        t_curr: float,
        t_next: float,
    ) -> torch.Tensor:
        """
        Perform one Euler step.

        x_{t_next} = x_t + (t_next - t_curr) * v_t
                   = x_t - dt * v_t   (since t_next < t_curr)

        Args:
            x_t: Current noisy sample.
            v_t: Predicted velocity at (x_t, t_curr).
            t_curr: Current timestep (scalar float).
            t_next: Next timestep (scalar float, < t_curr).

        Returns:
            x_next: Updated sample.
        """
        dt = t_next - t_curr  # negative (moving towards t=0)
        return x_t + dt * v_t


# ---------------------------------------------------------------------------
# Main Inference Pipeline
# ---------------------------------------------------------------------------

class InferencePipeline:
    """
    End-to-end inference pipeline for Alive-Wan2X.

    Supports:
    - T2VA  (Text-to-Video-and-Audio)
    - I2VA  (Image-to-Video-and-Audio)
    - Multi-CFG guidance
    - Cascaded 480P → 1080P refinement
    """

    def __init__(
        self,
        model: nn.Module,
        video_vae: nn.Module,
        audio_vae: nn.Module,
        text_encoder: nn.Module,
        device: str = "cuda",
        num_steps: int = 50,
        cfg_scale_text: float = 7.5,
        cfg_scale_mutual: float = 2.0,
    ):
        """
        Args:
            model: JointAVDiT model (in eval mode, optionally with EMA weights).
            video_vae: Wan-VAE encoder/decoder (16-channel, 4:1 temporal).
            audio_vae: Audio VAE encoder/decoder (32-channel, 320:1 temporal).
            text_encoder: Qwen3Encoder (returns [B, N, 4096] embeddings).
            device: Device string.
            num_steps: Default number of ODE steps.
            cfg_scale_text: Text CFG scale (default 7.5).
            cfg_scale_mutual: Mutual attention CFG scale (default 2.0).
        """
        self.model = model.to(device).eval()
        self.video_vae = video_vae.to(device).eval()
        self.audio_vae = audio_vae.to(device).eval()
        self.text_encoder = text_encoder.to(device).eval()
        self.device = device

        self.sampler = EulerSampler(num_steps=num_steps)
        self.multi_cfg = MultiCFG(
            cfg_scale_text=cfg_scale_text,
            cfg_scale_mutual=cfg_scale_mutual,
        )

    # ---------------------------------------------------------------------- #
    # Text encoding helpers
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def _encode_text(
        self,
        texts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a list of text prompts.

        Returns:
            embeds: [B, N, D]
            mask:   [B, N]
        """
        return self.text_encoder.encode(texts)

    # ---------------------------------------------------------------------- #
    # Noise initialisation
    # ---------------------------------------------------------------------- #

    def _init_video_noise(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        Initialise pure Gaussian noise in the video latent space.

        Wan-VAE compresses spatially by 8× and temporally by 4×.

        Args:
            batch_size: B
            num_frames: Number of video frames.
            height, width: Frame resolution (pixels).

        Returns:
            noise: [B, 16, T_v, H_v, W_v]
        """
        T_v = math.ceil(num_frames / 4)
        H_v = height // 8
        W_v = width // 8
        return torch.randn(batch_size, 16, T_v, H_v, W_v, device=self.device)

    def _init_audio_noise(
        self,
        batch_size: int,
        audio_duration: float,
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """
        Initialise pure Gaussian noise in the audio latent space.

        WavVAE compresses 320:1 at 16 kHz.

        Args:
            batch_size: B
            audio_duration: Duration in seconds.
            sample_rate: Audio sample rate (Hz).

        Returns:
            noise: [B, 32, T_a]
        """
        T_a = math.ceil(audio_duration * sample_rate / 320)
        return torch.randn(batch_size, 32, T_a, device=self.device)

    # ---------------------------------------------------------------------- #
    # Core denoising loop (single pass, no CFG)
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def _denoise(
        self,
        video_noise: torch.Tensor,
        audio_noise: torch.Tensor,
        text_embeds: torch.Tensor,
        uncond_embeds: torch.Tensor,
        speech_text: Optional[torch.Tensor] = None,
        style_text: Optional[torch.Tensor] = None,
        speech_mask: Optional[torch.Tensor] = None,
        style_mask: Optional[torch.Tensor] = None,
        show_progress: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the full ODE denoising loop with multi-CFG.

        At each step, three model evaluations are performed:
          1. Unconditional  (empty text, no mutual cross-attn)
          2. Text-only      (text conditioning, no mutual cross-attn)
          3. Full           (text + mutual cross-attn)

        Args:
            video_noise: Initial video latent noise [B, 16, T_v, H_v, W_v].
            audio_noise: Initial audio latent noise [B, 32, T_a].
            text_embeds: Conditional text embeddings [B, N, D].
            uncond_embeds: Unconditional text embeddings [B, N, D] (empty prompt).
            speech_text: Speech text for audio path (optional).
            style_text: Style text for audio path (optional).
            speech_mask, style_mask: Padding masks (optional).
            show_progress: Display tqdm progress bar.

        Returns:
            video_latent: Denoised video latent.
            audio_latent: Denoised audio latent.
        """
        x_v = video_noise  # [B, 16, T_v, H_v, W_v]
        x_a = audio_noise  # [B, 32, T_a]

        timesteps = self.sampler.get_timesteps(self.device)  # [num_steps]
        dt = -1.0 / len(timesteps)                            # step size (negative)

        loop = tqdm(enumerate(timesteps), total=len(timesteps), disable=not show_progress,
                    desc="Denoising")

        for i, t_val in loop:
            t_next = float(timesteps[i + 1]) if i + 1 < len(timesteps) else 0.0
            t_curr = float(t_val)

            B = x_v.shape[0]
            t_tensor = torch.full((B,), t_curr, device=self.device, dtype=x_v.dtype)

            # ---- (1) Unconditional: empty text, mutual attention off ----
            # Disable TA-CrossAttn by zeroing out the audio latent
            zeros_a = torch.zeros_like(x_a)
            v_v_uncond, v_a_uncond, _ = self.model(
                x_v, zeros_a, uncond_embeds, t_tensor,
            )

            # ---- (2) Text-only: real text, mutual attention off ----
            v_v_text, v_a_text, _ = self.model(
                x_v, zeros_a, text_embeds, t_tensor,
            )

            # ---- (3) Full: real text + real audio (mutual attn on) ----
            v_v_cond, v_a_cond, _ = self.model(
                x_v, x_a, text_embeds, t_tensor,
                speech_text=speech_text,
                style_text=style_text,
                speech_mask=speech_mask,
                style_mask=style_mask,
            )

            # ---- Multi-CFG combination ----
            v_v_guided, v_a_guided = self.multi_cfg.apply_separate(
                video_uncond=v_v_uncond, video_text=v_v_text, video_cond=v_v_cond,
                audio_uncond=v_a_uncond, audio_text=v_a_text, audio_cond=v_a_cond,
            )

            # ---- Euler step ----
            x_v = self.sampler.step(x_v, v_v_guided, t_curr, t_next)
            x_a = self.sampler.step(x_a, v_a_guided, t_curr, t_next)

        return x_v, x_a

    # ---------------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_frames: int = 128,
        width: int = 854,
        height: int = 480,
        audio_duration: float = 5.0,
        num_steps: int = 50,
        cfg_scale_text: float = 7.5,
        cfg_scale_mutual: float = 2.0,
        seed: Optional[int] = None,
        speech_prompt: Optional[str] = None,
        style_prompt: Optional[str] = None,
        show_progress: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate video and audio from a text prompt (T2VA).

        Args:
            prompt: Positive text prompt describing the scene.
            negative_prompt: Negative prompt; defaults to empty string.
            num_frames: Number of video frames to generate.
            width: Video width in pixels.
            height: Video height in pixels.
            audio_duration: Duration of generated audio in seconds.
            num_steps: Number of ODE denoising steps.
            cfg_scale_text: Text CFG scale (default 7.5).
            cfg_scale_mutual: Mutual attention CFG scale (default 2.0).
            seed: Random seed for reproducibility.
            speech_prompt: Speech transcript for audio conditioning (optional).
            style_prompt: Style/environment description for audio (optional).
            show_progress: Show denoising progress bar.

        Returns:
            video: Decoded video frames [B, T, H, W, 3], values in [0, 1].
            audio: Decoded waveform [B, T_audio], values in [-1, 1].
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Update CFG scales
        self.multi_cfg.cfg_scale_text = cfg_scale_text
        self.multi_cfg.cfg_scale_mutual = cfg_scale_mutual
        self.sampler.num_steps = num_steps

        # Encode text prompts
        neg = negative_prompt or ""
        text_embeds, _ = self._encode_text([prompt])            # [1, N, D]
        uncond_embeds, _ = self._encode_text([neg])             # [1, N, D]

        speech_text = style_text = speech_mask = style_mask = None
        if speech_prompt is not None:
            speech_text, speech_mask = self._encode_text([speech_prompt])
        if style_prompt is not None:
            style_text, style_mask = self._encode_text([style_prompt])

        # Initialise noise
        video_noise = self._init_video_noise(1, num_frames, height, width)
        audio_noise = self._init_audio_noise(1, audio_duration)

        # Denoising loop
        video_latent, audio_latent = self._denoise(
            video_noise, audio_noise,
            text_embeds, uncond_embeds,
            speech_text=speech_text,
            style_text=style_text,
            speech_mask=speech_mask,
            style_mask=style_mask,
            show_progress=show_progress,
        )

        # Decode to pixel space
        video = self.video_vae.decode(video_latent)   # [B, 3, T, H, W] or [B, T, H, W, 3]
        audio = self.audio_vae.decode(audio_latent)   # [B, 1, T_audio] or [B, T_audio]

        # Normalise output shape: video → [B, T, H, W, 3], audio → [B, T_audio]
        if video.ndim == 5 and video.shape[1] == 3:
            video = video.permute(0, 2, 3, 4, 1)   # [B, 3, T, H, W] → [B, T, H, W, 3]
        if audio.ndim == 3:
            audio = audio.squeeze(1)                # [B, 1, T_a] → [B, T_a]

        return video, audio

    @torch.no_grad()
    def generate_with_refiner(
        self,
        prompt: str,
        refiner: Optional[nn.Module] = None,
        num_frames: int = 128,
        base_resolution: Tuple[int, int] = (854, 480),
        refiner_resolution: Tuple[int, int] = (1920, 1080),
        num_steps_base: int = 50,
        num_steps_refine: int = 30,
        show_progress: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Two-stage cascaded generation: 480P base + 1080P refinement.

        Args:
            prompt: Text prompt.
            refiner: AVRefiner model; if None, returns base resolution output.
            num_frames: Number of video frames.
            base_resolution: (width, height) for the base model.
            refiner_resolution: (width, height) for the refiner.
            num_steps_base: ODE steps for base model.
            num_steps_refine: ODE steps for refiner.
            show_progress: Show progress bars.
            **kwargs: Additional arguments passed to generate().

        Returns:
            video: [B, T, H_refine, W_refine, 3]
            audio: [B, T_audio]
        """
        base_w, base_h = base_resolution

        # Stage 1: Generate at base resolution
        video_480p, audio = self.generate(
            prompt,
            num_frames=num_frames,
            width=base_w,
            height=base_h,
            num_steps=num_steps_base,
            show_progress=show_progress,
            **kwargs,
        )

        if refiner is None:
            return video_480p, audio

        # Stage 2: Refine to high resolution
        ref_w, ref_h = refiner_resolution
        video_1080p, audio = refiner.refine(
            base_video=video_480p,
            base_audio=audio,
            prompt=prompt,
            target_width=ref_w,
            target_height=ref_h,
            num_steps=num_steps_refine,
        )
        return video_1080p, audio


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_pipeline(
    config: Dict[str, Any],
    device: str = "cuda",
) -> InferencePipeline:
    """
    Create InferencePipeline from config dict.

    Expected config keys
    --------------------
    model (dict): Passed to create_joint_av_dit().
    vae.video (str): Video VAE type / path.
    vae.audio (str): Audio VAE type ("wavvae" | "stable_audio" | "dac").
    text_encoder (dict): Passed to create_qwen3_encoder().
    inference.num_steps (int): Default denoising steps.
    inference.cfg_scale_text (float): Text CFG scale.
    inference.cfg_scale_mutual (float): Mutual CFG scale.
    checkpoint (str): Path to JointAVDiT checkpoint.

    Returns:
        InferencePipeline ready for generation.
    """
    from ..models.joint.joint_av_dit import create_joint_av_dit
    from ..models.audio_dit.audio_vae import create_audio_vae
    from ..models.text_encoders.qwen3_encoder import create_qwen3_encoder
    from ..models.video_dit.wan2x_loader import WanModelLoader

    # Build main model
    model = create_joint_av_dit(config.get("model", {}))

    # Load checkpoint
    ckpt_path = config.get("checkpoint")
    if ckpt_path:
        state = torch.load(ckpt_path, map_location="cpu")
        model_state = state.get("model", state)
        model.load_state_dict(model_state, strict=False)

    # Load Wan VAE
    wan_path = config.get("vae", {}).get("video", "")
    loader = WanModelLoader(model_path=wan_path, device=device)
    video_vae = loader.load_vae()

    # Load Audio VAE
    audio_vae_type = config.get("vae", {}).get("audio", "wavvae")
    audio_vae_path = config.get("vae", {}).get("audio_path", None)
    audio_vae = create_audio_vae(audio_vae_type, pretrained_path=audio_vae_path)

    # Load text encoder
    text_encoder = create_qwen3_encoder(config.get("text_encoder", {}))
    text_encoder.load_model()

    infer_cfg = config.get("inference", {})
    return InferencePipeline(
        model=model,
        video_vae=video_vae,
        audio_vae=audio_vae,
        text_encoder=text_encoder,
        device=device,
        num_steps=infer_cfg.get("num_steps", 50),
        cfg_scale_text=infer_cfg.get("cfg_scale_text", 7.5),
        cfg_scale_mutual=infer_cfg.get("cfg_scale_mutual", 2.0),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Entry point for ``alive-infer`` CLI command.

    Usage:
        alive-infer --prompt "A cat playing piano" \\
                    --output output.mp4 \\
                    --config configs/model/joint_av_dit.yaml \\
                    --steps 50 \\
                    --seed 42
    """
    import torchaudio
    import torchvision

    parser = argparse.ArgumentParser(description="Alive-Wan2X Inference")
    parser.add_argument("--prompt", required=True, help="Text prompt")
    parser.add_argument("--negative_prompt", default="", help="Negative prompt")
    parser.add_argument("--output", default="output.mp4", help="Output file path")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint path")
    parser.add_argument("--duration", type=float, default=5.0, help="Audio duration (s)")
    parser.add_argument("--num_frames", type=int, default=128, help="Number of video frames")
    parser.add_argument("--width", type=int, default=854, help="Video width")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--steps", type=int, default=50, help="Denoising steps")
    parser.add_argument("--cfg_text", type=float, default=7.5, help="Text CFG scale")
    parser.add_argument("--cfg_mutual", type=float, default=2.0, help="Mutual CFG scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    if args.checkpoint:
        cfg["checkpoint"] = args.checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = create_pipeline(cfg, device=device)

    video, audio = pipeline.generate(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        audio_duration=args.duration,
        num_steps=args.steps,
        cfg_scale_text=args.cfg_text,
        cfg_scale_mutual=args.cfg_mutual,
        seed=args.seed,
    )

    # Save video: torchvision expects [T, H, W, C] uint8
    video_uint8 = (video[0].clamp(0, 1) * 255).byte()  # [T, H, W, 3]
    base_output = args.output.rsplit(".", 1)[0]
    video_path = base_output + ".mp4"
    audio_path = base_output + ".wav"

    torchvision.io.write_video(video_path, video_uint8, fps=24)
    print(f"Video saved: {video_path}")

    # Save audio: torchaudio expects [C, T] float
    torchaudio.save(audio_path, audio[0:1], sample_rate=16000)
    print(f"Audio saved: {audio_path}")
