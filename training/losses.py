"""
Loss Functions for Alive-Wan2X

Flow Matching losses and additional training objectives.

Flow Matching (Rectified Flow) formulation:
  - Forward process: x_t = (1 - t) * x0 + t * epsilon,  t in [0, 1]
  - Velocity target:  v_target = epsilon - x0
  - Model learns:     v_pred = model(x_t, t)
  - Loss:             ||v_pred - v_target||^2

Timestep sampling uses logit-normal distribution (as in SD3 / Wan2.x):
  u ~ N(0, 1),  t = sigmoid(u) in (0, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowMatchingLoss(nn.Module):
    """
    Flow Matching (Rectified Flow) loss.

    Wan2.x uses Flow Matching instead of traditional DDPM.
    The model predicts the velocity field v = dx/dt = epsilon - x0.
    """

    def __init__(
        self,
        video_weight: float = 1.0,
        audio_weight: float = 1.0,
        logit_normal_mean: float = 0.0,
        logit_normal_std: float = 1.0,
    ):
        """
        Initialize Flow Matching loss.

        Args:
            video_weight: Weight for video loss term.
            audio_weight: Weight for audio loss term.
            logit_normal_mean: Mean of the logit-normal timestep distribution.
            logit_normal_std: Std of the logit-normal timestep distribution.
        """
        super().__init__()

        self.video_weight = video_weight
        self.audio_weight = audio_weight
        self.logit_normal_mean = logit_normal_mean
        self.logit_normal_std = logit_normal_std

    def sample_timestep(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample timesteps using logit-normal distribution.

        t = sigmoid(N(mean, std)), which biases sampling towards the midpoint
        (t ≈ 0.5) where the model has the most to learn, similar to SD3.

        Args:
            batch_size: Number of timesteps to sample.
            device: Target device.

        Returns:
            t: Timesteps in (0, 1), shape [B].
        """
        u = torch.randn(batch_size, device=device) * self.logit_normal_std + self.logit_normal_mean
        return torch.sigmoid(u)

    def add_noise(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute noisy sample via rectified flow interpolation.

        x_t = (1 - t) * x0 + t * noise

        Args:
            x0: Clean data, shape [B, ...].
            noise: Sampled noise (epsilon ~ N(0, I)), same shape as x0.
            t: Timesteps in (0, 1), shape [B].

        Returns:
            x_t: Noisy sample, same shape as x0.
        """
        # Broadcast t from [B] to match x0 dimensions
        while t.dim() < x0.dim():
            t = t.unsqueeze(-1)
        return (1.0 - t) * x0 + t * noise

    def compute_velocity_target(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the target velocity field for rectified flow.

        v_target = noise - x0  (the direction from x0 to noise)

        Args:
            x0: Clean data.
            noise: Sampled noise (epsilon).

        Returns:
            v_target: Target velocity, same shape as x0.
        """
        return noise - x0

    def forward(
        self,
        x0_video: torch.Tensor,
        x0_audio: torch.Tensor,
        noise_video: torch.Tensor,
        noise_audio: torch.Tensor,
        v_pred_video: torch.Tensor,
        v_pred_audio: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weighted Flow Matching loss for video and audio.

        The trainer is responsible for:
          1. Sampling t via sample_timestep()
          2. Sampling noise_video / noise_audio ~ N(0, I)
          3. Computing x_t_video / x_t_audio via add_noise()
          4. Running the model on the noisy inputs to get v_pred
          5. Calling this forward() to get the loss

        Args:
            x0_video: Clean video latents, shape [B, C_v, T_v, H, W] or [B, T_v, C_v].
            x0_audio: Clean audio latents, shape [B, C_a, T_a] or [B, T_a, C_a].
            noise_video: Noise used to corrupt video, same shape as x0_video.
            noise_audio: Noise used to corrupt audio, same shape as x0_audio.
            v_pred_video: Model-predicted video velocity, same shape as x0_video.
            v_pred_audio: Model-predicted audio velocity, same shape as x0_audio.
            t: Timesteps in (0, 1), shape [B].

        Returns:
            loss: Scalar total loss.
        """
        v_target_video = self.compute_velocity_target(x0_video, noise_video)
        v_target_audio = self.compute_velocity_target(x0_audio, noise_audio)

        video_loss = F.mse_loss(v_pred_video, v_target_video)
        audio_loss = F.mse_loss(v_pred_audio, v_target_audio)

        return self.video_weight * video_loss + self.audio_weight * audio_loss


class SyncLoss(nn.Module):
    """
    Audio-Visual synchronization contrastive loss.

    Encourages temporally aligned video and audio tokens to have similar
    hidden representations, improving AV sync quality.
    """

    def __init__(self, weight: float = 0.1, temperature: float = 0.07):
        """
        Args:
            weight: Loss weight relative to the main flow matching loss.
            temperature: Contrastive softmax temperature.
        """
        super().__init__()
        self.weight = weight
        self.temperature = temperature

    def forward(
        self,
        video_hidden: torch.Tensor,
        audio_hidden: torch.Tensor,
        video_positions: torch.Tensor,
        audio_positions: torch.Tensor,
        time_tolerance: float = 0.1,
    ) -> torch.Tensor:
        """
        Compute InfoNCE-style AV sync loss over temporally aligned token pairs.

        For each video token, the audio token closest in physical time is the
        positive sample; all others are negatives.

        Args:
            video_hidden: Video hidden states [B, T_v, D_v].
            audio_hidden: Audio hidden states [B, T_a, D_a].
            video_positions: Video physical time positions [T_v] (seconds).
            audio_positions: Audio physical time positions [T_a] (seconds).
            time_tolerance: Tolerance in seconds to define alignment.

        Returns:
            sync_loss: Scalar synchronization loss.
        """
        B, T_v, D_v = video_hidden.shape
        T_a = audio_hidden.shape[1]

        # Project both modalities to a shared dimension (mean-pool to D_min)
        D = min(D_v, audio_hidden.shape[-1])
        v = video_hidden[..., :D]  # [B, T_v, D]
        a = audio_hidden[..., :D]  # [B, T_a, D]

        # Normalize for cosine similarity
        v = F.normalize(v, dim=-1)  # [B, T_v, D]
        a = F.normalize(a, dim=-1)  # [B, T_a, D]

        # Compute time-distance to identify positives
        # time_diff: [T_v, T_a]
        time_diff = torch.abs(video_positions.unsqueeze(1) - audio_positions.unsqueeze(0))
        # For each video token, find the closest audio token as positive
        _, pos_idx = time_diff.min(dim=1)  # [T_v]

        # Compute similarities: [B, T_v, T_a]
        sim = torch.bmm(v, a.transpose(1, 2)) / self.temperature  # [B, T_v, T_a]

        # Labels: positive audio index for each video token, replicated over batch
        labels = pos_idx.unsqueeze(0).expand(B, -1)  # [B, T_v]

        # Cross-entropy loss (video→audio direction)
        loss_v2a = F.cross_entropy(sim.reshape(B * T_v, T_a), labels.reshape(B * T_v))

        # Cross-entropy loss (audio→video direction)
        _, pos_idx_a = time_diff.min(dim=0)  # [T_a]
        labels_a = pos_idx_a.unsqueeze(0).expand(B, -1)  # [B, T_a]
        sim_t = sim.transpose(1, 2)  # [B, T_a, T_v]
        loss_a2v = F.cross_entropy(sim_t.reshape(B * T_a, T_v), labels_a.reshape(B * T_a))

        return self.weight * (loss_v2a + loss_a2v) * 0.5


class AudioQualityLoss(nn.Module):
    """
    Audio quality preservation loss.

    Helps prevent catastrophic forgetting during joint AV training by
    regularising the audio path with an L1 reconstruction objective.
    """

    def __init__(self, weight: float = 0.1):
        """
        Args:
            weight: Loss weight relative to the main flow matching loss.
        """
        super().__init__()
        self.weight = weight

    def forward(
        self,
        audio_pred: torch.Tensor,
        audio_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute L1 audio quality loss.

        Args:
            audio_pred: Predicted audio latent.
            audio_target: Reference audio latent (e.g. from audio-only pre-trained model).

        Returns:
            quality_loss: Scalar loss.
        """
        return self.weight * F.l1_loss(audio_pred, audio_target)


def create_loss(config: dict) -> FlowMatchingLoss:
    """
    Factory function to create FlowMatchingLoss from config.

    Args:
        config: Loss configuration dictionary. Supported keys:
            video_weight (float): Weight for video loss (default 1.0).
            audio_weight (float): Weight for audio loss (default 1.0).
            logit_normal_mean (float): Mean of logit-normal t-sampler (default 0.0).
            logit_normal_std (float): Std of logit-normal t-sampler (default 1.0).

    Returns:
        FlowMatchingLoss instance.
    """
    return FlowMatchingLoss(
        video_weight=config.get("video_weight", 1.0),
        audio_weight=config.get("audio_weight", 1.0),
        logit_normal_mean=config.get("logit_normal_mean", 0.0),
        logit_normal_std=config.get("logit_normal_std", 1.0),
    )
