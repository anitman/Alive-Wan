"""
Training Loop for Alive-Wan2X

Main training loop supporting:
  - Multi-stage training (warmup, joint, finetune, refiner)
  - Asymmetric learning rates (Audio DiT LR = 10× lower than CrossAttn)
  - Flow Matching loss
  - EMA with high decay (0.9999 / 0.99995 for Audio DiT)
  - FSDP or DDP distributed training
  - Gradient checkpointing
  - bfloat16 mixed precision
  - Checkpoint save / load
  - WandB / TensorBoard logging
"""

import copy
import os
import logging
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import FlowMatchingLoss, create_loss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exponential Moving Average
# ---------------------------------------------------------------------------

class EMA:
    """
    Exponential Moving Average of model parameters.

    Maintains a shadow copy of the model weights updated as:
      shadow = decay * shadow + (1 - decay) * param

    Two decay rates are supported:
      - default_decay: applied to most parameters
      - audio_dit_decay: higher decay (slower update) for Audio DiT weights
        to prevent catastrophic forgetting (see CLAUDE.md)
    """

    def __init__(
        self,
        model: nn.Module,
        default_decay: float = 0.9999,
        audio_dit_decay: float = 0.99995,
    ):
        """
        Args:
            model: Model whose parameters will be tracked.
            default_decay: EMA decay for non-Audio-DiT params.
            audio_dit_decay: EMA decay for Audio DiT params (higher = slower update).
        """
        self.default_decay = default_decay
        self.audio_dit_decay = audio_dit_decay
        # Shadow copy lives on CPU to save GPU memory; moved to device on apply()
        self.shadow = copy.deepcopy(model).cpu()
        self.shadow.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """
        Update shadow weights from the live model.

        Args:
            model: Live model (may be wrapped in DDP/FSDP).
        """
        # Unwrap DDP / FSDP if needed
        live = model.module if hasattr(model, "module") else model

        for (name, shadow_p), (_, live_p) in zip(
            self.shadow.named_parameters(), live.named_parameters()
        ):
            # Audio DiT parameters use slower EMA decay
            decay = self.audio_dit_decay if "audio" in name else self.default_decay
            shadow_p.data.mul_(decay).add_(live_p.data.cpu(), alpha=1.0 - decay)

    @torch.no_grad()
    def apply(self, model: nn.Module) -> None:
        """Copy shadow weights into the live model (for evaluation)."""
        live = model.module if hasattr(model, "module") else model
        for shadow_p, live_p in zip(self.shadow.parameters(), live.parameters()):
            live_p.data.copy_(shadow_p.data.to(live_p.device))

    def save(self, path: str) -> None:
        """Save EMA shadow weights to disk."""
        torch.save(self.shadow.state_dict(), path)

    def load(self, path: str, map_location: str = "cpu") -> None:
        """Load EMA shadow weights from disk."""
        state = torch.load(path, map_location=map_location)
        self.shadow.load_state_dict(state)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Main trainer for Alive-Wan2X.

    Usage
    -----
    trainer = Trainer(model, optimizer, scheduler, device, config)
    trainer.train(train_loader, num_steps=100_000)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: str = "cuda",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            model: JointAVDiT (or DDP/FSDP-wrapped).
            optimizer: Optimizer, potentially with per-group asymmetric LRs.
            scheduler: LR scheduler (CosineScheduleWithWarmup recommended).
            device: Training device string.
            config: Full training config dict. Expected sub-keys:
                loss (dict): Passed to create_loss().
                training.ema.enabled (bool)
                training.ema.decay (float)
                training.ema.audio_dit_decay (float)
                training.grad_clip (float, default 1.0)
                training.mixed_precision (bool, default True)
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config or {}

        training_cfg = self.config.get("training", {})

        # Loss function
        self.loss_fn = create_loss(self.config.get("loss", {}))

        # Gradient clipping
        self.grad_clip = training_cfg.get("grad_clip", 1.0)

        # Mixed precision
        self.mixed_precision = training_cfg.get("mixed_precision", True)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        # EMA
        self.ema: Optional[EMA] = None
        ema_cfg = training_cfg.get("ema", {})
        if ema_cfg.get("enabled", False):
            self.ema = EMA(
                model,
                default_decay=ema_cfg.get("decay", 0.9999),
                audio_dit_decay=ema_cfg.get("audio_dit_decay", 0.99995),
            )

        # Logging backend (optional WandB)
        self._wandb = None
        if training_cfg.get("wandb", {}).get("enabled", False):
            try:
                import wandb
                wandb.init(**training_cfg["wandb"].get("init_kwargs", {}))
                self._wandb = wandb
            except ImportError:
                logger.warning("wandb not installed; logging to console only.")

    # ---------------------------------------------------------------------- #
    # Single training step
    # ---------------------------------------------------------------------- #

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Execute one forward + backward + optimizer step.

        Expected batch keys
        -------------------
        video_latent   : [B, C_v, T_v, H, W]  clean video latents
        audio_latent   : [B, C_a, T_a]         clean audio latents
        text_embeds    : [B, N, D]              visual text embeddings
        speech_embeds  : [B, N_s, D_s]         speech text (optional)
        style_embeds   : [B, N_style, D_style]  style text (optional)
        speech_mask    : [B, N_s]               (optional)
        style_mask     : [B, N_style]           (optional)

        Returns
        -------
        Metrics dict with 'loss', 'video_loss', 'audio_loss'.
        """
        self.model.train()

        # Move data to device
        video_latent = batch["video_latent"].to(self.device)
        audio_latent = batch["audio_latent"].to(self.device)
        text_embeds  = batch["text_embeds"].to(self.device)

        speech_embeds = _to_device(batch.get("speech_embeds"), self.device)
        style_embeds  = _to_device(batch.get("style_embeds"),  self.device)
        speech_mask   = _to_device(batch.get("speech_mask"),   self.device)
        style_mask    = _to_device(batch.get("style_mask"),    self.device)

        B = video_latent.shape[0]

        # 1. Sample timestep  t ~ logit-normal
        t = self.loss_fn.sample_timestep(B, self.device)

        # 2. Sample noise
        noise_video = torch.randn_like(video_latent)
        noise_audio = torch.randn_like(audio_latent)

        # 3. Corrupt inputs:  x_t = (1-t)*x0 + t*eps
        video_noisy = self.loss_fn.add_noise(video_latent, noise_video, t)
        audio_noisy = self.loss_fn.add_noise(audio_latent, noise_audio, t)

        # 4. Forward pass (bfloat16)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=self.mixed_precision):
            v_pred_video, v_pred_audio, _ = self.model(
                video_noisy,
                audio_noisy,
                text_embeds,
                t,
                speech_text=speech_embeds,
                style_text=style_embeds,
                speech_mask=speech_mask,
                style_mask=style_mask,
            )

            # 5. Compute flow matching loss
            loss = self.loss_fn(
                x0_video=video_latent,
                x0_audio=audio_latent,
                noise_video=noise_video,
                noise_audio=noise_audio,
                v_pred_video=v_pred_video,
                v_pred_audio=v_pred_audio,
                t=t,
            )

        # 6. Backward + gradient clipping + optimizer step
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # 7. LR scheduler step
        if self.scheduler is not None:
            self.scheduler.step()

        # 8. EMA update
        if self.ema is not None:
            self.ema.update(self.model)

        return {"loss": loss.item()}

    # ---------------------------------------------------------------------- #
    # Full training loop
    # ---------------------------------------------------------------------- #

    def train(
        self,
        train_loader: DataLoader,
        num_steps: int,
        save_interval: int = 1000,
        eval_interval: int = 1000,
        log_interval: int = 100,
        checkpoint_dir: str = "./checkpoints",
        resume_from: Optional[str] = None,
    ) -> None:
        """
        Run the full training loop.

        Args:
            train_loader: Infinite (or repeating) DataLoader of AVDataset batches.
            num_steps: Total number of gradient steps.
            save_interval: Save a checkpoint every N steps.
            eval_interval: Run evaluation every N steps.
            log_interval: Log metrics every N steps.
            checkpoint_dir: Directory for checkpoints.
            resume_from: Path to a checkpoint to resume from.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Resume from checkpoint if requested
        start_step = 0
        if resume_from is not None:
            start_step = self.load_checkpoint(resume_from)
            logger.info(f"Resumed from step {start_step}: {resume_from}")

        step = start_step
        running_loss = 0.0

        pbar = tqdm(total=num_steps, initial=start_step, desc="Training")
        data_iter = _infinite_loader(train_loader)

        while step < num_steps:
            batch = next(data_iter)
            metrics = self.train_step(batch)
            running_loss += metrics["loss"]

            # Logging
            if (step + 1) % log_interval == 0:
                avg_loss = running_loss / log_interval
                running_loss = 0.0
                lr = self.optimizer.param_groups[0]["lr"]
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}"})
                if self._wandb is not None:
                    self._wandb.log({"train/loss": avg_loss, "train/lr": lr}, step=step)

            # Checkpointing
            if (step + 1) % save_interval == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{step + 1:07d}.pt")
                self.save_checkpoint(ckpt_path, step + 1)
                logger.info(f"Saved checkpoint: {ckpt_path}")

            pbar.update(1)
            step += 1

        pbar.close()
        # Save final checkpoint
        self.save_checkpoint(os.path.join(checkpoint_dir, "checkpoint_final.pt"), step)

    # ---------------------------------------------------------------------- #
    # Checkpoint I/O
    # ---------------------------------------------------------------------- #

    def save_checkpoint(self, path: str, step: int) -> None:
        """
        Save full training state to disk.

        Saved keys: step, model, optimizer, scheduler (if any), ema (if any).
        """
        live = self.model.module if hasattr(self.model, "module") else self.model
        state: Dict[str, Any] = {
            "step": step,
            "model": live.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        if self.ema is not None:
            state["ema"] = self.ema.shadow.state_dict()
        torch.save(state, path)

    def load_checkpoint(self, path: str) -> int:
        """
        Load training state from disk.

        Returns:
            step: The step at which the checkpoint was saved.
        """
        state = torch.load(path, map_location=self.device)
        live = self.model.module if hasattr(self.model, "module") else self.model
        live.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state and self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        if "ema" in state and self.ema is not None:
            self.ema.shadow.load_state_dict(state["ema"])
        return state.get("step", 0)

    # ---------------------------------------------------------------------- #
    # Evaluation
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def evaluate(self, eval_loader: DataLoader) -> Dict[str, float]:
        """
        Compute validation loss on eval_loader.

        Returns:
            Dict with 'val_loss'.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in eval_loader:
            video_latent = batch["video_latent"].to(self.device)
            audio_latent = batch["audio_latent"].to(self.device)
            text_embeds  = batch["text_embeds"].to(self.device)

            B = video_latent.shape[0]
            t = self.loss_fn.sample_timestep(B, self.device)

            noise_video = torch.randn_like(video_latent)
            noise_audio = torch.randn_like(audio_latent)
            video_noisy = self.loss_fn.add_noise(video_latent, noise_video, t)
            audio_noisy = self.loss_fn.add_noise(audio_latent, noise_audio, t)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=self.mixed_precision):
                v_pred_video, v_pred_audio, _ = self.model(
                    video_noisy, audio_noisy, text_embeds, t
                )
                loss = self.loss_fn(
                    video_latent, audio_latent,
                    noise_video, noise_audio,
                    v_pred_video, v_pred_audio, t,
                )

            total_loss += loss.item()
            num_batches += 1

        self.model.train()
        return {"val_loss": total_loss / max(num_batches, 1)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_device(
    x: Optional[torch.Tensor], device: str
) -> Optional[torch.Tensor]:
    """Move tensor to device if not None."""
    return x.to(device) if x is not None else None


def _infinite_loader(loader: DataLoader):
    """Wrap a DataLoader to cycle infinitely."""
    while True:
        yield from loader


def _build_optimizer(
    model: nn.Module,
    config: Dict[str, Any],
) -> torch.optim.Optimizer:
    """
    Build optimizer with per-group learning rates.

    Parameter groups (following Alive training stages):
      - "ta_cross_attn": Cross-attention params   (full LR)
      - "audio_dit":     Audio DiT params          (0.1× LR to prevent forgetting)
      - "video_dit":     Video DiT params           (frozen in stages 1-2 → LR=0)
      - "other":         All remaining params       (full LR)

    Args:
        model: JointAVDiT model.
        config: Training config with keys:
            optimizer.lr (float)
            optimizer.weight_decay (float)
            stage (str): "stage1" | "stage2" | "stage3"

    Returns:
        AdamW optimizer.
    """
    opt_cfg = config.get("optimizer", {})
    base_lr = opt_cfg.get("lr", 1e-4)
    weight_decay = opt_cfg.get("weight_decay", 1e-2)
    stage = config.get("stage", "stage1")

    # Stage-dependent LR multipliers
    # Stage 1: only CrossAttn trains
    # Stage 2: Audio DiT + CrossAttn train, Video DiT frozen
    # Stage 3: everything trains with asymmetric LR
    stage_scales = {
        "stage1": {"ta_cross_attn": 1.0, "audio_dit": 0.0, "video_dit": 0.0, "other": 0.0},
        "stage2": {"ta_cross_attn": 1.0, "audio_dit": 1.0, "video_dit": 0.0, "other": 1.0},
        "stage3": {"ta_cross_attn": 1.0, "audio_dit": 0.1, "video_dit": 0.1, "other": 1.0},
        "stage4": {"ta_cross_attn": 0.0, "audio_dit": 0.0, "video_dit": 1.0, "other": 0.0},
    }
    scales = stage_scales.get(stage, stage_scales["stage1"])

    param_groups: List[Dict[str, Any]] = []
    for name, param in model.named_parameters():
        if "ta_cross_attn" in name:
            group_name = "ta_cross_attn"
        elif "audio" in name:
            group_name = "audio_dit"
        elif "video" in name or "wan" in name:
            group_name = "video_dit"
        else:
            group_name = "other"

        lr = base_lr * scales[group_name]
        if lr == 0.0:
            param.requires_grad_(False)
            continue

        param_groups.append({
            "params": [param],
            "lr": lr,
            "weight_decay": weight_decay if param.ndim >= 2 else 0.0,
            "name": group_name,
        })

    return torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Entry point for ``alive-train`` CLI command.

    Usage:
        alive-train --config configs/train/stage1_warmup.yaml \\
                    --output_dir ./outputs/stage1

    Config file should be a YAML file with keys matching OmegaConf schema.
    See configs/train/ for examples.
    """
    import argparse
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    from ..models.joint.joint_av_dit import create_joint_av_dit
    from ..data.datasets.av_dataset import AVDataset
    from .schedulers import CosineScheduleWithWarmup

    parser = argparse.ArgumentParser(description="Train Alive-Wan2X")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--output_dir", default="./outputs", help="Output directory")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build model
    model = create_joint_av_dit(cfg.get("model", {})).to(device)

    # Build optimizer (asymmetric LR per training stage)
    optimizer = _build_optimizer(model, cfg)

    # Build LR scheduler
    train_cfg = cfg.get("training", {})
    scheduler = CosineScheduleWithWarmup(
        optimizer,
        warmup_steps=train_cfg.get("warmup_steps", 1000),
        total_steps=train_cfg.get("num_steps", 100_000),
        min_lr_ratio=train_cfg.get("min_lr_ratio", 0.1),
    )

    # Build dataset + loader
    data_cfg = cfg.get("data", {})
    dataset = AVDataset(
        data_root=data_cfg.get("data_root", "./data"),
        max_video_frames=data_cfg.get("max_video_frames", 128),
        audio_duration=data_cfg.get("audio_duration", 5.0),
    )
    train_loader = DataLoader(
        dataset,
        batch_size=train_cfg.get("batch_size", 1),
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )

    # Build trainer and run
    trainer = Trainer(model, optimizer, scheduler, device=device, config=cfg)
    trainer.train(
        train_loader,
        num_steps=train_cfg.get("num_steps", 100_000),
        save_interval=train_cfg.get("save_interval", 1000),
        eval_interval=train_cfg.get("eval_interval", 1000),
        log_interval=train_cfg.get("log_interval", 100),
        checkpoint_dir=args.output_dir,
        resume_from=args.resume,
    )
