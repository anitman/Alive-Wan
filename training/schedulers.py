"""
Learning Rate Schedulers for Alive-Wan2X
"""

from typing import Optional
import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineScheduleWithWarmup(_LRScheduler):
    """
    Cosine learning rate schedule with linear warmup.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1
    ):
        """
        Initialize scheduler.
        
        Args:
            optimizer: Optimizer
            warmup_steps: Warmup steps
            total_steps: Total training steps
            min_lr_ratio: Minimum LR ratio (min_lr / base_lr)
            last_epoch: Last epoch
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> list:
        """Get learning rates."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            return [
                base_lr * self.min_lr_ratio + (base_lr - base_lr * self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


class AsymmetricLR:
    """
    Asymmetric learning rate manager for different model components.
    
    Used to prevent catastrophic forgetting in Audio DiT during joint training.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_scales: dict
    ):
        """
        Initialize asymmetric LR manager.
        
        Args:
            optimizer: Optimizer
            lr_scales: Dictionary of parameter group name -> LR scale
                       e.g., {"audio_dit": 0.1, "video_dit": 0.0, "ta_cross_attn": 1.0}
        """
        self.optimizer = optimizer
        self.lr_scales = lr_scales
        
        # Store base learning rates
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        
        # Apply initial scaling
        self._apply_scales()
    
    def _apply_scales(self) -> None:
        """Apply LR scales to parameter groups."""
        for i, group in enumerate(self.optimizer.param_groups):
            name = group.get("name", f"group_{i}")
            scale = self.lr_scales.get(name, 1.0)
            group["lr"] = self.base_lrs[i] * scale
    
    def set_scales(self, lr_scales: dict) -> None:
        """
        Update LR scales.
        
        Args:
            lr_scales: New LR scales
        """
        self.lr_scales.update(lr_scales)
        self._apply_scales()