"""
Multi-Condition Classifier-Free Guidance

Implements multi-CFG for text and mutual cross-attention guidance.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn


class MultiCFG:
    """
    Multi-Condition Classifier-Free Guidance.
    
    Alive uses two types of guidance:
    1. Text CFG: Controls content generation direction
    2. Mutual CFG: Controls AV sync strength
    
    Formula:
    v_guided = v_uncond 
             + cfg_text * (v_text - v_uncond)
             + cfg_mutual * (v_cond - v_text)
    """
    
    def __init__(
        self,
        cfg_scale_text: float = 7.5,
        cfg_scale_mutual: float = 2.0
    ):
        """
        Initialize multi-CFG.
        
        Args:
            cfg_scale_text: Text guidance scale
            cfg_scale_mutual: Mutual attention guidance scale
        """
        self.cfg_scale_text = cfg_scale_text
        self.cfg_scale_mutual = cfg_scale_mutual
    
    def apply(
        self,
        v_uncond: torch.Tensor,
        v_text: torch.Tensor,
        v_cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply multi-condition guidance.
        
        Args:
            v_uncond: Unconditional prediction
            v_text: Text-only prediction (no mutual attention)
            v_cond: Full conditional prediction (with mutual attention)
            
        Returns:
            v_guided: Guided prediction
        """
        # Multi-CFG formula
        v_guided = v_uncond \
            + self.cfg_scale_text * (v_text - v_uncond) \
            + self.cfg_scale_mutual * (v_cond - v_text)
        
        return v_guided
    
    def apply_separate(
        self,
        video_uncond: torch.Tensor,
        video_text: torch.Tensor,
        video_cond: torch.Tensor,
        audio_uncond: torch.Tensor,
        audio_text: torch.Tensor,
        audio_cond: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply guidance separately to video and audio.
        
        Args:
            video_uncond: Unconditional video prediction
            video_text: Text-only video prediction
            video_cond: Full conditional video prediction
            audio_uncond: Unconditional audio prediction
            audio_text: Text-only audio prediction
            audio_cond: Full conditional audio prediction
            
        Returns:
            video_guided, audio_guided
        """
        video_guided = self.apply(video_uncond, video_text, video_cond)
        audio_guided = self.apply(audio_uncond, audio_text, audio_cond)
        
        return video_guided, audio_guided


class CFGScheduler:
    """
    CFG scale scheduler for progressive guidance.
    """
    
    def __init__(
        self,
        initial_text_cfg: float = 5.0,
        final_text_cfg: float = 7.5,
        initial_mutual_cfg: float = 1.0,
        final_mutual_cfg: float = 2.0,
        schedule_type: str = "linear"
    ):
        """
        Initialize CFG scheduler.
        
        Args:
            initial_text_cfg: Initial text CFG scale
            final_text_cfg: Final text CFG scale
            initial_mutual_cfg: Initial mutual CFG scale
            final_mutual_cfg: Final mutual CFG scale
            schedule_type: Schedule type (linear, cosine)
        """
        self.initial_text_cfg = initial_text_cfg
        self.final_text_cfg = final_text_cfg
        self.initial_mutual_cfg = initial_mutual_cfg
        self.final_mutual_cfg = final_mutual_cfg
        self.schedule_type = schedule_type
    
    def get_scale(
        self,
        step: int,
        total_steps: int
    ) -> Tuple[float, float]:
        """
        Get CFG scales for current step.
        
        Args:
            step: Current step
            total_steps: Total steps
            
        Returns:
            text_cfg, mutual_cfg
        """
        progress = step / total_steps
        
        if self.schedule_type == "linear":
            text_cfg = self.initial_text_cfg + (
                self.final_text_cfg - self.initial_text_cfg
            ) * progress
            mutual_cfg = self.initial_mutual_cfg + (
                self.final_mutual_cfg - self.initial_mutual_cfg
            ) * progress
        else:
            # Default to linear
            text_cfg = self.initial_text_cfg + (
                self.final_text_cfg - self.initial_text_cfg
            ) * progress
            mutual_cfg = self.initial_mutual_cfg + (
                self.final_mutual_cfg - self.initial_mutual_cfg
            ) * progress
        
        return text_cfg, mutual_cfg