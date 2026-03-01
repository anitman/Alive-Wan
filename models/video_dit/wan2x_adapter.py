"""
Wan2.x Video DiT Adapter

Adapts Wan2.1/Wan2.2 Video DiT for integration with Audio DiT in JointAVDiT.

The adapter does NOT subclass the Wan model directly (to avoid tight coupling).
Instead it holds a reference to the loaded Wan transformer and provides methods
to:
  1. Extract per-block video hidden states
  2. Inject TA-CrossAttn audio signals between Wan blocks
  3. Run the full adapted forward pass

Supports:
  - Wan2.1-14B (standard dense transformer, 40 blocks)
  - Wan2.2-A14B (MoE with dual-expert routing based on SNR)

For MoE (Wan2.2), the routing expert index is determined by the timestep SNR:
  expert_0 (quality-focused) when t < 0.5
  expert_1 (diversity-focused) when t >= 0.5
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .wan2x_loader import WanModelLoader, _WanBlockAdapter


class WanDiTAdapter(nn.Module):
    """
    Adapter for Wan2.x Video DiT models.

    This class handles:
      1. Loading Wan2.x pre-trained weights via WanModelLoader
      2. Wrapping each Wan block in _WanBlockAdapter for unified calling
      3. Running the adapted forward pass (with optional TA-CrossAttn injection)
      4. Gradient freezing / unfreezing per training stage

    The adapter is designed to slot into JointAVDiT's dual_stream blocks.
    For standalone usage, call ``forward()`` directly.
    """

    def __init__(
        self,
        model_type: str = "wan2.1-14b",
        pretrained_path: Optional[str] = None,
        freeze: bool = True,
        num_blocks_to_use: int = 16,
    ):
        """
        Args:
            model_type: "wan2.1-14b" or "wan2.2-a14b".
            pretrained_path: Local path to the Wan2.x checkpoint directory.
            freeze: Freeze Video DiT weights (True for stages 1-2).
            num_blocks_to_use: How many Wan blocks to expose (default 16,
                               matching the JointAVDiT dual_stream depth).
        """
        super().__init__()

        self.model_type = model_type
        self.pretrained_path = pretrained_path
        self.freeze = freeze
        self.num_blocks_to_use = num_blocks_to_use

        self._wan_model: Optional[nn.Module] = None
        self.blocks: Optional[nn.ModuleList] = None

        if pretrained_path is not None:
            self._load()

    def _load(self) -> None:
        """Load Wan2.x weights and build the adapted block list."""
        loader = WanModelLoader(
            model_path=self.pretrained_path,
            model_type=self.model_type.split("-")[0],  # "wan2.1" or "wan2.2"
        )
        wan_dit = loader.load_video_dit()

        if isinstance(wan_dit, nn.Identity):
            warnings.warn(
                "Wan2.x Video DiT could not be loaded. "
                "WanDiTAdapter will pass video hidden states through unchanged.",
                RuntimeWarning,
            )
            self._wan_model = wan_dit
            return

        self._wan_model = wan_dit

        # Extract individual blocks
        wan_blocks_raw = None
        for attr in ("blocks", "transformer_blocks", "layers"):
            if hasattr(wan_dit, attr):
                wan_blocks_raw = getattr(wan_dit, attr)
                break

        if wan_blocks_raw is None:
            warnings.warn(
                "Could not find transformer blocks in loaded Wan model. "
                "Using whole model as a single block.",
                RuntimeWarning,
            )
            self.blocks = nn.ModuleList([_WanBlockAdapter(wan_dit)])
        else:
            n = min(len(wan_blocks_raw), self.num_blocks_to_use)
            self.blocks = nn.ModuleList([
                _WanBlockAdapter(wan_blocks_raw[i]) for i in range(n)
            ])

        if self.freeze:
            self._wan_model.requires_grad_(False)

    # ---------------------------------------------------------------------- #
    # Freeze / unfreeze helpers (used during training stage transitions)
    # ---------------------------------------------------------------------- #

    def set_frozen(self, frozen: bool) -> None:
        """Freeze or unfreeze the Wan Video DiT weights."""
        self.freeze = frozen
        if self._wan_model is not None:
            self._wan_model.requires_grad_(not frozen)

    # ---------------------------------------------------------------------- #
    # Per-block inference (for injection into JointAVDiTBlock)
    # ---------------------------------------------------------------------- #

    def get_block(self, idx: int) -> Optional[_WanBlockAdapter]:
        """
        Return the adapted Wan block at position ``idx``.

        Returns None if Wan weights were not loaded.
        """
        if self.blocks is None or idx >= len(self.blocks):
            return None
        return self.blocks[idx]

    def inject_into_joint_model(self, joint_model: nn.Module) -> None:
        """
        Assign Wan blocks to the ``video_block`` slot of each
        ``JointAVDiTBlock`` in joint_model.dual_stream.

        Args:
            joint_model: JointAVDiT model.
        """
        if not hasattr(joint_model, "dual_stream") or self.blocks is None:
            warnings.warn("Cannot inject: dual_stream not found or blocks not loaded.")
            return

        num_dual = len(joint_model.dual_stream)
        num_wan  = len(self.blocks)

        for i in range(min(num_dual, num_wan)):
            joint_model.dual_stream[i].video_block = self.blocks[i]

        print(
            f"[WanDiTAdapter] Injected {min(num_dual, num_wan)} blocks "
            f"into JointAVDiT.dual_stream."
        )

    # ---------------------------------------------------------------------- #
    # Full forward pass (standalone use or testing)
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        video_latent: Tensor,
        audio_latent: Tensor,
        text_embeds: Tensor,
        timestep: Tensor,
        ta_cross_attn_blocks: Optional[nn.ModuleList] = None,
        return_intermediates: bool = False,
    ) -> Tuple[Tensor, Optional[Dict]]:
        """
        Run video hidden states through the adapted Wan blocks,
        optionally applying TA-CrossAttn between each block.

        Args:
            video_latent: [B, C_v, T_v, H, W] or [B, T_v, D_v].
            audio_latent: [B, C_a, T_a] or [B, T_a, D_a].
            text_embeds: [B, N, D].
            timestep: [B] in (0, 1).
            ta_cross_attn_blocks: Optional list of TACrossAttnBlock modules
                (one per Wan block). If provided, audio↔video attention is
                applied after each Wan block.
            return_intermediates: Return per-block hidden states.

        Returns:
            video_out: Video hidden states after all blocks.
            intermediates: Dict of per-block states (if requested).
        """
        if self._wan_model is None:
            raise RuntimeError(
                "WanDiTAdapter is not loaded. "
                "Pass pretrained_path to __init__ or call _load() first."
            )

        if self.blocks is None:
            # Model loaded but blocks could not be extracted — run whole model
            if isinstance(video_latent, Tensor) and video_latent.ndim == 5:
                B, C, T, H, W = video_latent.shape
                video_seq = video_latent.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
            else:
                video_seq = video_latent
            video_out = self._wan_model(video_seq)
            if isinstance(video_out, (tuple, list)):
                video_out = video_out[0]
            return video_out, None

        # Flatten [B, C, T, H, W] → [B, T*H*W, C] for block processing
        if video_latent.ndim == 5:
            B, C, T, H, W = video_latent.shape
            video_hidden = video_latent.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
        else:
            video_hidden = video_latent

        if audio_latent.ndim == 3 and audio_latent.shape[1] != audio_latent.shape[-1]:
            # [B, C_a, T_a] → [B, T_a, C_a]
            audio_hidden = audio_latent.transpose(1, 2)
        else:
            audio_hidden = audio_latent

        intermediates: Optional[Dict] = {} if return_intermediates else None

        for i, block in enumerate(self.blocks):
            video_hidden = block(video_hidden, timestep, text_embeds)

            # Optional TA-CrossAttn injection
            if ta_cross_attn_blocks is not None and i < len(ta_cross_attn_blocks):
                video_hidden, audio_hidden = ta_cross_attn_blocks[i](
                    video_hidden, audio_hidden
                )

            if return_intermediates:
                intermediates[f"block_{i}"] = video_hidden.detach()

        return video_hidden, intermediates


# ---------------------------------------------------------------------------
# Model-specific subclasses
# ---------------------------------------------------------------------------

class Wan21Adapter(WanDiTAdapter):
    """Wan2.1-14B specific adapter (dense transformer)."""

    def __init__(
        self,
        pretrained_path: Optional[str] = None,
        freeze: bool = True,
        num_blocks_to_use: int = 16,
    ):
        super().__init__(
            model_type="wan2.1-14b",
            pretrained_path=pretrained_path,
            freeze=freeze,
            num_blocks_to_use=num_blocks_to_use,
        )


class Wan22Adapter(WanDiTAdapter):
    """
    Wan2.2-A14B MoE specific adapter.

    Wan2.2 uses two expert networks per block with SNR-based routing:
      - expert_0: quality/detail-focused (activated at low noise, t < 0.5)
      - expert_1: diversity-focused (activated at high noise, t >= 0.5)

    The routing is implicit in the Wan2.2 forward pass; no explicit
    routing override is needed from our adapter.
    """

    def __init__(
        self,
        pretrained_path: Optional[str] = None,
        freeze: bool = True,
        num_blocks_to_use: int = 16,
    ):
        super().__init__(
            model_type="wan2.2-a14b",
            pretrained_path=pretrained_path,
            freeze=freeze,
            num_blocks_to_use=num_blocks_to_use,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_wan_adapter(
    model_type: str,
    **kwargs: Any,
) -> WanDiTAdapter:
    """
    Factory function to create the appropriate Wan adapter.

    Args:
        model_type: "wan2.1-14b" or "wan2.2-a14b".
        **kwargs: Additional keyword arguments passed to the adapter constructor.

    Returns:
        WanDiTAdapter subclass instance.

    Raises:
        ValueError: If ``model_type`` is not recognised.
    """
    adapters = {
        "wan2.1-14b": Wan21Adapter,
        "wan2.2-a14b": Wan22Adapter,
    }
    if model_type not in adapters:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Supported: {list(adapters.keys())}"
        )
    return adapters[model_type](**kwargs)
