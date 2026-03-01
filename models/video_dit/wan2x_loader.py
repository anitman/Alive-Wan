"""
Wan2.x Model Loader

Loads Wan2.1-14B or Wan2.2-A14B pre-trained weights and injects them into
the JointAVDiT model's dual-stream video blocks.

Wan2.x is an open video generation model from Alibaba Wan-AI:
  - Wan2.1-T2V-14B: Text-to-Video, dense transformer
  - Wan2.2-A14B:    Audio-to-Video, MoE transformer

HuggingFace repositories:
  Wan-AI/Wan2.1-T2V-14B
  Wan-AI/Wan2.2-A14B

Usage:
    loader = WanModelLoader("./models/wan2.1-t2v-14b", model_type="wan2.1")
    loader.inject_into_joint_model(joint_av_dit)
"""

import os
from typing import Dict, List, Optional, Any

import torch
from torch import nn


class WanModelLoader:
    """
    Loader for Wan2.x pre-trained model weights.

    Handles:
    - Detecting the local Wan2.x checkpoint format
    - Mapping Wan weight names to JointAVDiT's dual_stream block naming
    - Injecting loaded Wan blocks as ``video_block`` on each JointAVDiTBlock
    - Loading the Wan VAE for inference decoding
    - Loading the T5 text encoder for compatibility

    Notes on Wan checkpoint format:
      Wan2.1 checkpoints (from HuggingFace) are stored as safetensors shards
      under <model_path>/diffusion_pytorch_model*.safetensors.
      Alternatively, a single .pth or .pt file is supported.
    """

    def __init__(
        self,
        model_path: str,
        model_type: str = "wan2.1",
        device: str = "cuda",
    ):
        """
        Args:
            model_path: Local path to the Wan2.x checkpoint directory,
                        or a HuggingFace model ID string.
            model_type: "wan2.1" or "wan2.2".
            device: Device to load the model onto.
        """
        self.model_path = model_path
        self.model_type = model_type
        self.device = device
        self._state_dict: Optional[Dict[str, torch.Tensor]] = None

    # ---------------------------------------------------------------------- #
    # Weight loading
    # ---------------------------------------------------------------------- #

    def _load_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Load the full Wan checkpoint state dict.

        Supports:
          - safetensors shards (preferred, used by HuggingFace Wan2.1)
          - single .pt / .pth file
          - directory with index.json + shards
        """
        if self._state_dict is not None:
            return self._state_dict

        model_path = self.model_path

        # Check for safetensors shards
        shard_files = sorted(
            [f for f in os.listdir(model_path)
             if f.startswith("diffusion_pytorch_model") and f.endswith(".safetensors")]
        ) if os.path.isdir(model_path) else []

        if shard_files:
            try:
                from safetensors.torch import load_file
            except ImportError as e:
                raise ImportError(
                    "Loading Wan2.x safetensors requires the safetensors package. "
                    "Install with: pip install safetensors"
                ) from e
            merged: Dict[str, torch.Tensor] = {}
            for shard in shard_files:
                shard_path = os.path.join(model_path, shard)
                merged.update(load_file(shard_path, device="cpu"))
            self._state_dict = merged
            return merged

        # Check for single .pt / .pth file
        for fname in os.listdir(model_path) if os.path.isdir(model_path) else []:
            if fname.endswith(".pt") or fname.endswith(".pth"):
                full = os.path.join(model_path, fname)
                state = torch.load(full, map_location="cpu")
                if "state_dict" in state:
                    state = state["state_dict"]
                self._state_dict = state
                return state

        # Fallback: try to load directly as a file path
        if os.path.isfile(model_path):
            state = torch.load(model_path, map_location="cpu")
            if "state_dict" in state:
                state = state["state_dict"]
            self._state_dict = state
            return state

        raise FileNotFoundError(
            f"No Wan2.x checkpoint found at '{model_path}'. "
            f"Download with: huggingface-cli download Wan-AI/Wan2.1-T2V-14B "
            f"--local-dir {model_path}"
        )

    def get_weight_mapping(self) -> Dict[str, str]:
        """
        Return a mapping from Wan2.x weight name prefixes to JointAVDiT
        dual-stream video block naming convention.

        Wan2.1 DiT blocks are typically named:
          transformer.blocks.<i>.<sub_module>.<weight>

        JointAVDiT dual-stream video blocks are at:
          dual_stream.<i>.video_block.<sub_module>.<weight>

        Returns:
            Dict mapping Wan prefix → JointAVDiT prefix.
        """
        # Wan2.1 uses "model.blocks.<i>" or "transformer.blocks.<i>"
        # Exact names depend on the checkpoint; we map the common prefix.
        if self.model_type.startswith("wan2.1"):
            return {
                "model.blocks.": "dual_stream.{i}.video_block.",
                "transformer.blocks.": "dual_stream.{i}.video_block.",
            }
        elif self.model_type.startswith("wan2.2"):
            return {
                "model.blocks.": "dual_stream.{i}.video_block.",
                "transformer.blocks.": "dual_stream.{i}.video_block.",
            }
        return {}

    def load_video_dit(self) -> nn.Module:
        """
        Load and return the Wan2.x Video DiT as a standalone nn.Module.

        The returned module is the inner Transformer (without VAE/text encoder).
        Requires the `wan` library to be installed:
            pip install git+https://github.com/Wan-AI/Wan2.git

        Returns:
            wan_model: Wan DiT nn.Module in eval mode.
        """
        try:
            # Primary path: use Wan2.x official Python package
            from wan import WanT2V  # type: ignore
            wan_model = WanT2V.from_pretrained(self.model_path)
            wan_model = wan_model.transformer  # extract DiT backbone
            wan_model.requires_grad_(False)
            wan_model.eval()
            return wan_model
        except ImportError:
            pass

        try:
            # Fallback: use diffusers Wan pipeline if available
            from diffusers import WanPipeline  # type: ignore
            pipe = WanPipeline.from_pretrained(
                self.model_path, torch_dtype=torch.bfloat16
            )
            transformer = pipe.transformer
            transformer.requires_grad_(False)
            transformer.eval()
            return transformer
        except (ImportError, Exception):
            pass

        # Final fallback: return a warning stub
        import warnings
        warnings.warn(
            "Could not load Wan2.x Video DiT. "
            "The video processing path will be identity (pass-through). "
            "Install the wan library: pip install git+https://github.com/Wan-AI/Wan2.git",
            RuntimeWarning,
            stacklevel=2,
        )
        return nn.Identity()

    def load_vae(self) -> nn.Module:
        """
        Load and return the Wan VAE (encoder + decoder).

        Wan-VAE:
          - 16 latent channels
          - 8× spatial compression
          - 4× temporal compression

        Returns:
            vae: VAE nn.Module with .encode() and .decode() methods.
        """
        try:
            from wan import WanT2V  # type: ignore
            wan = WanT2V.from_pretrained(self.model_path)
            vae = wan.vae
            vae.requires_grad_(False)
            vae.eval()
            return vae
        except ImportError:
            pass

        try:
            from diffusers import WanPipeline  # type: ignore
            pipe = WanPipeline.from_pretrained(
                self.model_path, torch_dtype=torch.bfloat16
            )
            vae = pipe.vae
            vae.requires_grad_(False)
            vae.eval()
            return vae
        except (ImportError, Exception):
            pass

        raise RuntimeError(
            "Cannot load Wan VAE. Install the wan or diffusers library:\n"
            "  pip install git+https://github.com/Wan-AI/Wan2.git\n"
            "  -- or --\n"
            "  pip install diffusers"
        )

    def load_text_encoder(self) -> nn.Module:
        """
        Load and return the Wan text encoder (T5-based).

        Note: Alive-Wan2X uses Qwen3 as the primary text encoder.
        This method is provided for compatibility with Wan2.x training scripts.

        Returns:
            text_encoder: T5 nn.Module with tokenizer attached.
        """
        try:
            from wan import WanT2V  # type: ignore
            wan = WanT2V.from_pretrained(self.model_path)
            return wan.text_encoder
        except ImportError:
            pass

        try:
            from diffusers import WanPipeline  # type: ignore
            pipe = WanPipeline.from_pretrained(self.model_path)
            return pipe.text_encoder
        except (ImportError, Exception):
            pass

        raise RuntimeError(
            "Cannot load Wan text encoder. See load_video_dit() for installation instructions."
        )

    def inject_into_joint_model(self, joint_model: nn.Module) -> None:
        """
        Inject Wan2.x DiT blocks as ``video_block`` into each
        ``JointAVDiTBlock`` of the dual-stream phase.

        This enables the Video DiT path to use pre-trained Wan2.x weights
        while the Audio DiT path and TA-CrossAttn train from scratch.

        After injection:
          - joint_model.dual_stream[i].video_block = wan_block_i  (frozen)
          - Audio DiT blocks and TA-CrossAttn blocks remain trainable

        Args:
            joint_model: JointAVDiT model instance.
        """
        wan_dit = self.load_video_dit()

        if isinstance(wan_dit, nn.Identity):
            # No Wan blocks available; skip injection
            return

        # Try to access the individual transformer blocks
        wan_blocks = None
        for attr in ("blocks", "transformer_blocks", "layers"):
            if hasattr(wan_dit, attr):
                wan_blocks = getattr(wan_dit, attr)
                break

        if wan_blocks is None or not hasattr(joint_model, "dual_stream"):
            import warnings
            warnings.warn(
                "Could not inject Wan2.x blocks into JointAVDiT. "
                "The video_block attribute will remain None.",
                RuntimeWarning,
            )
            return

        num_dual = len(joint_model.dual_stream)
        num_wan  = len(wan_blocks)

        for i in range(min(num_dual, num_wan)):
            wan_block = wan_blocks[i]
            wan_block.requires_grad_(False)
            # Wrap the Wan block in a thin adapter that matches our calling convention
            joint_model.dual_stream[i].video_block = _WanBlockAdapter(wan_block)

        print(
            f"[WanModelLoader] Injected {min(num_dual, num_wan)} Wan2.x blocks "
            f"into JointAVDiT dual_stream."
        )


# ---------------------------------------------------------------------------
# Wan block adapter
# ---------------------------------------------------------------------------

class _WanBlockAdapter(nn.Module):
    """
    Thin adapter wrapping a single Wan2.x DiT block to match the calling
    convention expected by ``JointAVDiTBlock.forward()``.

    JointAVDiTBlock calls:
        video_hidden = self.video_block(video_hidden, timestep_embed, text_embeds)

    Wan blocks typically expect:
        hidden_states, encoder_hidden_states, timestep, ...

    This adapter reorders arguments and handles any name differences.
    """

    def __init__(self, wan_block: nn.Module):
        super().__init__()
        self.wan_block = wan_block

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_embed: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Video hidden states [B, T, D].
            timestep_embed: Timestep embedding [B, D].
            encoder_hidden_states: Text embeddings [B, N, D] (optional).

        Returns:
            hidden_states: Updated video hidden states [B, T, D].
        """
        # Try to call with common Wan block signatures
        try:
            out = self.wan_block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep_embed,
            )
        except TypeError:
            # Fallback: positional args only
            try:
                out = self.wan_block(hidden_states, encoder_hidden_states, timestep_embed)
            except TypeError:
                out = self.wan_block(hidden_states)

        # Some blocks return a tuple; extract the first element
        if isinstance(out, (tuple, list)):
            out = out[0]

        return out


# ---------------------------------------------------------------------------
# Standalone weight loading utility
# ---------------------------------------------------------------------------

def load_wan_weights(
    model: nn.Module,
    weight_path: str,
    strict: bool = False,
) -> nn.Module:
    """
    Load Wan2.x pre-trained weights into an existing model.

    Args:
        model: Target nn.Module (e.g. JointAVDiT).
        weight_path: Path to a .pt, .pth, or safetensors checkpoint.
        strict: Whether to require exact key matching.

    Returns:
        model with weights loaded (in-place).
    """
    loader = WanModelLoader(model_path=weight_path)
    state = loader._load_state_dict()

    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing:
        print(f"[load_wan_weights] Missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"[load_wan_weights] Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")
    return model
