"""
T5 Text Encoder

Wrapper for T5-based text encoders used as a compatibility fallback
for Wan2.x's original T5-XXL text encoding pipeline.

In Alive-Wan2X, Qwen3-8B is the *primary* text encoder (qwen3_encoder.py).
This T5 wrapper is provided for:
  1. Compatibility with Wan2.x pre-trained model pipelines
  2. Optional use as a secondary encoder for ablation studies
  3. Smooth migration path from Wan2.x's original text encoding

Supported models:
  google/flan-t5-xxl   — T5-XXL encoder-only, hidden_size=4096  (recommended)
  google/t5-v1_1-xxl   — T5 v1.1 XXL, hidden_size=4096
  google/t5-11b        — T5 11B encoder-only, hidden_size=1024 (requires projection)
"""

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn


class T5Encoder(nn.Module):
    """
    T5 Text Encoder wrapper.

    Uses T5EncoderModel (encoder-only) to extract per-token hidden states.
    The full T5 decoder is not loaded, saving significant memory.

    Usage:
        encoder = T5Encoder(model_path="google/flan-t5-xxl")
        encoder.load_model()
        embeds, mask = encoder.encode(["A cat playing piano"])
    """

    # Known hidden sizes per model family
    HIDDEN_SIZES = {
        "flan-t5-xxl": 4096,
        "t5-v1_1-xxl": 4096,
        "t5-11b":       1024,
        "t5-3b":        1024,
        "t5-large":      1024,
        "t5-base":       768,
    }

    def __init__(
        self,
        model_path: str = "google/flan-t5-xxl",
        max_length: int = 512,
        target_dim: int = 4096,
        freeze: bool = True,
        device: str = "cuda",
    ):
        """
        Args:
            model_path: HuggingFace model ID or local path.
            max_length: Maximum tokenised sequence length.
            target_dim: Output embedding dimension (default 4096).
                        A projection layer is added when hidden_size ≠ target_dim.
            freeze: Freeze T5 weights (strongly recommended).
            device: Device to load the model on.
        """
        super().__init__()

        self.model_path = model_path
        self.max_length = max_length
        self.target_dim = target_dim
        self.output_dim = target_dim
        self.freeze = freeze
        self.device = device

        # Infer hidden size from model name
        detected_hidden = 4096  # default
        for key, size in self.HIDDEN_SIZES.items():
            if key in model_path.lower():
                detected_hidden = size
                break
        self.hidden_size = detected_hidden

        # Projection layer if hidden_size ≠ target_dim
        if self.hidden_size != target_dim:
            self.proj = nn.Sequential(
                nn.LayerNorm(self.hidden_size),
                nn.Linear(self.hidden_size, target_dim),
            )
        else:
            self.proj = nn.Identity()

        # Lazy-loaded model + tokenizer
        self.tokenizer = None
        self.model = None

    def load_model(self) -> None:
        """
        Load T5EncoderModel and tokenizer.

        Uses T5EncoderModel (encoder-only) to avoid loading the decoder,
        which halves memory usage compared to the full T5 model.

        Call this once after __init__ to control when the model is loaded.
        """
        from transformers import T5EncoderModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        # T5EncoderModel loads only the encoder stack (no decoder)
        self.model = T5EncoderModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
        ).to(self.device)

        if self.freeze:
            self.model.requires_grad_(False)
            self.model.eval()

        # Update hidden_size from actual model config
        if hasattr(self.model.config, "d_model"):
            self.hidden_size = self.model.config.d_model
            if self.hidden_size != self.target_dim and isinstance(self.proj, nn.Identity):
                self.proj = nn.Sequential(
                    nn.LayerNorm(self.hidden_size),
                    nn.Linear(self.hidden_size, self.target_dim),
                ).to(self.device)

    @torch.no_grad()
    def encode(
        self,
        texts: list,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of text strings.

        Args:
            texts: List of prompt strings.

        Returns:
            embeddings: [B, N, target_dim] per-token hidden states.
            mask: [B, N] attention mask (1 = valid token, 0 = padding).

        Raises:
            RuntimeError: If load_model() has not been called.
        """
        if self.tokenizer is None or self.model is None:
            raise RuntimeError(
                "T5Encoder.load_model() must be called before encode()."
            )

        inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        # T5EncoderModel.forward returns BaseModelOutput with last_hidden_state
        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        hidden = outputs.last_hidden_state   # [B, N, hidden_size]
        mask   = inputs["attention_mask"]    # [B, N]

        # Project to target dimension
        hidden = self.proj(hidden)           # [B, N, target_dim]

        return hidden, mask

    def forward(self, texts: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass (alias for encode)."""
        return self.encode(texts)


def create_t5_encoder(config: dict) -> T5Encoder:
    """
    Factory function to create T5Encoder from config.

    Args:
        config: Dict with keys:
            model (str): HuggingFace model path (default "google/flan-t5-xxl").
            max_length (int): Max token length (default 512).
            target_dim (int): Output embedding dim (default 4096).
            freeze (bool): Freeze backbone (default True).

    Returns:
        T5Encoder (model NOT yet loaded — call .load_model()).
    """
    return T5Encoder(
        model_path=config.get("model", "google/flan-t5-xxl"),
        max_length=config.get("max_length", 512),
        target_dim=config.get("target_dim", 4096),
        freeze=config.get("freeze", True),
    )
