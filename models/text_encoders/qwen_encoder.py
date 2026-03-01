"""
Qwen2.5 Text Encoder

Secondary text encoder using Qwen2.5-32B-Instruct.
Acts as a higher-capacity companion to the primary Qwen3 encoder,
providing richer semantic understanding for complex prompts.

Primary encoder:  Qwen3-8B  → 4096-dim  (see qwen3_encoder.py)
Secondary encoder: Qwen2.5-32B → 5120-dim → projected to 4096-dim

In most training stages only the primary (Qwen3) encoder is used.
The secondary encoder is an optional enhancement for inference quality.
"""

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn


class QwenEncoder(nn.Module):
    """
    Qwen2.5 Text Encoder wrapper for use as a secondary text encoder.

    Loads Qwen2.5-32B-Instruct (or a smaller variant) via HuggingFace
    transformers and extracts per-token hidden states from the final layer.

    Usage:
        encoder = QwenEncoder(model_path="Qwen/Qwen2.5-32B-Instruct")
        encoder.load_model()
        embeds, mask = encoder.encode(["A cat playing piano"])
    """

    VARIANT_HIDDEN_SIZES = {
        "7b":  3584,
        "14b": 5120,
        "32b": 5120,
        "72b": 8192,
    }

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-32B-Instruct",
        variant: Literal["7b", "14b", "32b", "72b"] = "32b",
        target_dim: int = 4096,
        max_length: int = 1024,
        freeze: bool = True,
        pooling: Literal["last_hidden", "mean", "eos"] = "last_hidden",
        device: str = "cuda",
    ):
        """
        Args:
            model_path: HuggingFace model ID or local path.
            variant: Model size variant, determines hidden_size.
            target_dim: Output embedding dimension (default 4096).
            max_length: Maximum tokenised sequence length.
            freeze: Freeze encoder weights (recommended).
            pooling: How to extract sequence features:
                "last_hidden" → [B, N, target_dim]
                "mean"        → [B, target_dim] (mean over valid tokens)
                "eos"         → [B, target_dim] (last valid token)
            device: Device for the model.
        """
        super().__init__()

        self.model_path = model_path
        self.variant = variant
        self.max_length = max_length
        self.freeze = freeze
        self.pooling = pooling
        self.device = device

        self.hidden_size = self.VARIANT_HIDDEN_SIZES.get(variant)
        if self.hidden_size is None:
            raise ValueError(
                f"Unknown variant '{variant}'. Supported: {list(self.VARIANT_HIDDEN_SIZES)}"
            )

        self.target_dim = target_dim
        self.output_dim = target_dim

        # Linear projection if hidden_size ≠ target_dim
        if self.hidden_size != target_dim:
            self.proj = nn.Sequential(
                nn.LayerNorm(self.hidden_size),
                nn.Linear(self.hidden_size, target_dim),
            )
        else:
            self.proj = nn.Identity()

        # Lazy-loaded model and tokenizer
        self.tokenizer = None
        self.model = None

    def load_model(self) -> None:
        """
        Load Qwen2.5 model and tokenizer.

        Call this once after __init__ to control when the large model
        is loaded into GPU memory.
        """
        from transformers import AutoTokenizer, AutoModel

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)

        if self.freeze:
            self.model.requires_grad_(False)
            self.model.eval()

    @torch.no_grad()
    def encode(
        self,
        texts: list,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of texts.

        Args:
            texts: List of prompt strings.

        Returns:
            embeddings: [B, N, target_dim] or [B, target_dim] depending on pooling.
            mask: [B, N] attention mask.

        Raises:
            RuntimeError: If load_model() has not been called.
        """
        if self.tokenizer is None or self.model is None:
            raise RuntimeError(
                "QwenEncoder.load_model() must be called before encode()."
            )

        inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state  # [B, N, hidden_size]
        mask   = inputs["attention_mask"]   # [B, N]

        hidden = self.proj(hidden)          # [B, N, target_dim]

        if self.pooling == "last_hidden":
            return hidden, mask
        elif self.pooling == "mean":
            mask_exp = mask.unsqueeze(-1).float()
            pooled = (hidden * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
            return pooled, mask
        elif self.pooling == "eos":
            lengths = mask.sum(dim=1).long() - 1
            batch_idx = torch.arange(hidden.size(0), device=hidden.device)
            pooled = hidden[batch_idx, lengths]
            return pooled, mask
        else:
            raise ValueError(f"Unknown pooling mode: '{self.pooling}'")

    def forward(self, texts: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass (alias for encode)."""
        return self.encode(texts)


def create_qwen_encoder(config: dict) -> QwenEncoder:
    """
    Factory function to create QwenEncoder from config.

    Args:
        config: Dict with keys:
            model (str): HuggingFace model path.
            variant (str): "7b", "14b", "32b", or "72b".
            target_dim (int): Output dimension (default 4096).
            max_length (int): Max token length (default 1024).
            freeze (bool): Freeze backbone (default True).
            pooling (str): "last_hidden", "mean", or "eos".

    Returns:
        QwenEncoder (model NOT yet loaded — call .load_model()).
    """
    return QwenEncoder(
        model_path=config.get("model", "Qwen/Qwen2.5-32B-Instruct"),
        variant=config.get("variant", "32b"),
        target_dim=config.get("target_dim", 4096),
        max_length=config.get("max_length", 1024),
        freeze=config.get("freeze", True),
        pooling=config.get("pooling", "last_hidden"),
    )
