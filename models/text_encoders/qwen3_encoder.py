"""
Qwen3 Text Encoder

Primary text encoder for Alive-Wan2X, replacing T5-XXL.
Supports Qwen3-8B (hidden_size=4096) and Qwen3-4B (hidden_size=2560).

Qwen3-8B: hidden_size=4096, drop-in replacement for T5's output dim.
Qwen3-4B: hidden_size=2560, requires projection to 4096.
"""

from typing import Optional, Tuple, Literal
import torch
import torch.nn as nn


class Qwen3Encoder(nn.Module):
    """
    Qwen3 Text Encoder for Alive-Wan2X.

    Uses Qwen3 as a text feature extractor by taking hidden states
    from the last layer (no autoregressive generation).

    Architecture (Qwen3-8B / Qwen3-4B):
    - Qwen3-8B: 36 layers, hidden_size=4096, 32 Q-heads, 8 KV-heads, head_dim=128
    - Qwen3-4B: 36 layers, hidden_size=2560, 32 Q-heads, 8 KV-heads, head_dim=128

    When hidden_size != target_dim, a learned linear projection is added.
    """

    # Known hidden sizes per variant
    VARIANT_HIDDEN_SIZES = {
        "8b": 4096,
        "4b": 2560,
    }

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-8B",
        variant: Literal["8b", "4b"] = "8b",
        target_dim: int = 4096,
        max_length: int = 512,
        freeze: bool = True,
        pooling: Literal["last_hidden", "mean", "eos"] = "last_hidden",
        device: str = "cuda",
    ):
        """
        Initialize Qwen3 Encoder.

        Args:
            model_path: HuggingFace model path or local path.
            variant: Model variant, "8b" or "4b". Determines hidden_size.
            target_dim: Target output dimension. If hidden_size != target_dim,
                        a learned projection is inserted.
            max_length: Maximum token sequence length.
            freeze: Freeze backbone weights (recommended for text encoder role).
            pooling: How to extract sequence-level features.
                     "last_hidden" — return full sequence hidden states [B, N, D].
                     "mean" — mean-pool over non-padding tokens → [B, D].
                     "eos" — take the last non-padding token → [B, D].
            device: Device for model and tensors.
        """
        super().__init__()

        self.model_path = model_path
        self.variant = variant
        self.max_length = max_length
        self.freeze = freeze
        self.pooling = pooling
        self.device = device

        # Resolve hidden size from variant
        self.hidden_size = self.VARIANT_HIDDEN_SIZES.get(variant)
        if self.hidden_size is None:
            raise ValueError(
                f"Unknown variant '{variant}'. Supported: {list(self.VARIANT_HIDDEN_SIZES)}"
            )

        self.target_dim = target_dim
        self.output_dim = target_dim

        # Projection if hidden_size != target_dim (e.g. 4B: 2560 → 4096)
        if self.hidden_size != target_dim:
            self.proj = nn.Sequential(
                nn.LayerNorm(self.hidden_size),
                nn.Linear(self.hidden_size, target_dim),
            )
        else:
            self.proj = nn.Identity()

        # Lazy-loaded model and tokenizer (populated by load_model())
        self.tokenizer = None
        self.model = None

    def load_model(self) -> None:
        """
        Load the Qwen3 model and tokenizer from model_path.

        Call this explicitly after __init__ to control when the large model
        is loaded into GPU memory.
        """
        from transformers import AutoTokenizer, AutoModel

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        # Use AutoModel (not CausalLM) to get hidden states without the LM head
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
            embeddings: [B, N, target_dim] — per-token hidden states
                        (or [B, target_dim] if pooling is "mean"/"eos").
            mask: [B, N] attention mask (1 = valid token, 0 = padding).

        Raises:
            RuntimeError: If load_model() has not been called.
        """
        if self.tokenizer is None or self.model is None:
            raise RuntimeError(
                "Qwen3Encoder.load_model() must be called before encode(). "
                "Model and tokenizer are not loaded yet."
            )

        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        # Forward through backbone
        outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state  # [B, N, hidden_size]
        mask = inputs["attention_mask"]      # [B, N]

        # Project to target dim if needed (e.g. 2560 → 4096)
        hidden = self.proj(hidden)           # [B, N, target_dim]

        # Pooling
        if self.pooling == "last_hidden":
            return hidden, mask
        elif self.pooling == "mean":
            # Mean pool over valid tokens
            mask_expanded = mask.unsqueeze(-1).float()  # [B, N, 1]
            pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            return pooled, mask
        elif self.pooling == "eos":
            # Take hidden state at last valid token position
            lengths = mask.sum(dim=1).long() - 1  # [B]
            batch_idx = torch.arange(hidden.size(0), device=hidden.device)
            pooled = hidden[batch_idx, lengths]    # [B, target_dim]
            return pooled, mask
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling}")

    def forward(
        self,
        texts: list,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass (alias for encode).

        Args:
            texts: List of prompt strings.

        Returns:
            embeddings, mask
        """
        return self.encode(texts)


def create_qwen3_encoder(config: dict) -> Qwen3Encoder:
    """
    Factory function to create Qwen3Encoder from config dict.

    Expected config keys:
        model: str — HuggingFace model path
        variant: str — "8b" or "4b"
        target_dim: int — target output dimension (default 4096)
        max_length: int — max token length (default 512)
        freeze: bool — freeze backbone (default True)
        pooling: str — "last_hidden", "mean", or "eos" (default "last_hidden")

    Returns:
        Qwen3Encoder instance (model NOT yet loaded — call .load_model()).
    """
    return Qwen3Encoder(
        model_path=config.get("model", "Qwen/Qwen3-8B"),
        variant=config.get("variant", "8b"),
        target_dim=config.get("target_dim", 4096),
        max_length=config.get("max_length", 512),
        freeze=config.get("freeze", True),
        pooling=config.get("pooling", "last_hidden"),
    )
