"""Abstract backend interface for neuro-scan.

Extends the layer-scan backend interface with neuroanatomy methods:
ablation, hidden state extraction, and attention weight extraction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Backend(ABC):
    """Abstract base for inference backends.

    A backend must provide:
    1. Model loading and tokenization
    2. Standard forward pass
    3. Forward pass with layer ablation (zeroing out layers)
    4. Forward pass returning per-layer hidden states
    5. Forward pass returning per-layer attention weights
    """

    @abstractmethod
    def load(self, model_path: str, **kwargs) -> None:
        """Load a model from the given path or HuggingFace ID."""

    @abstractmethod
    def get_total_layers(self) -> int:
        """Return the total number of transformer decoder layers."""

    @abstractmethod
    def get_tokenizer(self):
        """Return the model's tokenizer."""

    @abstractmethod
    def forward(self, text: str) -> torch.Tensor:
        """Run standard forward pass (baseline).

        Returns:
            Logits tensor of shape (vocab_size,) for the last token position.
        """

    @abstractmethod
    def forward_with_ablation(
        self, text: str, ablated_layers: list[int]
    ) -> torch.Tensor:
        """Run forward pass with specified layers skipped (identity bypass).

        The ablated layers pass their input directly to output without
        any transformation, effectively removing their contribution.

        Args:
            text: Input text to process.
            ablated_layers: List of layer indices to ablate.

        Returns:
            Logits tensor of shape (vocab_size,) for the last token position.
        """

    @abstractmethod
    def forward_with_hidden_states(
        self, text: str
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run forward pass returning per-layer hidden states.

        Used for logit lens analysis: project each layer's hidden state
        through the LM head to see when the correct token emerges.

        Args:
            text: Input text to process.

        Returns:
            Tuple of (final_logits, [hidden_state_per_layer]).
            Each hidden state has shape (seq_len, hidden_dim).
        """

    @abstractmethod
    def forward_with_attention(
        self, text: str
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run forward pass returning per-layer attention weights.

        Args:
            text: Input text to process.

        Returns:
            Tuple of (final_logits, [attention_weights_per_layer]).
            Each attention tensor has shape (num_heads, seq_len, seq_len).
        """

    def get_norm_and_head(self) -> tuple:
        """Return (norm_fn, lm_head_fn) for logit lens projection.

        Returns:
            Tuple of (final_norm_module, lm_head_module).

        Raises:
            NotImplementedError: If backend doesn't support logit lens.
        """
        raise NotImplementedError("This backend does not expose norm/head for logit lens")

    def cleanup(self) -> None:
        """Release resources (GPU memory, etc.). Optional."""
