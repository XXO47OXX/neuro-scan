from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from neuro_scan.backends.base import Backend

logger = logging.getLogger(__name__)


class AffineTranslator(nn.Module):
    """Per-layer affine probe: y = x @ W + b.

    Initialized to identity (W=I, b=0) so that untrained lens
    degrades to vanilla logit lens.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.weight + self.bias


class TunedLens:
    """Collection of per-layer affine translators.

    Usage:
        lens = TunedLens.train(backend, calibration_texts)
        lens.save("lens.safetensors")

        lens = TunedLens.load("lens.safetensors")
        logits = lens.project(hidden_state, layer_idx, norm_fn, head_fn)
    """

    def __init__(self, translators: list[AffineTranslator]):
        self.translators = translators

    @classmethod
    def train(
        cls,
        backend: Backend,
        calibration_texts: list[str],
        steps: int = 250,
        lr: float = 1.0,
        grad_clip: float = 1.0,
    ) -> TunedLens:
        """Train per-layer affine probes using KL divergence to final logits.

        Args:
            backend: Loaded backend with model.
            calibration_texts: Texts to use for calibration (~2K-8K tokens).
            steps: Number of SGD steps per translator.
            lr: Learning rate for SGD with Nesterov momentum.
            grad_clip: Gradient clipping value.

        Returns:
            Trained TunedLens instance.
        """
        norm_fn, head_fn = backend.get_norm_and_head()
        total_layers = backend.get_total_layers()

        # Collect hidden states from calibration texts
        logger.info("Collecting hidden states from %d calibration texts...", len(calibration_texts))

        all_hidden_states: list[list[torch.Tensor]] = [[] for _ in range(total_layers)]
        all_final_logits: list[torch.Tensor] = []

        for text in calibration_texts:
            final_logits, hidden_states = backend.forward_with_hidden_states(text)
            # Take last token position from each layer
            for layer_idx, hs in enumerate(hidden_states):
                all_hidden_states[layer_idx].append(hs[-1].detach().clone())
            # Get final logits target distribution
            all_final_logits.append(final_logits.detach().clone())

        # Determine d_model from first hidden state
        d_model = all_hidden_states[0][0].shape[-1]
        device = all_hidden_states[0][0].device

        # Stack calibration data
        final_log_probs = F.log_softmax(torch.stack(all_final_logits).float(), dim=-1)

        # Detect model dtype for norm/head compatibility
        device_dtype = all_hidden_states[0][0].dtype

        # Train one translator per layer
        translators = []
        logger.info("Training %d translators (d_model=%d, steps=%d)...", total_layers, d_model, steps)

        for layer_idx in range(total_layers):
            translator = AffineTranslator(d_model).to(device)
            optimizer = torch.optim.SGD(
                translator.parameters(), lr=lr, momentum=0.9, nesterov=True
            )

            layer_hs = torch.stack(all_hidden_states[layer_idx]).float()  # (N, d_model)

            for step in range(steps):
                optimizer.zero_grad()

                # Project through translator + norm + head
                translated = translator(layer_hs)
                # Apply norm and head to get logits
                # Cast to model dtype for norm/head compatibility, then back to float
                normed = norm_fn(translated.unsqueeze(1).to(device_dtype))  # (N, 1, d_model)
                logits = head_fn(normed)[:, 0, :].float()  # (N, vocab_size)

                tuned_log_probs = F.log_softmax(logits.float(), dim=-1)

                # KL divergence: KL(final || tuned)
                # = sum(final * (log_final - log_tuned))
                kl = F.kl_div(tuned_log_probs, final_log_probs, log_target=True, reduction="batchmean")

                kl.backward()
                torch.nn.utils.clip_grad_norm_(translator.parameters(), grad_clip)
                optimizer.step()

            translators.append(translator)
            if (layer_idx + 1) % 8 == 0 or layer_idx == total_layers - 1:
                logger.info("  Trained layer %d/%d (final KL=%.4f)", layer_idx + 1, total_layers, kl.item())

        return cls(translators)

    def project(
        self,
        hidden_state: torch.Tensor,
        layer_idx: int,
        norm_fn,
        head_fn,
    ) -> torch.Tensor:
        """Project a hidden state through the tuned lens to logits.

        Args:
            hidden_state: Hidden state tensor of shape (..., d_model).
            layer_idx: Which layer this hidden state came from.
            norm_fn: Final layer normalization function.
            head_fn: LM head function.

        Returns:
            Logits tensor.
        """
        translator = self.translators[layer_idx]
        # Ensure translator is on the same device as hidden_state
        device = hidden_state.device
        translator = translator.to(device)
        translated = translator(hidden_state.float())
        # Cast back to hidden_state dtype for norm/head compatibility
        normed = norm_fn(translated.to(hidden_state.dtype))
        return head_fn(normed)

    def save(self, path: str | Path) -> None:
        """Save all translators to a safetensors file."""
        from safetensors.torch import save_file

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state_dict = {}
        for i, translator in enumerate(self.translators):
            state_dict[f"layer_{i}_weight"] = translator.weight.data.cpu()
            state_dict[f"layer_{i}_bias"] = translator.bias.data.cpu()

        # Store metadata
        save_file(state_dict, str(path))
        logger.info("Tuned lens saved to %s (%d layers)", path, len(self.translators))

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> TunedLens:
        """Load translators from a safetensors file."""
        from safetensors.torch import load_file

        path = Path(path)
        state_dict = load_file(str(path))

        # Discover how many layers are stored
        layer_indices = set()
        for key in state_dict:
            if key.startswith("layer_") and key.endswith("_weight"):
                idx = int(key.split("_")[1])
                layer_indices.add(idx)

        num_layers = max(layer_indices) + 1
        d_model = state_dict["layer_0_weight"].shape[0]

        translators = []
        for i in range(num_layers):
            translator = AffineTranslator(d_model)
            translator.weight.data = state_dict[f"layer_{i}_weight"].to(device)
            translator.bias.data = state_dict[f"layer_{i}_bias"].to(device)
            translators.append(translator)

        logger.info("Tuned lens loaded from %s (%d layers, d_model=%d)", path, num_layers, d_model)
        return cls(translators)
