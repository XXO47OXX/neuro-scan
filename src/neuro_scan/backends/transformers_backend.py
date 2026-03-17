"""HuggingFace Transformers backend — reference implementation for neuro-scan.

Extends the layer-scan Transformers backend with neuroanatomy methods:
layer ablation, hidden state extraction, and attention weight extraction.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from neuro_scan.backends.base import Backend

logger = logging.getLogger(__name__)


class TransformersBackend(Backend):
    """HuggingFace Transformers backend with neuroanatomy analysis methods."""

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._layers: list[Any] = []
        self._total_layers: int = 0

    def load(self, model_path: str, **kwargs) -> None:
        """Load a CausalLM model from path or HuggingFace ID.

        Kwargs:
            dtype: Torch dtype string (default: "float16").
            device_map: Device map for model parallelism (default: "auto").
            trust_remote_code: Whether to trust remote code (default: False).
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_str = kwargs.get("dtype", "float16")
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(dtype_str, torch.float16)

        logger.info("Loading tokenizer from %s", model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=kwargs.get("trust_remote_code", False),
        )

        logger.info("Loading model from %s (dtype=%s)", model_path, dtype_str)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=kwargs.get("device_map", "auto"),
            trust_remote_code=kwargs.get("trust_remote_code", False),
        )
        self._model.eval()

        self._layers = self._find_layers()
        self._total_layers = len(self._layers)
        logger.info("Model loaded: %d layers discovered", self._total_layers)

    def _find_layers(self) -> list[Any]:
        """Discover the decoder layers in the model."""
        model = self._model

        for attr_path in [
            "model.layers",              # LLaMA, Mistral, Qwen2
            "transformer.h",             # GPT-2, GPT-Neo
            "gpt_neox.layers",           # GPT-NeoX, Pythia
            "transformer.blocks",        # MPT
            "model.decoder.layers",      # OPT
        ]:
            obj = model
            found = True
            for part in attr_path.split("."):
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    found = False
                    break
            if found and hasattr(obj, "__len__"):
                return list(obj)

        raise RuntimeError(
            "Could not find decoder layers. Supported architectures: "
            "LLaMA, Mistral, Qwen2, GPT-2, GPT-NeoX, MPT, OPT. "
            "For other architectures, use a custom backend."
        )

    def get_total_layers(self) -> int:
        if self._total_layers == 0:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._total_layers

    def get_tokenizer(self):
        if self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._tokenizer

    def _get_base_model(self):
        """Get the base model object (before lm_head)."""
        model = self._model
        if hasattr(model, "model"):
            return model.model  # LLaMA-style
        elif hasattr(model, "transformer"):
            return model.transformer  # GPT-style
        raise RuntimeError("Cannot determine base model location")

    def _get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute token embeddings from input IDs."""
        base = self._get_base_model()
        if hasattr(base, "embed_tokens"):
            return base.embed_tokens(input_ids)
        elif hasattr(base, "wte"):
            return base.wte(input_ids)
        raise RuntimeError("Cannot find embedding layer")

    def _apply_norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the final layer norm."""
        base = self._get_base_model()
        if hasattr(base, "norm"):
            return base.norm(hidden_states)
        elif hasattr(base, "ln_f"):
            return base.ln_f(hidden_states)
        raise RuntimeError("Cannot find final norm layer")

    def _apply_lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project through the LM head to get logits."""
        model = self._model
        base = self._get_base_model()
        if hasattr(model, "lm_head"):
            return model.lm_head(hidden_states)
        elif hasattr(base, "lm_head"):
            return base.lm_head(hidden_states)
        raise RuntimeError("Cannot find LM head for logit projection")

    def _tokenize(self, text: str) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Tokenize text and return (input_ids, attention_mask) on model device."""
        inputs = self._tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self._model.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._model.device)
        return input_ids, attention_mask

    @torch.no_grad()
    def forward(self, text: str) -> torch.Tensor:
        """Standard forward pass returning last-token logits."""
        input_ids, attention_mask = self._tokenize(text)
        outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits[0, -1, :]

    @staticmethod
    def _compute_position_embeddings(
        base_model,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        """Pre-compute rotary position embeddings (cos, sin).

        Transformers 5.x moved rotary_emb out of individual decoder layers.
        Returns None for architectures without rotary embeddings (e.g. GPT-2).
        """
        if hasattr(base_model, "rotary_emb"):
            return base_model.rotary_emb(hidden_states, position_ids)
        return None

    def _layer_forward(
        self,
        layer,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        position_embeddings=None,
    ) -> torch.Tensor:
        """Call a single decoder layer with correct kwargs.

        Note: attention_mask is set to None for direct layer calls because
        the raw tokenizer mask (long int) is incompatible with SDPA. For
        single-sequence inference there is no padding, so None is correct.
        """
        kwargs: dict = {
            "attention_mask": None,
            "position_ids": position_ids,
        }
        if position_embeddings is not None:
            kwargs["position_embeddings"] = position_embeddings

        layer_output = layer(hidden_states, **kwargs)
        if isinstance(layer_output, tuple):
            return layer_output[0]
        return layer_output

    @torch.no_grad()
    def forward_with_ablation(
        self, text: str, ablated_layers: list[int]
    ) -> torch.Tensor:
        """Forward pass with specified layers skipped (identity bypass).

        Ablated layers pass input through unchanged, effectively removing
        their learned transformation.
        """
        input_ids, attention_mask = self._tokenize(text)
        hidden_states = self._get_embeddings(input_ids)

        ablated_set = set(ablated_layers)
        position_ids = torch.arange(
            input_ids.shape[1], device=input_ids.device
        ).unsqueeze(0)

        base = self._get_base_model()
        position_embeddings = self._compute_position_embeddings(
            base, hidden_states, position_ids
        )

        for layer_idx, layer in enumerate(self._layers):
            if layer_idx in ablated_set:
                continue

            hidden_states = self._layer_forward(
                layer, hidden_states, attention_mask,
                position_ids, position_embeddings,
            )

        hidden_states = self._apply_norm(hidden_states)
        logits = self._apply_lm_head(hidden_states)
        return logits[0, -1, :]

    @torch.no_grad()
    def forward_with_hidden_states(
        self, text: str
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass collecting hidden states after each layer.

        Returns:
            (final_logits, [hidden_state_per_layer]).
            Each hidden state is shape (seq_len, hidden_dim).
        """
        input_ids, attention_mask = self._tokenize(text)
        hidden_states = self._get_embeddings(input_ids)

        position_ids = torch.arange(
            input_ids.shape[1], device=input_ids.device
        ).unsqueeze(0)

        base = self._get_base_model()
        position_embeddings = self._compute_position_embeddings(
            base, hidden_states, position_ids
        )

        all_hidden_states = []

        for layer in self._layers:
            hidden_states = self._layer_forward(
                layer, hidden_states, attention_mask,
                position_ids, position_embeddings,
            )

            # Store a copy of hidden states after this layer
            all_hidden_states.append(hidden_states[0].clone())

        normed = self._apply_norm(hidden_states)
        logits = self._apply_lm_head(normed)

        return logits[0, -1, :], all_hidden_states

    @torch.no_grad()
    def forward_with_attention(
        self, text: str
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass returning per-layer attention weights.

        Uses output_attentions=True to get attention from the model.
        """
        input_ids, attention_mask = self._tokenize(text)

        outputs = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )

        logits = outputs.logits[0, -1, :]
        # outputs.attentions is tuple of (batch, heads, seq, seq) per layer
        attention_weights = [attn[0] for attn in outputs.attentions]

        return logits, attention_weights

    def get_norm_and_head(self) -> tuple:
        """Return (norm_fn, lm_head_fn) for logit lens projection."""
        return self._apply_norm, self._apply_lm_head

    def cleanup(self) -> None:
        """Free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._layers = []
        self._total_layers = 0

        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
