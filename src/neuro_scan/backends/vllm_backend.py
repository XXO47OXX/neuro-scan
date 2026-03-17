"""vLLM inference backend for neuro-scan.

Uses vLLM for model loading with tensor parallelism,
then accesses the underlying model for layer-level operations
(ablation, hidden states, attention).

Usage:
    neuro-scan map --model <path> --backend vllm
    neuro-scan ablate --model <path> --backend vllm
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from neuro_scan.backends.base import Backend

logger = logging.getLogger(__name__)


class VLLMBackend(Backend):
    """vLLM backend with tensor parallelism and neuroanatomy support.

    For baseline forward: accesses vLLM's underlying model directly.
    For ablation/hidden states: layer-by-layer execution on the model.
    """

    def __init__(self) -> None:
        self._llm: Any = None
        self._model: Any = None
        self._tokenizer: Any = None
        self._layers: list[Any] = []
        self._total_layers: int = 0

    def load(self, model_path: str, **kwargs) -> None:
        """Load a model via vLLM.

        Kwargs:
            dtype: Torch dtype string (default: "float16").
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            trust_remote_code: Whether to trust remote code.
            max_model_len: Maximum sequence length.
            enforce_eager: Disable CUDA graphs (default: True).
        """
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError(
                "vLLM is required for this backend.\n"
                "Install it with: pip install vllm"
            ) from None

        dtype_str = kwargs.get("dtype", "float16")
        tp_size = kwargs.get("tensor_parallel_size", 1)

        logger.info(
            "Loading model via vLLM: %s (dtype=%s, tp=%d)",
            model_path, dtype_str, tp_size,
        )

        self._llm = LLM(
            model=model_path,
            dtype=dtype_str,
            tensor_parallel_size=tp_size,
            trust_remote_code=kwargs.get("trust_remote_code", False),
            max_model_len=kwargs.get("max_model_len", None),
            enforce_eager=kwargs.get("enforce_eager", True),
        )

        self._tokenizer = self._llm.get_tokenizer()

        self._model = self._extract_underlying_model()
        if self._model is not None:
            self._layers = self._find_layers()
            self._total_layers = len(self._layers)
            logger.info("Model loaded: %d layers discovered", self._total_layers)
        else:
            raise RuntimeError(
                "Could not access underlying model from vLLM. "
                "Neuroanatomy analysis requires layer-level access."
            )

    def _extract_underlying_model(self) -> Any | None:
        """Extract the underlying HF model from vLLM internals."""
        try:
            executor = self._llm.llm_engine.model_executor
            if hasattr(executor, "driver_worker"):
                worker = executor.driver_worker
                if hasattr(worker, "model_runner"):
                    return worker.model_runner.model
        except (AttributeError, RuntimeError) as e:
            logger.debug("Could not extract model: %s", e)
        return None

    def _find_layers(self) -> list[Any]:
        """Discover decoder layers in the underlying model."""
        model = self._model
        for attr_path in [
            "model.layers",
            "transformer.h",
            "gpt_neox.layers",
            "transformer.blocks",
            "model.decoder.layers",
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
            "Could not find decoder layers in vLLM model. "
            "Supported: LLaMA, Mistral, Qwen2, GPT-2, GPT-NeoX."
        )

    def _get_base_model(self) -> Any:
        """Get the base model (before lm_head)."""
        for attr in ["model", "transformer", "gpt_neox"]:
            if hasattr(self._model, attr):
                return getattr(self._model, attr)
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
        """Apply final layer normalization."""
        base = self._get_base_model()
        if hasattr(base, "norm"):
            return base.norm(hidden_states)
        elif hasattr(base, "ln_f"):
            return base.ln_f(hidden_states)
        raise RuntimeError("Cannot find final norm layer")

    def _apply_lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to logits via LM head."""
        model = self._model
        base = self._get_base_model()
        if hasattr(model, "lm_head"):
            return model.lm_head(hidden_states)
        elif hasattr(base, "lm_head"):
            return base.lm_head(hidden_states)
        raise RuntimeError("Cannot find LM head")

    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text, return input_ids on model device."""
        device = next(self._model.parameters()).device
        inputs = self._tokenizer(text, return_tensors="pt")
        return inputs["input_ids"].to(device)

    @staticmethod
    def _compute_position_embeddings(base, hidden_states, position_ids):
        """Pre-compute rotary embeddings if available."""
        if hasattr(base, "rotary_emb"):
            return base.rotary_emb(hidden_states, position_ids)
        return None

    def _layer_forward(self, layer, hidden_states, position_ids, position_embeddings=None):
        """Execute a single decoder layer."""
        kwargs: dict[str, Any] = {
            "attention_mask": None,
            "position_ids": position_ids,
        }
        if position_embeddings is not None:
            kwargs["position_embeddings"] = position_embeddings

        output = layer(hidden_states, **kwargs)
        return output[0] if isinstance(output, tuple) else output

    def get_total_layers(self) -> int:
        if self._total_layers == 0:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._total_layers

    def get_tokenizer(self):
        if self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._tokenizer

    @torch.no_grad()
    def forward(self, text: str) -> torch.Tensor:
        """Standard forward pass returning last-token logits."""
        input_ids = self._tokenize(text)
        hidden_states = self._get_embeddings(input_ids)
        base = self._get_base_model()

        position_ids = torch.arange(
            input_ids.shape[1], device=input_ids.device
        ).unsqueeze(0)
        pos_emb = self._compute_position_embeddings(base, hidden_states, position_ids)

        for layer in self._layers:
            hidden_states = self._layer_forward(layer, hidden_states, position_ids, pos_emb)

        normed = self._apply_norm(hidden_states)
        logits = self._apply_lm_head(normed)
        return logits[0, -1, :]

    @torch.no_grad()
    def forward_with_ablation(
        self, text: str, ablated_layers: list[int]
    ) -> torch.Tensor:
        """Forward pass with specified layers skipped (identity bypass)."""
        input_ids = self._tokenize(text)
        hidden_states = self._get_embeddings(input_ids)
        base = self._get_base_model()
        ablated_set = set(ablated_layers)

        position_ids = torch.arange(
            input_ids.shape[1], device=input_ids.device
        ).unsqueeze(0)
        pos_emb = self._compute_position_embeddings(base, hidden_states, position_ids)

        for idx, layer in enumerate(self._layers):
            if idx in ablated_set:
                continue
            hidden_states = self._layer_forward(layer, hidden_states, position_ids, pos_emb)

        normed = self._apply_norm(hidden_states)
        logits = self._apply_lm_head(normed)
        return logits[0, -1, :]

    @torch.no_grad()
    def forward_with_hidden_states(
        self, text: str
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass collecting hidden states after each layer."""
        input_ids = self._tokenize(text)
        hidden_states = self._get_embeddings(input_ids)
        base = self._get_base_model()

        position_ids = torch.arange(
            input_ids.shape[1], device=input_ids.device
        ).unsqueeze(0)
        pos_emb = self._compute_position_embeddings(base, hidden_states, position_ids)

        all_hidden = []
        for layer in self._layers:
            hidden_states = self._layer_forward(layer, hidden_states, position_ids, pos_emb)
            all_hidden.append(hidden_states[0].clone())

        normed = self._apply_norm(hidden_states)
        logits = self._apply_lm_head(normed)
        return logits[0, -1, :], all_hidden

    @torch.no_grad()
    def forward_with_attention(
        self, text: str
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass returning per-layer attention weights.

        Note: vLLM's optimized kernels may not support attention extraction.
        Falls back to manual forward with output_attentions where possible.
        """
        raise NotImplementedError(
            "Attention extraction is not supported with vLLM backend. "
            "Use the transformers backend for attention analysis."
        )

    def get_norm_and_head(self) -> tuple:
        """Return (norm_fn, lm_head_fn) for logit lens projection."""
        return self._apply_norm, self._apply_lm_head

    def cleanup(self) -> None:
        """Release resources."""
        if self._llm is not None:
            del self._llm
            self._llm = None
        self._model = None
        self._tokenizer = None
        self._layers = []
        self._total_layers = 0

        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
