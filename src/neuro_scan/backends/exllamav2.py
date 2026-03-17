from __future__ import annotations

import logging

import torch

from neuro_scan.backends.base import Backend

logger = logging.getLogger(__name__)


class ExLlamaV2Backend(Backend):
    """ExLlamaV2 backend with neuroanatomy analysis methods."""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._cache = None
        self._config = None
        self._total_layers: int = 0

    def load(self, model_path: str, **kwargs) -> None:
        """Load an EXL2/GPTQ quantized model.

        Kwargs:
            gpu_split: List of GPU memory limits in MB.
            max_seq_len: Maximum sequence length (default: 4096).
            rope_scale: RoPE scaling factor (default: 1.0).
        """
        try:
            from exllamav2 import (
                ExLlamaV2,
                ExLlamaV2Cache,
                ExLlamaV2Config,
                ExLlamaV2Tokenizer,
            )
        except ImportError as e:
            raise ImportError(
                "ExLlamaV2 not installed. Install with: "
                "pip install neuro-scan[exllamav2]"
            ) from e

        logger.info("Loading ExLlamaV2 model from %s", model_path)

        self._config = ExLlamaV2Config(model_path)
        self._config.max_seq_len = kwargs.get("max_seq_len", 4096)

        if "rope_scale" in kwargs:
            self._config.scale_pos_emb = kwargs["rope_scale"]

        self._model = ExLlamaV2(self._config)

        gpu_split = kwargs.get("gpu_split")
        if gpu_split:
            self._model.load(gpu_split)
        else:
            self._model.load_autosplit()

        self._tokenizer = ExLlamaV2Tokenizer(self._config)
        self._cache = ExLlamaV2Cache(self._model, max_seq_len=self._config.max_seq_len)

        self._total_layers = self._count_decoder_layers()
        logger.info("ExLlamaV2 model loaded: %d decoder layers", self._total_layers)

    def _count_decoder_layers(self) -> int:
        """Count the number of decoder layers in the ExLlamaV2 model."""
        count = 0
        for module in self._model.modules:
            module_name = type(module).__name__
            if "Attention" in module_name:
                count += 1
        return count

    def get_total_layers(self) -> int:
        if self._total_layers == 0:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._total_layers

    def get_tokenizer(self):
        """Return a tokenizer adapter compatible with HuggingFace interface."""
        if self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return _ExLlamaV2TokenizerAdapter(self._tokenizer)

    def _get_layer_module_map(self) -> dict[int, list[int]]:
        """Map decoder layer indices to ExLlamaV2 module indices."""
        layer_map: dict[int, list[int]] = {}
        current_layer = -1

        for mod_idx, module in enumerate(self._model.modules):
            name = type(module).__name__
            if "Attention" in name:
                current_layer += 1
                layer_map[current_layer] = [mod_idx]
            elif "MLP" in name and current_layer >= 0:
                layer_map[current_layer].append(mod_idx)

        return layer_map

    def _get_post_layer_modules(self) -> list[int]:
        """Get module indices for norm and lm_head (after all decoder layers)."""
        post_modules = []
        found_last_mlp = False

        for mod_idx in range(len(self._model.modules) - 1, -1, -1):
            name = type(self._model.modules[mod_idx]).__name__
            if "MLP" in name or "Attention" in name:
                if not found_last_mlp:
                    found_last_mlp = True
                    post_modules = list(range(mod_idx + 1, len(self._model.modules)))
                break

        return post_modules

    @torch.no_grad()
    def forward(self, text: str) -> torch.Tensor:
        """Standard forward pass."""
        input_ids = self._tokenizer.encode(text)
        input_ids = input_ids.to(self._model.modules[0].device())
        self._cache.current_seq_len = 0
        logits = self._model.forward(input_ids, self._cache)
        return logits[0, -1, :]

    @torch.no_grad()
    def forward_with_ablation(
        self, text: str, ablated_layers: list[int]
    ) -> torch.Tensor:
        """Forward pass with specified layers skipped."""
        input_ids = self._tokenizer.encode(text)
        input_ids = input_ids.to(self._model.modules[0].device())
        self._cache.current_seq_len = 0

        layer_modules = self._get_layer_module_map()
        ablated_set = set(ablated_layers)

        # Execute embedding
        hidden = self._model.modules[0].forward(input_ids, self._cache)

        # Execute decoder layers, skipping ablated ones
        for layer_idx in range(self._total_layers):
            if layer_idx in ablated_set:
                continue
            for mod_idx in layer_modules[layer_idx]:
                hidden = self._model.modules[mod_idx].forward(hidden, self._cache)

        # Execute final norm and head
        for mod_idx in self._get_post_layer_modules():
            hidden = self._model.modules[mod_idx].forward(hidden, self._cache)

        return hidden[0, -1, :]

    @torch.no_grad()
    def forward_with_hidden_states(
        self, text: str
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass collecting hidden states after each layer."""
        input_ids = self._tokenizer.encode(text)
        input_ids = input_ids.to(self._model.modules[0].device())
        self._cache.current_seq_len = 0

        layer_modules = self._get_layer_module_map()
        all_hidden_states = []

        # Execute embedding
        hidden = self._model.modules[0].forward(input_ids, self._cache)

        # Execute each layer and collect hidden states
        for layer_idx in range(self._total_layers):
            for mod_idx in layer_modules[layer_idx]:
                hidden = self._model.modules[mod_idx].forward(hidden, self._cache)
            all_hidden_states.append(hidden[0].clone())

        # Execute final norm and head
        for mod_idx in self._get_post_layer_modules():
            hidden = self._model.modules[mod_idx].forward(hidden, self._cache)

        return hidden[0, -1, :], all_hidden_states

    @torch.no_grad()
    def forward_with_attention(
        self, text: str
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass returning attention weights.

        Note: ExLlamaV2 does not natively expose attention weights.
        This method raises NotImplementedError.
        """
        raise NotImplementedError(
            "ExLlamaV2 backend does not support attention weight extraction. "
            "Use the Transformers backend for attention analysis."
        )

    def cleanup(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._cache is not None:
            del self._cache
            self._cache = None
        self._tokenizer = None
        self._total_layers = 0

        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class _ExLlamaV2TokenizerAdapter:
    """Adapter to make ExLlamaV2 tokenizer compatible with HuggingFace interface."""

    def __init__(self, exl2_tokenizer) -> None:
        self._tokenizer = exl2_tokenizer

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = self._tokenizer.encode(text)
        return ids[0].tolist() if hasattr(ids, "tolist") else list(ids[0])

    def decode(self, token_ids: list[int]) -> str:
        import torch as _torch

        ids = _torch.tensor([token_ids])
        return self._tokenizer.decode(ids)[0]
