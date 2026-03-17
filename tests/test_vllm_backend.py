from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch


class TestVLLMBackendImport:
    def test_import_error_without_vllm(self):
        from neuro_scan.backends.vllm_backend import VLLMBackend

        backend = VLLMBackend()
        with patch.dict("sys.modules", {"vllm": None}):
            with pytest.raises(ImportError, match="vllm"):
                backend.load("test-model")

    def test_class_instantiates(self):
        from neuro_scan.backends.vllm_backend import VLLMBackend

        backend = VLLMBackend()
        assert backend._llm is None
        assert backend._total_layers == 0


class TestVLLMBackendInterface:
    def test_get_total_layers_before_load(self):
        from neuro_scan.backends.vllm_backend import VLLMBackend

        backend = VLLMBackend()
        with pytest.raises(RuntimeError, match="not loaded"):
            backend.get_total_layers()

    def test_get_tokenizer_before_load(self):
        from neuro_scan.backends.vllm_backend import VLLMBackend

        backend = VLLMBackend()
        with pytest.raises(RuntimeError, match="not loaded"):
            backend.get_tokenizer()

    def test_cleanup_safe_when_not_loaded(self):
        from neuro_scan.backends.vllm_backend import VLLMBackend

        backend = VLLMBackend()
        backend.cleanup()

    def test_attention_not_implemented(self):
        from neuro_scan.backends.vllm_backend import VLLMBackend

        backend = VLLMBackend()
        backend._model = MagicMock()
        backend._total_layers = 1
        with pytest.raises(NotImplementedError, match="attention"):
            backend.forward_with_attention("test")


class TestVLLMBackendWithMock:
    @pytest.fixture
    def mock_vllm_backend(self):
        from neuro_scan.backends.vllm_backend import VLLMBackend

        backend = VLLMBackend()

        # Mock the model with LLaMA-style architecture
        mock_model = MagicMock()
        mock_base = MagicMock()
        mock_model.model = mock_base

        # Layers
        mock_layers = [MagicMock() for _ in range(8)]
        for layer in mock_layers:
            layer.return_value = (torch.randn(1, 5, 64),)
        mock_base.layers = mock_layers

        # Embeddings
        mock_base.embed_tokens.return_value = torch.randn(1, 5, 64)
        mock_base.norm.return_value = torch.randn(1, 5, 64)
        mock_model.lm_head.return_value = torch.randn(1, 5, 100)

        # No rotary embeddings
        if hasattr(mock_base, "rotary_emb"):
            del mock_base.rotary_emb

        # Mock parameters for device detection
        mock_model.parameters.return_value = iter([torch.zeros(1)])

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.randint(0, 100, (1, 5))}

        def mock_encode(text, add_special_tokens=True):
            if len(text) == 1 and text.isdigit():
                return [48 + int(text)]
            return [100]

        mock_tokenizer.encode = mock_encode

        backend._model = mock_model
        backend._tokenizer = mock_tokenizer
        backend._layers = mock_layers
        backend._total_layers = 8

        return backend

    def test_get_total_layers(self, mock_vllm_backend):
        assert mock_vllm_backend.get_total_layers() == 8

    def test_get_tokenizer(self, mock_vllm_backend):
        assert mock_vllm_backend.get_tokenizer() is not None

    def test_forward(self, mock_vllm_backend):
        logits = mock_vllm_backend.forward("test text")
        assert logits.shape == (100,)

    def test_forward_with_ablation(self, mock_vllm_backend):
        logits = mock_vllm_backend.forward_with_ablation("test", [2, 5])
        assert logits.shape == (100,)

    def test_forward_with_hidden_states(self, mock_vllm_backend):
        logits, hidden = mock_vllm_backend.forward_with_hidden_states("test")
        assert logits.shape == (100,)
        assert len(hidden) == 8

    def test_get_norm_and_head(self, mock_vllm_backend):
        norm_fn, head_fn = mock_vllm_backend.get_norm_and_head()
        assert callable(norm_fn)
        assert callable(head_fn)

    def test_cleanup(self, mock_vllm_backend):
        mock_vllm_backend.cleanup()
        assert mock_vllm_backend._model is None
        assert mock_vllm_backend._total_layers == 0


class TestVLLMFindLayers:
    def test_finds_llama_layers(self):
        from neuro_scan.backends.vllm_backend import VLLMBackend

        backend = VLLMBackend()
        mock_model = MagicMock()
        mock_model.model.layers = [MagicMock() for _ in range(4)]
        backend._model = mock_model

        layers = backend._find_layers()
        assert len(layers) == 4

    def test_raises_on_unknown_arch(self):
        from neuro_scan.backends.vllm_backend import VLLMBackend

        backend = VLLMBackend()
        mock_model = MagicMock(spec=[])  # No known attributes
        backend._model = mock_model

        with pytest.raises(RuntimeError, match="Could not find"):
            backend._find_layers()
