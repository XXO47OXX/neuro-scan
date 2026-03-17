"""White-box tests for backend interface and mock backends."""

import pytest
import torch

from tests.conftest import MockBackend, MockTokenizer


class TestMockBackend:
    """Tests for the mock backend used in all other tests."""

    def test_load(self):
        b = MockBackend()
        b.load("test-model")
        assert b.get_total_layers() == 32

    def test_get_total_layers_before_load(self):
        b = MockBackend()
        with pytest.raises(RuntimeError, match="not loaded"):
            b.get_total_layers()

    def test_get_tokenizer_before_load(self):
        b = MockBackend()
        with pytest.raises(RuntimeError, match="not loaded"):
            b.get_tokenizer()

    def test_forward_returns_tensor(self, mock_backend):
        result = mock_backend.forward("hello")
        assert isinstance(result, torch.Tensor)
        assert result.shape == (32000,)

    def test_forward_with_ablation_returns_tensor(self, mock_backend):
        result = mock_backend.forward_with_ablation("hello", [5, 10])
        assert isinstance(result, torch.Tensor)
        assert result.shape == (32000,)

    def test_forward_with_hidden_states(self, mock_backend):
        logits, hidden_states = mock_backend.forward_with_hidden_states("hello")
        assert isinstance(logits, torch.Tensor)
        assert len(hidden_states) == 32
        assert hidden_states[0].shape == (5, 768)

    def test_forward_with_attention(self, mock_backend):
        logits, attention = mock_backend.forward_with_attention("hello")
        assert isinstance(logits, torch.Tensor)
        assert len(attention) == 32
        # Each attention: (num_heads, seq_len, seq_len)
        assert attention[0].shape == (8, 5, 5)

    def test_cleanup(self, mock_backend):
        mock_backend.cleanup()
        with pytest.raises(RuntimeError):
            mock_backend.get_total_layers()

    def test_get_norm_and_head(self, mock_backend):
        norm_fn, head_fn = mock_backend.get_norm_and_head()
        assert callable(norm_fn)
        assert callable(head_fn)

    def test_different_layer_counts(self):
        for n in [1, 8, 32, 80, 128]:
            b = MockBackend(total_layers=n)
            b.load("test")
            assert b.get_total_layers() == n

    def test_ablation_changes_output(self, mock_backend):
        """Ablating different layers should produce different outputs."""
        out_no_ablation = mock_backend.forward("test")
        out_ablated = mock_backend.forward_with_ablation("test", [5])
        # With different seeds, outputs should differ
        assert not torch.allclose(out_no_ablation, out_ablated)


class TestMockTokenizer:
    """Tests for the mock tokenizer."""

    def test_single_digit_encoding(self):
        t = MockTokenizer()
        for digit in range(10):
            ids = t.encode(str(digit), add_special_tokens=False)
            assert len(ids) == 1
            assert ids[0] == 48 + digit

    def test_multi_char_encoding(self):
        t = MockTokenizer()
        ids = t.encode("hello", add_special_tokens=False)
        assert len(ids) == 5

    def test_decode(self):
        t = MockTokenizer()
        text = t.decode([72, 105])
        assert text == "Hi"


class TestTransformersBackendInterface:
    """Tests for the abstract backend interface."""

    def test_backend_is_abstract(self):
        from neuro_scan.backends.base import Backend

        with pytest.raises(TypeError):
            Backend()

    def test_backend_has_required_methods(self):
        from neuro_scan.backends.base import Backend

        required = [
            "load", "get_total_layers", "get_tokenizer",
            "forward", "forward_with_ablation",
            "forward_with_hidden_states", "forward_with_attention",
        ]
        for method in required:
            assert hasattr(Backend, method)


class TestTransformersBackendFindLayers:
    """Test _find_layers with mock models."""

    def test_llama_style_layers(self):
        from neuro_scan.backends.transformers_backend import TransformersBackend

        b = TransformersBackend()

        # Mock a LLaMA-style model
        mock_model = type("MockModel", (), {})()
        mock_base = type("MockBase", (), {})()
        mock_layers = [f"layer_{i}" for i in range(32)]
        mock_base.layers = mock_layers
        mock_model.model = mock_base
        b._model = mock_model

        found = b._find_layers()
        assert len(found) == 32

    def test_gpt2_style_layers(self):
        from neuro_scan.backends.transformers_backend import TransformersBackend

        b = TransformersBackend()

        mock_model = type("MockModel", (), {})()
        mock_transformer = type("MockTransformer", (), {})()
        mock_transformer.h = [f"block_{i}" for i in range(12)]
        mock_model.transformer = mock_transformer
        b._model = mock_model

        found = b._find_layers()
        assert len(found) == 12

    def test_unknown_architecture_raises(self):
        from neuro_scan.backends.transformers_backend import TransformersBackend

        b = TransformersBackend()
        b._model = type("EmptyModel", (), {})()

        with pytest.raises(RuntimeError, match="Could not find decoder layers"):
            b._find_layers()


class TestExLlamaV2BackendImport:
    """Test ExLlamaV2 backend import behavior."""

    def test_exllamav2_not_installed_raises(self):
        from neuro_scan.backends.exllamav2 import ExLlamaV2Backend

        b = ExLlamaV2Backend()
        with pytest.raises(ImportError, match="ExLlamaV2 not installed"):
            b.load("nonexistent-model")

    def test_exllamav2_attention_not_supported(self):
        from neuro_scan.backends.exllamav2 import ExLlamaV2Backend

        b = ExLlamaV2Backend()
        b._total_layers = 32  # Bypass load
        with pytest.raises(NotImplementedError, match="does not support attention"):
            b.forward_with_attention("test")
