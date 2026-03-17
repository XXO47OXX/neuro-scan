import pytest
import torch

from neuro_scan.config import LogitLensStep, NeuroScanConfig
from neuro_scan.probes.math_probe import MathProbe
from neuro_scan.scoring import entropy_from_logits, top_k_tokens


class TestLogitLensStep:
    """Tests for the LogitLensStep dataclass."""

    def test_creation(self):
        step = LogitLensStep(
            layer_idx=5,
            top_token="hello",
            top_token_prob=0.8,
            target_token_prob=0.3,
            entropy=2.5,
        )
        assert step.layer_idx == 5
        assert step.top_token == "hello"
        assert step.top_token_prob == 0.8

    def test_frozen(self):
        step = LogitLensStep(
            layer_idx=0, top_token="x", top_token_prob=0.1,
            target_token_prob=0.0, entropy=10.0,
        )
        with pytest.raises(AttributeError):
            step.layer_idx = 1


class TestEntropyFromLogits:
    """Tests for entropy computation from logits."""

    def test_uniform_distribution(self):
        logits = torch.zeros(100)
        ent = entropy_from_logits(logits)
        # log(100) ≈ 4.605
        assert abs(ent - 4.605) < 0.1

    def test_peaked_distribution(self):
        logits = torch.zeros(100)
        logits[0] = 100.0  # Very peaked
        ent = entropy_from_logits(logits)
        assert ent < 0.1

    def test_two_equal_peaks(self):
        logits = torch.tensor([0.0, 0.0])
        ent = entropy_from_logits(logits)
        assert abs(ent - 0.693) < 0.01  # ln(2)

    def test_handles_inf(self):
        logits = torch.tensor([float("inf"), 0.0, 0.0])
        ent = entropy_from_logits(logits)
        assert not torch.isnan(torch.tensor(ent))

    def test_returns_float(self):
        logits = torch.randn(1000)
        ent = entropy_from_logits(logits)
        assert isinstance(ent, float)

    def test_entropy_is_nonnegative(self):
        for _ in range(10):
            logits = torch.randn(100)
            ent = entropy_from_logits(logits)
            assert ent >= -1e-6  # Allow tiny floating point errors


class TestTopKTokens:
    """Tests for top_k_tokens function."""

    def test_returns_k_tokens(self):
        from tests.conftest import MockTokenizer

        logits = torch.randn(100)
        tokenizer = MockTokenizer()
        result = top_k_tokens(logits, tokenizer, k=5)
        assert len(result) == 5

    def test_returns_tuples(self):
        from tests.conftest import MockTokenizer

        logits = torch.randn(100)
        tokenizer = MockTokenizer()
        result = top_k_tokens(logits, tokenizer, k=3)
        for token_str, prob in result:
            assert isinstance(token_str, str)
            assert isinstance(prob, float)

    def test_probabilities_sum_approximately(self):
        from tests.conftest import MockTokenizer

        logits = torch.randn(10)
        tokenizer = MockTokenizer()
        result = top_k_tokens(logits, tokenizer, k=10)
        total_prob = sum(p for _, p in result)
        assert abs(total_prob - 1.0) < 1e-5

    def test_sorted_descending(self):
        from tests.conftest import MockTokenizer

        logits = torch.randn(100)
        tokenizer = MockTokenizer()
        result = top_k_tokens(logits, tokenizer, k=10)
        probs = [p for _, p in result]
        assert probs == sorted(probs, reverse=True)

    def test_k_larger_than_vocab(self):
        from tests.conftest import MockTokenizer

        logits = torch.randn(5)
        tokenizer = MockTokenizer()
        result = top_k_tokens(logits, tokenizer, k=10)
        assert len(result) == 5


class TestLogitLensIntegration:
    """Integration tests for logit lens with mock backend."""

    def test_run_logit_lens(self, mock_backend):
        from neuro_scan.scanner import run_logit_lens

        config = NeuroScanConfig(model_path="mock", batch_size=2)
        probe = MathProbe()

        trajectories = run_logit_lens(mock_backend, probe, config)

        assert len(trajectories) == 2  # batch_size=2
        assert len(trajectories[0]) == 32  # 32 layers
        assert isinstance(trajectories[0][0], LogitLensStep)
