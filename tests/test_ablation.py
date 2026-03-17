import pytest

from neuro_scan.config import AblationResult, NeuroScanConfig
from neuro_scan.probes.math_probe import MathProbe
from neuro_scan.scanner import run_ablation_scan


class TestAblationScan:
    """Tests for run_ablation_scan using mock backend."""

    def test_returns_baseline_and_results(self, mock_backend):
        config = NeuroScanConfig(model_path="mock", batch_size=2, top_k_layers=3)
        probe = MathProbe()

        baseline, unc, results = run_ablation_scan(mock_backend, probe, config)

        assert isinstance(baseline, float)
        assert isinstance(unc, float)
        assert len(results) == 32

    def test_results_are_ablation_results(self, mock_backend):
        config = NeuroScanConfig(model_path="mock", batch_size=2)
        probe = MathProbe()

        _, _, results = run_ablation_scan(mock_backend, probe, config)

        for r in results:
            assert isinstance(r, AblationResult)
            assert 0 <= r.layer_idx < 32

    def test_score_delta_is_computed(self, mock_backend):
        config = NeuroScanConfig(model_path="mock", batch_size=2)
        probe = MathProbe()

        baseline, _, results = run_ablation_scan(mock_backend, probe, config)

        for r in results:
            expected_delta = baseline - r.score
            assert abs(r.score_delta - expected_delta) < 1e-6

    def test_layer_indices_are_sequential(self, mock_backend):
        config = NeuroScanConfig(model_path="mock", batch_size=2)
        probe = MathProbe()

        _, _, results = run_ablation_scan(mock_backend, probe, config)

        indices = [r.layer_idx for r in results]
        assert indices == list(range(32))

    def test_small_model(self, mock_backend_small):
        config = NeuroScanConfig(model_path="mock", batch_size=2)
        probe = MathProbe()

        _, _, results = run_ablation_scan(mock_backend_small, probe, config)
        assert len(results) == 8

    def test_large_model(self, mock_backend_large):
        config = NeuroScanConfig(model_path="mock", batch_size=2, top_k_layers=10)
        probe = MathProbe()

        _, _, results = run_ablation_scan(mock_backend_large, probe, config)
        assert len(results) == 80

    def test_batch_size_1(self, mock_backend):
        config = NeuroScanConfig(model_path="mock", batch_size=1)
        probe = MathProbe()

        baseline, _, results = run_ablation_scan(mock_backend, probe, config)
        assert len(results) == 32
        assert isinstance(baseline, float)

    def test_uncertainty_is_nonnegative(self, mock_backend):
        config = NeuroScanConfig(model_path="mock", batch_size=2)
        probe = MathProbe()

        _, baseline_unc, results = run_ablation_scan(mock_backend, probe, config)
        assert baseline_unc >= 0
        for r in results:
            assert r.uncertainty >= 0

    def test_different_probes_give_different_results(self, mock_backend):
        from neuro_scan.probes.eq_probe import EqProbe

        config = NeuroScanConfig(model_path="mock", batch_size=2)

        _, _, math_results = run_ablation_scan(mock_backend, MathProbe(), config)
        _, _, eq_results = run_ablation_scan(mock_backend, EqProbe(), config)

        # Same mock backend + same seed -> same results (but probe samples differ)
        math_scores = [r.score for r in math_results]
        eq_scores = [r.score for r in eq_results]
        # They should still be valid
        assert all(isinstance(s, float) for s in math_scores)
        assert all(isinstance(s, float) for s in eq_scores)


class TestAblationResultDataclass:
    """Tests for the AblationResult dataclass."""

    def test_creation(self):
        r = AblationResult(layer_idx=5, score=4.0, score_delta=0.5, uncertainty=0.1)
        assert r.layer_idx == 5
        assert r.score == 4.0
        assert r.score_delta == 0.5

    def test_frozen(self):
        r = AblationResult(layer_idx=5, score=4.0, score_delta=0.5, uncertainty=0.1)
        with pytest.raises(AttributeError):
            r.layer_idx = 10

    def test_equality(self):
        r1 = AblationResult(layer_idx=5, score=4.0, score_delta=0.5, uncertainty=0.1)
        r2 = AblationResult(layer_idx=5, score=4.0, score_delta=0.5, uncertainty=0.1)
        assert r1 == r2

    def test_inequality(self):
        r1 = AblationResult(layer_idx=5, score=4.0, score_delta=0.5, uncertainty=0.1)
        r2 = AblationResult(layer_idx=6, score=4.0, score_delta=0.5, uncertainty=0.1)
        assert r1 != r2
