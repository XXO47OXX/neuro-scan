from __future__ import annotations

import numpy as np
import pytest

from neuro_scan.cross_probe import (
    CrossProbeReport,
    CrossProbeResult,
    compute_probe_correlation,
    run_cross_probe_analysis,
)


# --- CrossProbeResult immutability ---


class TestCrossProbeResult:
    """Tests for the CrossProbeResult frozen dataclass."""

    def test_creation(self):
        result = CrossProbeResult(
            probe_name="math",
            baseline_score=4.5,
            ablation_deltas=[0.1, 0.2, 0.3],
            top_layers=[2, 1, 0],
        )
        assert result.probe_name == "math"
        assert result.baseline_score == 4.5
        assert result.ablation_deltas == [0.1, 0.2, 0.3]
        assert result.top_layers == [2, 1, 0]

    def test_immutability(self):
        result = CrossProbeResult(
            probe_name="math",
            baseline_score=4.5,
            ablation_deltas=[0.1, 0.2, 0.3],
            top_layers=[2, 1, 0],
        )
        with pytest.raises(AttributeError):
            result.probe_name = "eq"
        with pytest.raises(AttributeError):
            result.baseline_score = 0.0

    def test_equality(self):
        r1 = CrossProbeResult(
            probe_name="math",
            baseline_score=4.5,
            ablation_deltas=[0.1, 0.2],
            top_layers=[1, 0],
        )
        r2 = CrossProbeResult(
            probe_name="math",
            baseline_score=4.5,
            ablation_deltas=[0.1, 0.2],
            top_layers=[1, 0],
        )
        assert r1 == r2


# --- CrossProbeReport fields ---


class TestCrossProbeReport:
    """Tests for the CrossProbeReport dataclass."""

    def test_creation(self):
        pr1 = CrossProbeResult(
            probe_name="math",
            baseline_score=4.5,
            ablation_deltas=[0.1, 0.5, 0.3, 0.2],
            top_layers=[1, 2],
        )
        pr2 = CrossProbeResult(
            probe_name="eq",
            baseline_score=3.8,
            ablation_deltas=[0.4, 0.1, 0.6, 0.2],
            top_layers=[2, 0],
        )
        report = CrossProbeReport(
            probe_names=["math", "eq"],
            total_layers=4,
            per_probe=[pr1, pr2],
            correlation_matrix=np.eye(2),
            universal_layers=[2],
            probe_specific_layers={"math": [1], "eq": [0]},
            total_time_seconds=5.0,
        )
        assert report.probe_names == ["math", "eq"]
        assert report.total_layers == 4
        assert len(report.per_probe) == 2
        assert report.universal_layers == [2]
        assert report.probe_specific_layers == {"math": [1], "eq": [0]}
        assert report.total_time_seconds == 5.0

    def test_empty_probes(self):
        report = CrossProbeReport(
            probe_names=[],
            total_layers=8,
            per_probe=[],
            correlation_matrix=np.array([[]]),
            universal_layers=[],
            probe_specific_layers={},
            total_time_seconds=0.0,
        )
        assert report.probe_names == []
        assert report.per_probe == []
        assert report.universal_layers == []


# --- Correlation matrix computation ---


class TestComputeProbeCorrelation:
    """Tests for correlation matrix computation."""

    def test_single_probe(self):
        delta_matrix = np.array([[0.1, 0.2, 0.3]])
        corr = compute_probe_correlation(delta_matrix)
        assert corr.shape == (1, 1)
        assert corr[0, 0] == 1.0

    def test_identical_probes(self):
        deltas = [0.1, 0.5, 0.3, 0.2]
        delta_matrix = np.array([deltas, deltas])
        corr = compute_probe_correlation(delta_matrix)
        assert corr.shape == (2, 2)
        np.testing.assert_allclose(corr[0, 1], 1.0)
        np.testing.assert_allclose(corr[1, 0], 1.0)

    def test_anticorrelated_probes(self):
        delta_matrix = np.array([
            [0.1, 0.5, 0.3, 0.2],
            [-0.1, -0.5, -0.3, -0.2],
        ])
        corr = compute_probe_correlation(delta_matrix)
        assert corr.shape == (2, 2)
        np.testing.assert_allclose(corr[0, 1], -1.0)

    def test_uncorrelated_probes(self):
        # Construct two orthogonal vectors
        delta_matrix = np.array([
            [1.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, -1.0],
        ])
        corr = compute_probe_correlation(delta_matrix)
        assert corr.shape == (2, 2)
        np.testing.assert_allclose(corr[0, 1], 0.0, atol=1e-10)

    def test_three_probes(self):
        delta_matrix = np.array([
            [0.1, 0.5, 0.3],
            [0.2, 0.4, 0.6],
            [0.3, 0.1, 0.2],
        ])
        corr = compute_probe_correlation(delta_matrix)
        assert corr.shape == (3, 3)
        # Diagonal should be 1.0
        for i in range(3):
            np.testing.assert_allclose(corr[i, i], 1.0)
        # Should be symmetric
        np.testing.assert_allclose(corr, corr.T)


# --- Universal layer detection ---


class TestUniversalLayers:
    """Tests for universal layer detection logic."""

    def test_all_probes_share_layers(self):
        pr1 = CrossProbeResult(
            probe_name="math",
            baseline_score=4.0,
            ablation_deltas=[0.0] * 8,
            top_layers=[3, 5, 7],
        )
        pr2 = CrossProbeResult(
            probe_name="eq",
            baseline_score=3.5,
            ablation_deltas=[0.0] * 8,
            top_layers=[3, 5, 7],
        )
        # Compute universal = intersection of top_layers
        top_sets = [set(pr.top_layers) for pr in [pr1, pr2]]
        universal = sorted(set.intersection(*top_sets))
        assert universal == [3, 5, 7]

    def test_no_shared_layers(self):
        pr1 = CrossProbeResult(
            probe_name="math",
            baseline_score=4.0,
            ablation_deltas=[0.0] * 8,
            top_layers=[0, 1, 2],
        )
        pr2 = CrossProbeResult(
            probe_name="eq",
            baseline_score=3.5,
            ablation_deltas=[0.0] * 8,
            top_layers=[5, 6, 7],
        )
        top_sets = [set(pr.top_layers) for pr in [pr1, pr2]]
        universal = sorted(set.intersection(*top_sets))
        assert universal == []

    def test_partial_overlap(self):
        pr1 = CrossProbeResult(
            probe_name="math",
            baseline_score=4.0,
            ablation_deltas=[0.0] * 8,
            top_layers=[1, 3, 5],
        )
        pr2 = CrossProbeResult(
            probe_name="eq",
            baseline_score=3.5,
            ablation_deltas=[0.0] * 8,
            top_layers=[3, 5, 6],
        )
        pr3 = CrossProbeResult(
            probe_name="json",
            baseline_score=3.0,
            ablation_deltas=[0.0] * 8,
            top_layers=[2, 3, 5],
        )
        top_sets = [set(pr.top_layers) for pr in [pr1, pr2, pr3]]
        universal = sorted(set.intersection(*top_sets))
        assert universal == [3, 5]


# --- Probe-specific layer detection ---


class TestProbeSpecificLayers:
    """Tests for probe-specific layer detection logic."""

    def test_unique_layers_detected(self):
        pr1 = CrossProbeResult(
            probe_name="math",
            baseline_score=4.0,
            ablation_deltas=[0.0] * 8,
            top_layers=[0, 1, 3],
        )
        pr2 = CrossProbeResult(
            probe_name="eq",
            baseline_score=3.5,
            ablation_deltas=[0.0] * 8,
            top_layers=[3, 5, 6],
        )
        probe_results = [pr1, pr2]
        probe_specific: dict[str, list[int]] = {}
        for i, pr in enumerate(probe_results):
            other_tops: set[int] = set()
            for j, other_pr in enumerate(probe_results):
                if i != j:
                    other_tops.update(other_pr.top_layers)
            specific = sorted(set(pr.top_layers) - other_tops)
            if specific:
                probe_specific[pr.probe_name] = specific

        assert probe_specific == {"math": [0, 1], "eq": [5, 6]}

    def test_no_specific_when_all_shared(self):
        pr1 = CrossProbeResult(
            probe_name="math",
            baseline_score=4.0,
            ablation_deltas=[0.0] * 8,
            top_layers=[3, 5],
        )
        pr2 = CrossProbeResult(
            probe_name="eq",
            baseline_score=3.5,
            ablation_deltas=[0.0] * 8,
            top_layers=[3, 5],
        )
        probe_results = [pr1, pr2]
        probe_specific: dict[str, list[int]] = {}
        for i, pr in enumerate(probe_results):
            other_tops: set[int] = set()
            for j, other_pr in enumerate(probe_results):
                if i != j:
                    other_tops.update(other_pr.top_layers)
            specific = sorted(set(pr.top_layers) - other_tops)
            if specific:
                probe_specific[pr.probe_name] = specific

        assert probe_specific == {}

    def test_three_probes_specificity(self):
        pr1 = CrossProbeResult(
            probe_name="math",
            baseline_score=4.0,
            ablation_deltas=[0.0] * 10,
            top_layers=[0, 1, 5],
        )
        pr2 = CrossProbeResult(
            probe_name="eq",
            baseline_score=3.5,
            ablation_deltas=[0.0] * 10,
            top_layers=[1, 5, 7],
        )
        pr3 = CrossProbeResult(
            probe_name="json",
            baseline_score=3.0,
            ablation_deltas=[0.0] * 10,
            top_layers=[5, 8, 9],
        )
        probe_results = [pr1, pr2, pr3]
        probe_specific: dict[str, list[int]] = {}
        for i, pr in enumerate(probe_results):
            other_tops: set[int] = set()
            for j, other_pr in enumerate(probe_results):
                if i != j:
                    other_tops.update(other_pr.top_layers)
            specific = sorted(set(pr.top_layers) - other_tops)
            if specific:
                probe_specific[pr.probe_name] = specific

        # math: 0 unique (1 shared with eq, 5 shared with both)
        # eq: 7 unique (1 shared with math, 5 shared with both)
        # json: 8, 9 unique (5 shared with both)
        assert probe_specific == {"math": [0], "eq": [7], "json": [8, 9]}


# --- Integration test with MockBackend ---


class TestRunCrossProbeAnalysis:
    """Integration tests for run_cross_probe_analysis."""

    def test_single_probe(self, mock_backend_small):
        from neuro_scan.config import NeuroScanConfig
        from neuro_scan.probes.math_probe import MathProbe

        config = NeuroScanConfig(
            model_path="mock-small",
            probe_name="math",
            batch_size=4,
            top_k_layers=3,
        )
        probe = MathProbe()
        probe.validate(mock_backend_small.get_tokenizer())

        report = run_cross_probe_analysis(
            mock_backend_small, [probe], config, top_k=3
        )

        assert len(report.probe_names) == 1
        assert report.probe_names[0] == "math"
        assert report.total_layers == 8
        assert len(report.per_probe) == 1
        assert len(report.per_probe[0].ablation_deltas) == 8
        assert len(report.per_probe[0].top_layers) == 3
        assert report.correlation_matrix.shape == (1, 1)
        assert report.correlation_matrix[0, 0] == 1.0
        assert report.total_time_seconds >= 0

    def test_two_probes(self, mock_backend_small):
        from neuro_scan.config import NeuroScanConfig
        from neuro_scan.probes.eq_probe import EqProbe
        from neuro_scan.probes.math_probe import MathProbe

        config = NeuroScanConfig(
            model_path="mock-small",
            probe_name="math,eq",
            batch_size=4,
            top_k_layers=3,
        )
        probes = [MathProbe(), EqProbe()]
        for p in probes:
            p.validate(mock_backend_small.get_tokenizer())

        report = run_cross_probe_analysis(
            mock_backend_small, probes, config, top_k=3
        )

        assert len(report.probe_names) == 2
        assert report.probe_names == ["math", "eq"]
        assert report.total_layers == 8
        assert len(report.per_probe) == 2
        assert report.correlation_matrix.shape == (2, 2)
        # Diagonal should be 1.0
        np.testing.assert_allclose(report.correlation_matrix[0, 0], 1.0)
        np.testing.assert_allclose(report.correlation_matrix[1, 1], 1.0)
        # Symmetric
        np.testing.assert_allclose(
            report.correlation_matrix, report.correlation_matrix.T
        )

    def test_empty_probes(self, mock_backend_small):
        from neuro_scan.config import NeuroScanConfig

        config = NeuroScanConfig(
            model_path="mock-small",
            probe_name="",
            batch_size=4,
            top_k_layers=3,
        )

        report = run_cross_probe_analysis(
            mock_backend_small, [], config, top_k=3
        )

        assert report.probe_names == []
        assert report.per_probe == []
        assert report.universal_layers == []
        assert report.probe_specific_layers == {}

    def test_report_per_probe_deltas_match_layers(self, mock_backend_small):
        from neuro_scan.config import NeuroScanConfig
        from neuro_scan.probes.math_probe import MathProbe

        config = NeuroScanConfig(
            model_path="mock-small",
            probe_name="math",
            batch_size=4,
            top_k_layers=3,
        )
        report = run_cross_probe_analysis(
            mock_backend_small, [MathProbe()], config, top_k=3
        )

        for pr in report.per_probe:
            assert len(pr.ablation_deltas) == report.total_layers

    def test_top_k_respected(self, mock_backend_small):
        from neuro_scan.config import NeuroScanConfig
        from neuro_scan.probes.math_probe import MathProbe

        config = NeuroScanConfig(
            model_path="mock-small",
            probe_name="math",
            batch_size=4,
            top_k_layers=2,
        )
        report = run_cross_probe_analysis(
            mock_backend_small, [MathProbe()], config, top_k=2
        )

        for pr in report.per_probe:
            assert len(pr.top_layers) <= 2
