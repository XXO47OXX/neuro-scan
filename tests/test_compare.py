"""Tests for multi-model comparison."""
import json

import numpy as np
import pytest

from neuro_scan.compare import (
    CompareReport,
    ModelSummary,
    extract_model_summary,
    generate_comparison_text,
    load_report,
    run_comparison,
)


@pytest.fixture
def sample_report_a(tmp_path):
    """32-layer model report."""
    labels = {}
    for i in range(32):
        if i < 3:
            labels[str(i)] = "early_processing"
        elif 14 <= i <= 18:
            labels[str(i)] = "reasoning"
        elif i >= 28:
            labels[str(i)] = "output"
        else:
            labels[str(i)] = "semantic_processing"

    data = {
        "model": "model-a-7b",
        "total_layers": 32,
        "baseline_score": 5.0,
        "top_important_layers": [14, 15, 16, 17, 18],
        "layer_labels": labels,
        "ablation_results": [
            {"layer_idx": i, "score_delta": 0.5 if 14 <= i <= 18 else 0.05}
            for i in range(32)
        ],
    }
    path = tmp_path / "report_a.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def sample_report_b(tmp_path):
    """64-layer model report."""
    labels = {}
    for i in range(64):
        if i < 6:
            labels[str(i)] = "early_processing"
        elif 28 <= i <= 36:
            labels[str(i)] = "reasoning"
        elif i >= 58:
            labels[str(i)] = "output"
        else:
            labels[str(i)] = "semantic_processing"

    data = {
        "model": "model-b-13b",
        "total_layers": 64,
        "baseline_score": 6.0,
        "top_important_layers": [28, 30, 32, 34, 36],
        "layer_labels": labels,
        "ablation_results": [
            {"layer_idx": i, "score_delta": 0.8 if 28 <= i <= 36 else 0.03}
            for i in range(64)
        ],
    }
    path = tmp_path / "report_b.json"
    path.write_text(json.dumps(data))
    return path


class TestModelSummary:
    def test_extract_from_report(self, sample_report_a):
        data = json.loads(sample_report_a.read_text())
        summary = extract_model_summary(data)
        assert summary.model_name == "model-a-7b"
        assert summary.total_layers == 32
        assert summary.baseline_score == 5.0
        assert len(summary.reasoning_layers) == 5
        assert 14 in summary.reasoning_layers

    def test_reasoning_fraction(self, sample_report_a):
        data = json.loads(sample_report_a.read_text())
        summary = extract_model_summary(data)
        assert abs(summary.reasoning_fraction - 5 / 32) < 0.01


class TestComparison:
    def test_two_model_comparison(self, sample_report_a, sample_report_b):
        report = run_comparison([sample_report_a, sample_report_b])
        assert len(report.models) == 2
        assert report.similarity_matrix.shape == (2, 2)

    def test_similarity_diagonal_is_one(self, sample_report_a, sample_report_b):
        report = run_comparison([sample_report_a, sample_report_b])
        for i in range(len(report.models)):
            assert abs(report.similarity_matrix[i, i] - 1.0) < 1e-6

    def test_rankings_sorted(self, sample_report_a, sample_report_b):
        report = run_comparison([sample_report_a, sample_report_b])
        sensitivities = [s for _, s in report.model_rankings]
        assert sensitivities == sorted(sensitivities, reverse=True)

    def test_shared_reasoning_exists(self, sample_report_a, sample_report_b):
        report = run_comparison([sample_report_a, sample_report_b])
        assert len(report.shared_reasoning_layers) == 1  # one pair


class TestComparisonText:
    def test_generates_text(self, sample_report_a, sample_report_b):
        report = run_comparison([sample_report_a, sample_report_b])
        text = generate_comparison_text(report)
        assert "MODEL COMPARISON" in text
        assert "model-a-7b" in text
        assert "model-b-13b" in text

    def test_single_model(self, sample_report_a):
        # Single model comparison should still work
        report = run_comparison([sample_report_a, sample_report_a])
        text = generate_comparison_text(report)
        assert "model-a-7b" in text
