"""Real-world scenario tests using mock backends.

Tests end-to-end workflows that simulate real model analysis.
"""

import json
import pytest

from neuro_scan.config import NeuroReport, NeuroScanConfig
from neuro_scan.probes.math_probe import MathProbe
from neuro_scan.probes.eq_probe import EqProbe
from neuro_scan.probes.json_probe import JsonProbe


class TestFullMapScenario:
    """Tests for the full map workflow."""

    def test_80_layer_model_map(self, mock_backend_large):
        """Simulate a Qwen2-style 80-layer model scan."""
        from neuro_scan.scanner import run_map

        config = NeuroScanConfig(
            model_path="Qwen2-7B",
            batch_size=2,
            top_k_layers=10,
        )
        probe = MathProbe()

        report = run_map(mock_backend_large, probe, config)

        assert isinstance(report, NeuroReport)
        assert report.total_layers == 80
        assert len(report.ablation_results) == 80
        assert len(report.top_important_layers) == 10
        assert len(report.layer_labels) == 80

    def test_small_model_map(self, mock_backend_small):
        """8-layer model should work with minimal layers."""
        from neuro_scan.scanner import run_map

        config = NeuroScanConfig(
            model_path="tiny-model",
            batch_size=2,
            top_k_layers=3,
        )
        probe = MathProbe()

        report = run_map(mock_backend_small, probe, config)

        assert report.total_layers == 8
        assert len(report.ablation_results) == 8
        assert len(report.top_important_layers) <= 3

    def test_all_outputs_generated(self, mock_backend, tmp_output_dir):
        """Full map generates all expected output files."""
        from neuro_scan.export import export_csv, export_json
        from neuro_scan.scanner import run_map
        from neuro_scan.visualization import (
            generate_ablation_chart,
            generate_attention_heatmap,
            generate_logit_lens_heatmap,
        )

        config = NeuroScanConfig(
            model_path="test-model",
            batch_size=2,
        )
        probe = MathProbe()

        report = run_map(mock_backend, probe, config)

        generate_ablation_chart(report, tmp_output_dir / "ablation.html")
        generate_logit_lens_heatmap(report, tmp_output_dir / "logit_lens.html")
        generate_attention_heatmap(report, tmp_output_dir / "attention.html")
        export_json(report, tmp_output_dir / "report.json")
        export_csv(report, tmp_output_dir / "ablation.csv")

        assert (tmp_output_dir / "ablation.html").exists()
        assert (tmp_output_dir / "logit_lens.html").exists()
        assert (tmp_output_dir / "attention.html").exists()
        assert (tmp_output_dir / "report.json").exists()
        assert (tmp_output_dir / "ablation.csv").exists()

    def test_json_export_valid(self, mock_backend, tmp_output_dir):
        """Exported JSON is valid and contains expected fields."""
        from neuro_scan.export import export_json
        from neuro_scan.scanner import run_map

        config = NeuroScanConfig(model_path="test", batch_size=2)
        probe = MathProbe()
        report = run_map(mock_backend, probe, config)

        path = export_json(report, tmp_output_dir / "report.json")
        data = json.loads(path.read_text())

        assert "model" in data
        assert "total_layers" in data
        assert "ablation_results" in data
        assert "layer_labels" in data
        assert len(data["ablation_results"]) == 32

    def test_csv_export_valid(self, mock_backend, tmp_output_dir):
        """Exported CSV has correct structure."""
        from neuro_scan.export import export_csv
        from neuro_scan.scanner import run_map

        config = NeuroScanConfig(model_path="test", batch_size=2)
        probe = MathProbe()
        report = run_map(mock_backend, probe, config)

        path = export_csv(report, tmp_output_dir / "ablation.csv")
        lines = path.read_text().strip().split("\n")

        assert lines[0] == "layer_idx,score,score_delta,uncertainty,log_odds,accuracy,label"
        assert len(lines) == 33  # header + 32 layers


class TestMultiProbeComparison:
    """Tests comparing different probes on the same model."""

    def test_different_probes_complete(self, mock_backend):
        """All three probes should complete successfully."""
        from neuro_scan.scanner import run_ablation_scan

        config = NeuroScanConfig(model_path="mock", batch_size=2)

        for probe_cls in [MathProbe, EqProbe, JsonProbe]:
            probe = probe_cls()
            baseline, _, results = run_ablation_scan(mock_backend, probe, config)
            assert len(results) == 32
            assert isinstance(baseline, float)


class TestCustomProbeScenario:
    """Tests for custom probe file loading and scanning."""

    def test_custom_probe_scan(self, mock_backend, sample_probe_json):
        """Custom probe file should work for ablation scan."""
        from neuro_scan.probes.custom import CustomProbe
        from neuro_scan.scanner import run_ablation_scan

        probe = CustomProbe(sample_probe_json)
        config = NeuroScanConfig(model_path="mock", batch_size=2)

        _, _, results = run_ablation_scan(mock_backend, probe, config)
        assert len(results) == 32

    def test_custom_probe_nonexistent_file(self):
        """Loading nonexistent probe file should raise."""
        from neuro_scan.probes.custom import CustomProbe

        with pytest.raises(FileNotFoundError):
            CustomProbe("/nonexistent/probe.json")
