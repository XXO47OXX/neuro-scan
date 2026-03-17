"""Tests for visualization output generation."""


from neuro_scan.visualization import (
    generate_ablation_chart,
    generate_attention_heatmap,
    generate_logit_lens_heatmap,
    generate_summary_text,
)


class TestAblationChart:
    """Tests for ablation sensitivity bar chart."""

    def test_generates_html_file(self, sample_report, tmp_output_dir):
        path = generate_ablation_chart(
            sample_report, tmp_output_dir / "ablation.html"
        )
        assert path.exists()
        assert path.suffix == ".html"

    def test_html_contains_plotly(self, sample_report, tmp_output_dir):
        path = generate_ablation_chart(
            sample_report, tmp_output_dir / "ablation.html"
        )
        content = path.read_text()
        assert "plotly" in content.lower()

    def test_custom_title(self, sample_report, tmp_output_dir):
        path = generate_ablation_chart(
            sample_report,
            tmp_output_dir / "ablation.html",
            title="Custom Title Test",
        )
        content = path.read_text()
        assert "Custom Title Test" in content

    def test_creates_parent_dirs(self, sample_report, tmp_path):
        path = generate_ablation_chart(
            sample_report,
            tmp_path / "nested" / "dir" / "chart.html",
        )
        assert path.exists()


class TestLogitLensHeatmap:
    """Tests for logit lens trajectory heatmap."""

    def test_generates_html_file(self, sample_report, tmp_output_dir):
        path = generate_logit_lens_heatmap(
            sample_report, tmp_output_dir / "logit_lens.html"
        )
        assert path.exists()
        assert path.suffix == ".html"

    def test_empty_trajectory(self, sample_report, tmp_output_dir):
        """Should handle empty trajectory gracefully."""
        sample_report.logit_lens_trajectory = []
        path = generate_logit_lens_heatmap(
            sample_report, tmp_output_dir / "logit_lens.html"
        )
        assert path.exists()
        content = path.read_text()
        assert "No logit lens data" in content

    def test_html_contains_plotly(self, sample_report, tmp_output_dir):
        path = generate_logit_lens_heatmap(
            sample_report, tmp_output_dir / "logit_lens.html"
        )
        content = path.read_text()
        assert "plotly" in content.lower()


class TestAttentionHeatmap:
    """Tests for attention entropy heatmap."""

    def test_generates_html_file(self, sample_report, tmp_output_dir):
        path = generate_attention_heatmap(
            sample_report, tmp_output_dir / "attention.html"
        )
        assert path.exists()

    def test_no_attention_data(self, sample_report, tmp_output_dir):
        """Should handle None attention data gracefully."""
        sample_report.attention_entropy = None
        path = generate_attention_heatmap(
            sample_report, tmp_output_dir / "attention.html"
        )
        assert path.exists()
        content = path.read_text()
        assert "No attention data" in content

    def test_empty_attention_data(self, sample_report, tmp_output_dir):
        sample_report.attention_entropy = []
        path = generate_attention_heatmap(
            sample_report, tmp_output_dir / "attention.html"
        )
        assert path.exists()


class TestSummaryText:
    """Tests for text summary generation."""

    def test_contains_model_info(self, sample_report):
        text = generate_summary_text(sample_report)
        assert "mock-model" in text
        assert "math" in text

    def test_contains_baseline(self, sample_report):
        text = generate_summary_text(sample_report)
        assert "4.5" in text  # baseline_score

    def test_contains_layer_count(self, sample_report):
        text = generate_summary_text(sample_report)
        assert "32" in text

    def test_contains_top_layers(self, sample_report):
        text = generate_summary_text(sample_report)
        assert "TOP IMPORTANT LAYERS" in text

    def test_contains_label_distribution(self, sample_report):
        text = generate_summary_text(sample_report)
        assert "LAYER FUNCTION DISTRIBUTION" in text

    def test_neuro_scan_header(self, sample_report):
        text = generate_summary_text(sample_report)
        assert "NEURO-SCAN" in text
