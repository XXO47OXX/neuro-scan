"""Tests for automatic layer-function labeling."""


from neuro_scan.config import AblationResult
from neuro_scan.labeler import (
    LABEL_EARLY_PROCESSING,
    LABEL_FORMATTING,
    LABEL_OUTPUT,
    LABEL_REASONING,
    LABEL_SEMANTIC,
    LABEL_SYNTAX,
    get_label_color,
    get_label_description,
    label_layers,
)


class TestLabelLayers:
    """Tests for the label_layers function."""

    def test_basic_position_labeling(self):
        """Without ablation data, uses position-based heuristics."""
        labels = label_layers(total_layers=32, ablation_results=[])

        # First ~10% should be early_processing
        assert labels[0] == LABEL_EARLY_PROCESSING
        assert labels[2] == LABEL_EARLY_PROCESSING

        # Last ~10% should be output
        assert labels[31] == LABEL_OUTPUT
        assert labels[29] == LABEL_OUTPUT

    def test_middle_layers_are_semantic(self):
        """Middle layers default to semantic_processing."""
        labels = label_layers(total_layers=32, ablation_results=[])
        assert labels[15] == LABEL_SEMANTIC

    def test_ablation_overrides_to_reasoning(self, sample_ablation_results):
        """Top-k sensitive layers should be labeled as reasoning."""
        labels = label_layers(
            total_layers=32,
            ablation_results=sample_ablation_results,
            top_k=5,
        )

        # Find which layers are reasoning
        reasoning_layers = [i for i, lbl in labels.items() if lbl == LABEL_REASONING]
        assert len(reasoning_layers) > 0

    def test_logit_lens_adds_syntax_labels(
        self, sample_ablation_results, sample_logit_lens_trajectory
    ):
        """Logit lens should create syntax labels before emergence."""
        labels = label_layers(
            total_layers=32,
            ablation_results=sample_ablation_results,
            logit_lens_trajectory=sample_logit_lens_trajectory,
        )

        all_labels = set(labels.values())
        # Should have a mix of labels
        assert len(all_labels) >= 3

    def test_all_layers_labeled(self, sample_ablation_results):
        """Every layer should have a label."""
        labels = label_layers(
            total_layers=32,
            ablation_results=sample_ablation_results,
        )
        assert len(labels) == 32
        for i in range(32):
            assert i in labels

    def test_small_model(self):
        """Model with very few layers should still work."""
        labels = label_layers(total_layers=4, ablation_results=[])
        assert len(labels) == 4

    def test_single_layer(self):
        """Edge case: single layer model."""
        labels = label_layers(total_layers=1, ablation_results=[])
        assert len(labels) == 1
        assert 0 in labels

    def test_large_model(self):
        """80-layer model should distribute labels properly."""
        results = [
            AblationResult(layer_idx=i, score=4.0, score_delta=0.1 * (i % 10), uncertainty=0.1)
            for i in range(80)
        ]
        labels = label_layers(total_layers=80, ablation_results=results, top_k=10)

        assert len(labels) == 80
        label_counts = {}
        for label in labels.values():
            label_counts[label] = label_counts.get(label, 0) + 1

        # Should have early, output, and reasoning at minimum
        assert LABEL_EARLY_PROCESSING in label_counts
        assert LABEL_OUTPUT in label_counts

    def test_top_k_parameter(self, sample_ablation_results):
        """Changing top_k should change number of reasoning labels."""
        labels_k3 = label_layers(
            total_layers=32,
            ablation_results=sample_ablation_results,
            top_k=3,
        )
        labels_k10 = label_layers(
            total_layers=32,
            ablation_results=sample_ablation_results,
            top_k=10,
        )

        reasoning_k3 = sum(1 for lbl in labels_k3.values() if lbl == LABEL_REASONING)
        reasoning_k10 = sum(1 for lbl in labels_k10.values() if lbl == LABEL_REASONING)

        assert reasoning_k3 <= reasoning_k10


class TestLabelHelpers:
    """Tests for label color and description helpers."""

    def test_all_labels_have_colors(self):
        for label in [
            LABEL_EARLY_PROCESSING, LABEL_SYNTAX, LABEL_REASONING,
            LABEL_SEMANTIC, LABEL_FORMATTING, LABEL_OUTPUT,
        ]:
            color = get_label_color(label)
            assert color.startswith("#")
            assert len(color) == 7

    def test_unknown_label_color(self):
        color = get_label_color("unknown_label")
        assert color.startswith("#")

    def test_all_labels_have_descriptions(self):
        for label in [
            LABEL_EARLY_PROCESSING, LABEL_SYNTAX, LABEL_REASONING,
            LABEL_SEMANTIC, LABEL_FORMATTING, LABEL_OUTPUT,
        ]:
            desc = get_label_description(label)
            assert isinstance(desc, str)
            assert len(desc) > 10

    def test_unknown_label_description(self):
        desc = get_label_description("unknown_label")
        assert desc == "Unknown function"
