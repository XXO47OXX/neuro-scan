"""Automatic layer-function labeling — the core differentiator of neuro-scan.

Labels each transformer layer with a functional role based on:
1. Ablation sensitivity (high delta = critical layer)
2. Logit lens trajectory (when does the correct token first emerge?)
3. Layer position heuristics (early/middle/late)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuro_scan.config import AblationResult, LogitLensStep

# Label constants
LABEL_EARLY_PROCESSING = "early_processing"
LABEL_SYNTAX = "syntax"
LABEL_REASONING = "reasoning"
LABEL_SEMANTIC = "semantic_processing"
LABEL_FORMATTING = "formatting"
LABEL_OUTPUT = "output"


def label_layers(
    total_layers: int,
    ablation_results: list[AblationResult],
    logit_lens_trajectory: list[list[LogitLensStep]] | None = None,
    top_k: int = 5,
) -> dict[int, str]:
    """Automatically label each layer's function based on multi-signal analysis.

    Algorithm:
    1. Position-based baseline: first 10% = early_processing, last 10% = output
    2. Ablation-based override: top-k most sensitive layers = reasoning core
    3. Logit lens refinement: layers where correct token first emerges mark
       the transition from syntax to reasoning

    Args:
        total_layers: Number of decoder layers.
        ablation_results: Per-layer ablation results (score_delta).
        logit_lens_trajectory: Per-sample logit lens trajectories.
        top_k: Number of top-sensitive layers to label as reasoning.

    Returns:
        Dict mapping layer_idx -> label string.
    """
    labels: dict[int, str] = {}

    # Phase 1: Position-based baseline
    early_boundary = max(1, int(total_layers * 0.10))
    late_boundary = total_layers - max(1, int(total_layers * 0.10))

    for i in range(total_layers):
        if i < early_boundary:
            labels[i] = LABEL_EARLY_PROCESSING
        elif i >= late_boundary:
            labels[i] = LABEL_OUTPUT
        else:
            labels[i] = LABEL_SEMANTIC

    # Phase 2: Ablation-based override — top-k sensitive layers = reasoning
    if ablation_results:
        sorted_by_delta = sorted(
            ablation_results,
            key=lambda r: abs(r.score_delta),
            reverse=True,
        )
        reasoning_layers = {r.layer_idx for r in sorted_by_delta[:top_k]}

        for layer_idx in reasoning_layers:
            if layer_idx in labels:
                labels[layer_idx] = LABEL_REASONING

    # Phase 3: Logit lens refinement — find emergence layer
    if logit_lens_trajectory:
        emergence_layer = _find_emergence_layer(logit_lens_trajectory, total_layers)

        if emergence_layer is not None:
            # Layers before emergence in the middle zone = syntax
            for i in range(early_boundary, min(emergence_layer, late_boundary)):
                if labels[i] == LABEL_SEMANTIC:
                    labels[i] = LABEL_SYNTAX

            # Layers around and after emergence (before output zone) = formatting
            formatting_start = max(emergence_layer + 1, late_boundary - max(1, int(total_layers * 0.05)))
            for i in range(formatting_start, late_boundary):
                if labels[i] not in (LABEL_REASONING,):
                    labels[i] = LABEL_FORMATTING

    return labels


def _find_emergence_layer(
    trajectories: list[list[LogitLensStep]],
    total_layers: int,
    threshold: float = 0.1,
) -> int | None:
    """Find the first layer where the correct token consistently emerges.

    Uses two strategies:
    1. Log-odds crossing: First layer where target token becomes the most
       likely token among score candidates (log-odds > 0) for >50% of
       samples. This has clearer information-theoretic meaning.
    2. Fallback: If log-odds crossing is not detected, falls back to the
       legacy threshold method (target_token_prob > threshold).

    Args:
        trajectories: Per-sample logit lens trajectories.
        total_layers: Total number of layers.
        threshold: Probability threshold for legacy emergence detection.

    Returns:
        Layer index of emergence, or None if not found.
    """
    if not trajectories:
        return None

    num_score_tokens = 10  # digits 0-9

    # Strategy 1: Log-odds crossing (target prob > 1/num_tokens = uniform)
    # The target token is "dominant" when its probability exceeds uniform chance.
    # For 10 tokens, uniform = 0.1. We check if target_token_prob > max(other_probs).
    # Simplified: check if target_token_prob > 1/num_score_tokens
    # (i.e., better than chance within the restricted set)
    uniform_prob = 1.0 / num_score_tokens

    for layer_idx in range(total_layers):
        above_count = 0
        total_samples = 0

        for sample_trajectory in trajectories:
            if layer_idx < len(sample_trajectory):
                step = sample_trajectory[layer_idx]
                total_samples += 1
                # Log-odds positive ⟺ target prob > sum of other probs
                # As a practical proxy: target_prob > uniform is a weaker
                # but more reliable signal than target_prob > 0.5
                if step.target_token_prob > uniform_prob:
                    above_count += 1

        if total_samples > 0 and above_count / total_samples > 0.5:
            return layer_idx

    # Strategy 2: Legacy fallback with raw threshold
    for layer_idx in range(total_layers):
        above_count = 0
        total_samples = 0

        for sample_trajectory in trajectories:
            if layer_idx < len(sample_trajectory):
                step = sample_trajectory[layer_idx]
                total_samples += 1
                if step.target_token_prob > threshold:
                    above_count += 1

        if total_samples > 0 and above_count / total_samples > 0.5:
            return layer_idx

    return None


def get_label_color(label: str) -> str:
    """Get a display color for a layer label.

    Returns:
        CSS color string for visualization.
    """
    color_map = {
        LABEL_EARLY_PROCESSING: "#636EFA",  # blue
        LABEL_SYNTAX: "#00CC96",            # green
        LABEL_REASONING: "#EF553B",         # red
        LABEL_SEMANTIC: "#AB63FA",          # purple
        LABEL_FORMATTING: "#FFA15A",        # orange
        LABEL_OUTPUT: "#19D3F3",            # cyan
    }
    return color_map.get(label, "#B6B6B6")


def get_label_description(label: str) -> str:
    """Get a human-readable description for a layer label."""
    desc_map = {
        LABEL_EARLY_PROCESSING: "Input embedding and initial token processing",
        LABEL_SYNTAX: "Syntactic structure and grammatical pattern recognition",
        LABEL_REASONING: "Core reasoning and task-critical computation",
        LABEL_SEMANTIC: "Semantic understanding and knowledge retrieval",
        LABEL_FORMATTING: "Output formatting and response structuring",
        LABEL_OUTPUT: "Final output preparation and token selection",
    }
    return desc_map.get(label, "Unknown function")
