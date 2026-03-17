"""Multi-model comparison — compare neuroanatomy across models.

Loads multiple neuro-scan report.json files and generates:
- Side-by-side layer importance comparison
- Reasoning layer overlap analysis
- Model similarity score based on ablation profile
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelSummary:
    """Summary of a single model's neuroanatomy."""

    model_name: str
    total_layers: int
    baseline_score: float
    top_important_layers: list[int]
    layer_labels: dict[str, str]  # str(layer_idx) -> label
    ablation_deltas: list[float]
    reasoning_layers: list[int]
    reasoning_fraction: float


@dataclass
class CompareReport:
    """Complete multi-model comparison report."""

    models: list[ModelSummary]
    similarity_matrix: np.ndarray  # model-model correlation of delta profiles
    shared_reasoning_layers: dict[str, list[int]]  # pair -> shared reasoning layer indices (normalized)
    model_rankings: list[tuple[str, float]]  # (model_name, mean_ablation_sensitivity) sorted


def load_report(path: str | Path) -> dict:
    """Load a neuro-scan report.json file."""
    with open(path) as f:
        return json.load(f)


def extract_model_summary(report: dict) -> ModelSummary:
    """Extract a ModelSummary from a loaded report dict."""
    model_name = Path(report.get("model", "unknown")).name
    total_layers = report.get("total_layers", 0)
    baseline = report.get("baseline_score", 0.0)
    top_layers = report.get("top_important_layers", [])
    labels = report.get("layer_labels", {})

    # Extract ablation deltas
    ablation_results = report.get("ablation_results", [])
    deltas = [r.get("score_delta", 0.0) for r in ablation_results]

    # Find reasoning layers
    reasoning = [
        int(idx) for idx, label in labels.items()
        if label == "reasoning"
    ]
    reasoning_fraction = len(reasoning) / total_layers if total_layers > 0 else 0.0

    return ModelSummary(
        model_name=model_name,
        total_layers=total_layers,
        baseline_score=baseline,
        top_important_layers=top_layers,
        layer_labels=labels,
        ablation_deltas=deltas,
        reasoning_layers=reasoning,
        reasoning_fraction=reasoning_fraction,
    )


def run_comparison(report_paths: list[str | Path]) -> CompareReport:
    """Compare multiple neuro-scan reports.

    Args:
        report_paths: Paths to neuro-scan report.json files.

    Returns:
        CompareReport with comparison analysis.
    """
    models: list[ModelSummary] = []
    for path in report_paths:
        report = load_report(path)
        models.append(extract_model_summary(report))

    # Similarity matrix: correlation of normalized delta profiles
    # Normalize deltas to [0, 1] range per model for comparison across different scales
    normalized_deltas = []
    for m in models:
        if m.ablation_deltas:
            d = np.array(m.ablation_deltas)
            d_range = d.max() - d.min()
            if d_range > 0:
                normalized_deltas.append((d - d.min()) / d_range)
            else:
                normalized_deltas.append(np.zeros_like(d))
        else:
            normalized_deltas.append(np.array([]))

    # For models with different layer counts, compare by relative position
    n = len(models)
    similarity = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            if len(normalized_deltas[i]) > 0 and len(normalized_deltas[j]) > 0:
                # Interpolate to same length for comparison
                target_len = max(len(normalized_deltas[i]), len(normalized_deltas[j]))
                d_i = np.interp(
                    np.linspace(0, 1, target_len),
                    np.linspace(0, 1, len(normalized_deltas[i])),
                    normalized_deltas[i],
                )
                d_j = np.interp(
                    np.linspace(0, 1, target_len),
                    np.linspace(0, 1, len(normalized_deltas[j])),
                    normalized_deltas[j],
                )
                corr = float(np.corrcoef(d_i, d_j)[0, 1])
                similarity[i, j] = corr
                similarity[j, i] = corr

    # Shared reasoning layers (by relative position)
    shared_reasoning: dict[str, list[int]] = {}
    for i in range(n):
        for j in range(i + 1, n):
            m_i, m_j = models[i], models[j]
            # Normalize to relative positions (0-100%)
            rel_i = (
                {round(layer / m_i.total_layers * 100) for layer in m_i.reasoning_layers}
                if m_i.total_layers > 0
                else set()
            )
            rel_j = (
                {round(layer / m_j.total_layers * 100) for layer in m_j.reasoning_layers}
                if m_j.total_layers > 0
                else set()
            )
            shared = sorted(rel_i & rel_j)
            key = f"{m_i.model_name} vs {m_j.model_name}"
            shared_reasoning[key] = shared

    # Model rankings by mean ablation sensitivity
    rankings = []
    for m in models:
        mean_sensitivity = float(np.mean(np.abs(m.ablation_deltas))) if m.ablation_deltas else 0.0
        rankings.append((m.model_name, mean_sensitivity))
    rankings.sort(key=lambda x: x[1], reverse=True)

    return CompareReport(
        models=models,
        similarity_matrix=similarity,
        shared_reasoning_layers=shared_reasoning,
        model_rankings=rankings,
    )


def generate_comparison_text(report: CompareReport) -> str:
    """Generate a text summary of the comparison."""
    lines = [
        "=" * 60,
        "NEURO-SCAN MODEL COMPARISON",
        "=" * 60,
        f"Models compared: {len(report.models)}",
        "",
    ]

    for m in report.models:
        lines.extend([
            f"[{m.model_name}]",
            f"  Layers: {m.total_layers}",
            f"  Baseline: {m.baseline_score:.4f}",
            f"  Reasoning layers: {len(m.reasoning_layers)} ({m.reasoning_fraction:.1%})",
            f"  Top important: {m.top_important_layers[:5]}",
            "",
        ])

    if len(report.models) > 1:
        lines.extend(["SIMILARITY MATRIX:", "-" * 60])
        header = "           " + "  ".join(
            f"{m.model_name[:10]:>10s}" for m in report.models
        )
        lines.append(header)
        for i, m in enumerate(report.models):
            row = f"  {m.model_name[:8]:>8s}"
            for j in range(len(report.models)):
                row += f"  {report.similarity_matrix[i, j]:10.4f}"
            lines.append(row)
        lines.append("")

    if report.shared_reasoning_layers:
        lines.extend(["SHARED REASONING (by relative position %):", "-" * 60])
        for pair, positions in report.shared_reasoning_layers.items():
            if positions:
                lines.append(f"  {pair}: {positions}%")
            else:
                lines.append(f"  {pair}: no overlap")
        lines.append("")

    lines.extend(["MODEL RANKINGS (by mean ablation sensitivity):", "-" * 60])
    for rank, (name, sensitivity) in enumerate(report.model_rankings, 1):
        lines.append(f"  #{rank}: {name} (mean |delta| = {sensitivity:.4f})")

    lines.extend(["", "=" * 60])
    return "\n".join(lines)
