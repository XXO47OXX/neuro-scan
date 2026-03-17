"""Cross-probe layer analysis — compare layer importance across capabilities.

Runs ablation scans with multiple probes to answer:
- Which layers are critical for math but not EQ?
- Which layers are universally important?
- How correlated are different probe sensitivities?
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from neuro_scan.backends.base import Backend
    from neuro_scan.config import NeuroScanConfig
    from neuro_scan.probes.base import Probe

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CrossProbeResult:
    """Ablation results for a single probe."""

    probe_name: str
    baseline_score: float
    ablation_deltas: list[float]  # score_delta per layer
    top_layers: list[int]  # top-k most sensitive layers


@dataclass
class CrossProbeReport:
    """Complete cross-probe analysis report."""

    probe_names: list[str]
    total_layers: int
    per_probe: list[CrossProbeResult]
    correlation_matrix: np.ndarray  # probe-probe correlation of delta vectors
    universal_layers: list[int]  # layers important for ALL probes
    probe_specific_layers: dict[str, list[int]]  # layers important for ONE probe only
    total_time_seconds: float


def compute_probe_correlation(delta_matrix: np.ndarray) -> np.ndarray:
    """Compute correlation matrix between probe delta vectors.

    Args:
        delta_matrix: Array of shape (num_probes, num_layers) where each
            row is the ablation delta vector for one probe.

    Returns:
        Correlation matrix of shape (num_probes, num_probes).
        Returns [[1.0]] for a single probe.
    """
    if delta_matrix.shape[0] <= 1:
        return np.array([[1.0]])
    return np.corrcoef(delta_matrix)


def run_cross_probe_analysis(
    backend: Backend,
    probes: list[Probe],
    config: NeuroScanConfig,
    top_k: int = 5,
) -> CrossProbeReport:
    """Run ablation with each probe and compare layer importance.

    Args:
        backend: Loaded inference backend.
        probes: List of probe instances to compare.
        config: Scan configuration.
        top_k: Number of top layers to track per probe.

    Returns:
        CrossProbeReport with per-probe results and cross-probe analysis.
    """
    from neuro_scan.scanner import run_ablation_scan

    start_time = time.time()
    total_layers = backend.get_total_layers()
    probe_results: list[CrossProbeResult] = []

    for probe in probes:
        logger.info("Running ablation for probe: %s", probe.name)
        baseline, _, ablation_results = run_ablation_scan(backend, probe, config)

        deltas = [r.score_delta for r in ablation_results]
        sorted_by_delta = sorted(
            ablation_results, key=lambda r: abs(r.score_delta), reverse=True
        )
        top_layers = [r.layer_idx for r in sorted_by_delta[:top_k]]

        probe_results.append(
            CrossProbeResult(
                probe_name=probe.name,
                baseline_score=baseline,
                ablation_deltas=deltas,
                top_layers=top_layers,
            )
        )

    # Correlation matrix between probes' delta vectors
    delta_matrix = np.array([pr.ablation_deltas for pr in probe_results])
    correlation = compute_probe_correlation(delta_matrix)

    # Universal layers: in top-k for ALL probes
    if probe_results:
        top_sets = [set(pr.top_layers) for pr in probe_results]
        universal = sorted(set.intersection(*top_sets))
    else:
        universal = []

    # Probe-specific layers: in top-k for exactly ONE probe
    probe_specific: dict[str, list[int]] = {}
    for i, pr in enumerate(probe_results):
        other_tops: set[int] = set()
        for j, other_pr in enumerate(probe_results):
            if i != j:
                other_tops.update(other_pr.top_layers)
        specific = sorted(set(pr.top_layers) - other_tops)
        if specific:
            probe_specific[pr.probe_name] = specific

    total_time = time.time() - start_time

    return CrossProbeReport(
        probe_names=[pr.probe_name for pr in probe_results],
        total_layers=total_layers,
        per_probe=probe_results,
        correlation_matrix=correlation,
        universal_layers=universal,
        probe_specific_layers=probe_specific,
        total_time_seconds=total_time,
    )
