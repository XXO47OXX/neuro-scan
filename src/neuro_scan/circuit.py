from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from neuro_scan.similarity import compute_block_influence, compute_layer_similarity

if TYPE_CHECKING:
    from neuro_scan.backends.base import Backend
    from neuro_scan.config import AblationResult
    from neuro_scan.probes.base import Probe

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InteractionResult:
    """Result of pairwise layer ablation interaction analysis."""

    layer_i: int
    layer_j: int
    individual_delta_i: float
    individual_delta_j: float
    joint_delta: float
    interaction_effect: float
    interaction_type: str  # "synergistic" | "redundant" | "independent"


@dataclass(frozen=True)
class CircuitConfig:
    """Configuration for circuit detection."""

    top_k_pairs: int = 10
    strategy: str = "fast"  # "fast" | "thorough" | "exhaustive"
    interaction_threshold: float = 0.05


@dataclass
class CircuitReport:
    """Complete circuit detection report."""

    top_k: int
    strategy: str
    candidate_pairs: list[tuple[int, int]]
    interactions: list[InteractionResult]
    similarity_matrix: np.ndarray | None
    block_influence: list[float] | None
    synergistic_pairs: list[InteractionResult]
    redundant_pairs: list[InteractionResult]
    total_time_seconds: float
    total_evals: int = 0


def _select_top_k_layers(
    ablation_results: list[AblationResult],
    k: int,
) -> list[int]:
    """Select top-K layers by ablation sensitivity."""
    sorted_results = sorted(
        ablation_results,
        key=lambda r: abs(r.score_delta),
        reverse=True,
    )
    return [r.layer_idx for r in sorted_results[:k]]


def _generate_candidate_pairs(
    top_k_layers: list[int],
    total_layers: int,
    strategy: str = "fast",
    similarity_matrix: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Generate candidate layer pairs for pairwise ablation.

    Args:
        top_k_layers: Top-K most sensitive layers.
        total_layers: Total number of layers.
        strategy: "fast", "thorough", or "exhaustive".
        similarity_matrix: Optional similarity matrix for filtering.

    Returns:
        List of (i, j) pairs to test.
    """
    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int]] = []

    def add_pair(a: int, b: int) -> None:
        pair = (min(a, b), max(a, b))
        if pair not in seen and pair[0] != pair[1]:
            seen.add(pair)
            pairs.append(pair)

    if strategy == "exhaustive":
        for i in range(total_layers):
            for j in range(i + 1, total_layers):
                add_pair(i, j)
        return pairs

    # Top-K pairs: K(K-1)/2
    for idx_a, layer_a in enumerate(top_k_layers):
        for layer_b in top_k_layers[idx_a + 1 :]:
            add_pair(layer_a, layer_b)

    # Adjacent pairs for top-K layers
    for layer in top_k_layers:
        if layer > 0:
            add_pair(layer, layer - 1)
        if layer < total_layers - 1:
            add_pair(layer, layer + 1)

    if strategy == "thorough" and similarity_matrix is not None:
        # Add pairs with high similarity (potentially redundant)
        for i in range(total_layers):
            for j in range(i + 1, total_layers):
                if similarity_matrix[i, j] > 0.95:
                    add_pair(i, j)

    return pairs


def run_circuit_detection(
    backend: Backend,
    probe: Probe,
    ablation_results: list[AblationResult],
    baseline_score: float,
    circuit_config: CircuitConfig | None = None,
) -> CircuitReport:
    """Run the three-phase circuit detection pipeline.

    Args:
        backend: Loaded backend.
        probe: Evaluation probe.
        ablation_results: Pre-computed single-layer ablation results.
        baseline_score: Baseline score (no ablation).
        circuit_config: Configuration. Defaults to fast strategy.

    Returns:
        CircuitReport with interaction analysis.
    """
    from neuro_scan.scoring import aggregate_scores_full, score_from_logits

    if circuit_config is None:
        circuit_config = CircuitConfig()

    start_time = time.time()
    total_layers = backend.get_total_layers()
    tokenizer = backend.get_tokenizer()
    samples = probe.get_samples(count=min(16, len(probe.get_samples())))
    token_ids, score_values = probe.get_score_token_ids(tokenizer)

    # Build delta lookup from single-layer ablation
    delta_lookup = {r.layer_idx: r.score_delta for r in ablation_results}

    # Phase 1: Select top-K sensitive layers
    top_k_layers = _select_top_k_layers(ablation_results, circuit_config.top_k_pairs)
    logger.info("Top-%d layers: %s", circuit_config.top_k_pairs, top_k_layers)

    # Phase 2: Compute similarity (one forward pass)
    similarity_matrix = None
    block_influence = None

    if circuit_config.strategy in ("thorough", "exhaustive"):
        logger.info("Computing layer similarity matrix...")
        _, hidden_states = backend.forward_with_hidden_states(samples[0].full_text)
        similarity_matrix = compute_layer_similarity(hidden_states)
        block_influence = compute_block_influence(hidden_states)

    # Generate candidate pairs
    candidate_pairs = _generate_candidate_pairs(
        top_k_layers, total_layers, circuit_config.strategy, similarity_matrix
    )
    logger.info(
        "Testing %d candidate pairs (strategy=%s)",
        len(candidate_pairs),
        circuit_config.strategy,
    )

    # Phase 3: Pairwise ablation
    interactions: list[InteractionResult] = []
    total_evals = 0

    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeRemainingColumn,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(
            f"Circuit detection ({len(candidate_pairs)} pairs)",
            total=len(candidate_pairs),
        )

        for layer_i, layer_j in candidate_pairs:
            # Pairwise ablation
            pair_scores = []
            for sample in samples:
                logits = backend.forward_with_ablation(
                    text=sample.full_text,
                    ablated_layers=[layer_i, layer_j],
                )
                result = score_from_logits(
                    logits=logits,
                    score_token_ids=token_ids,
                    score_values=score_values,
                    correct_answer=sample.correct_answer,
                    tokenizer=tokenizer,
                )
                pair_scores.append(result)
            total_evals += len(samples)

            agg = aggregate_scores_full(pair_scores)
            joint_delta = baseline_score - agg.mean_score

            individual_i = delta_lookup.get(layer_i, 0.0)
            individual_j = delta_lookup.get(layer_j, 0.0)
            interaction = joint_delta - (individual_i + individual_j)

            threshold = circuit_config.interaction_threshold
            if interaction > threshold:
                itype = "synergistic"
            elif interaction < -threshold:
                itype = "redundant"
            else:
                itype = "independent"

            interactions.append(
                InteractionResult(
                    layer_i=layer_i,
                    layer_j=layer_j,
                    individual_delta_i=individual_i,
                    individual_delta_j=individual_j,
                    joint_delta=joint_delta,
                    interaction_effect=interaction,
                    interaction_type=itype,
                )
            )

            progress.update(task, advance=1)

    # Sort results
    synergistic = sorted(
        [r for r in interactions if r.interaction_type == "synergistic"],
        key=lambda r: r.interaction_effect,
        reverse=True,
    )
    redundant = sorted(
        [r for r in interactions if r.interaction_type == "redundant"],
        key=lambda r: r.interaction_effect,
    )

    total_time = time.time() - start_time

    return CircuitReport(
        top_k=circuit_config.top_k_pairs,
        strategy=circuit_config.strategy,
        candidate_pairs=candidate_pairs,
        interactions=interactions,
        similarity_matrix=similarity_matrix,
        block_influence=block_influence,
        synergistic_pairs=synergistic,
        redundant_pairs=redundant,
        total_time_seconds=total_time,
        total_evals=total_evals,
    )
