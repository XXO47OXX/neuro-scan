from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from neuro_scan.config import (
    AblationResult,
    LogitLensStep,
    NeuroReport,
    NeuroScanConfig,
)
from neuro_scan.labeler import label_layers
from neuro_scan.scoring import (
    ScoreResult,
    aggregate_scores_full,
    entropy_from_logits,
    score_from_logits,
)

if TYPE_CHECKING:
    from neuro_scan.backends.base import Backend
    from neuro_scan.probes.base import Probe
    from neuro_scan.tuned_lens import TunedLens


def run_ablation_scan(
    backend: Backend,
    probe: Probe,
    config: NeuroScanConfig,
) -> tuple[float, float, list[AblationResult]]:
    """Run single-layer ablation scan across all layers.

    For each layer, skip it and measure the score delta vs baseline.
    High delta = the layer is critical for the task.

    Returns:
        Tuple of (baseline_score, baseline_uncertainty, ablation_results).
    """
    total_layers = backend.get_total_layers()
    tokenizer = backend.get_tokenizer()
    samples = probe.get_samples(count=config.batch_size)
    token_ids, score_values = probe.get_score_token_ids(tokenizer)

    # Baseline: no ablation
    baseline_scores = _evaluate_baseline(
        backend=backend,
        samples=samples,
        score_token_ids=token_ids,
        score_values=score_values,
        tokenizer=tokenizer,
    )
    baseline_agg = aggregate_scores_full(baseline_scores)
    baseline_score = baseline_agg.mean_score
    baseline_unc = baseline_agg.mean_uncertainty

    # Per-layer ablation
    results: list[AblationResult] = []

    with _progress_bar() as progress:
        task = progress.add_task(
            f"Ablation scan ({total_layers} layers, baseline={baseline_score:.3f})",
            total=total_layers,
        )

        for layer_idx in range(total_layers):
            layer_scores = _evaluate_with_ablation(
                backend=backend,
                samples=samples,
                ablated_layers=[layer_idx],
                score_token_ids=token_ids,
                score_values=score_values,
                tokenizer=tokenizer,
            )
            agg = aggregate_scores_full(layer_scores)
            delta = baseline_score - agg.mean_score

            results.append(
                AblationResult(
                    layer_idx=layer_idx,
                    score=agg.mean_score,
                    score_delta=delta,
                    uncertainty=agg.mean_uncertainty,
                    log_odds=agg.mean_log_odds,
                    accuracy=agg.accuracy,
                )
            )
            progress.update(task, advance=1)

    return baseline_score, baseline_unc, results


def run_logit_lens(
    backend: Backend,
    probe: Probe,
    config: NeuroScanConfig,
    tuned_lens: TunedLens | None = None,
) -> list[list[LogitLensStep]]:
    """Run logit lens analysis: project each layer's hidden state to vocabulary.

    For each probe sample, extract hidden states at every layer,
    project through norm + lm_head, and record what token the model
    "thinks" at each layer.

    Returns:
        List of per-sample trajectories. Each trajectory is a list of
        LogitLensStep, one per layer.
    """
    samples = probe.get_samples(count=config.batch_size)
    token_ids, score_values = probe.get_score_token_ids(backend.get_tokenizer())
    tokenizer = backend.get_tokenizer()
    norm_fn, head_fn = backend.get_norm_and_head()

    all_trajectories: list[list[LogitLensStep]] = []

    with _progress_bar() as progress:
        task = progress.add_task(
            f"Logit lens ({len(samples)} samples)",
            total=len(samples),
        )

        for sample in samples:
            final_logits, hidden_states = backend.forward_with_hidden_states(
                sample.full_text
            )

            trajectory: list[LogitLensStep] = []

            for layer_idx, hs in enumerate(hidden_states):
                last_token_hs = hs[-1:]  # (1, hidden_dim)

                if tuned_lens is not None:
                    # Tuned lens projection
                    layer_logits = tuned_lens.project(
                        last_token_hs, layer_idx, norm_fn, head_fn
                    )
                    if layer_logits.dim() > 1:
                        layer_logits = layer_logits.squeeze()
                else:
                    # Vanilla logit lens
                    last_token_hs_3d = last_token_hs.unsqueeze(0)  # (1, 1, hidden_dim)
                    normed = norm_fn(last_token_hs_3d)
                    layer_logits = head_fn(normed)[0, 0, :]  # (vocab_size,)

                # Top token at this layer
                probs = torch.softmax(layer_logits.float(), dim=0)
                top_id = torch.argmax(probs).item()
                top_prob = probs[top_id].item()
                top_token = tokenizer.decode([top_id])

                # Target token probability (use first score token as reference)
                target_prob = 0.0
                if token_ids:
                    target_probs = probs[token_ids]
                    target_prob = float(target_probs.max().item())

                # Entropy of the distribution
                ent = entropy_from_logits(layer_logits)

                trajectory.append(
                    LogitLensStep(
                        layer_idx=layer_idx,
                        top_token=top_token,
                        top_token_prob=float(top_prob),
                        target_token_prob=float(target_prob),
                        entropy=float(ent),
                    )
                )

            all_trajectories.append(trajectory)
            progress.update(task, advance=1)

    return all_trajectories


def run_attention_entropy(
    backend: Backend,
    probe: Probe,
    config: NeuroScanConfig,
) -> list[list[float]] | None:
    """Compute per-layer, per-head attention entropy.

    Runs one forward pass per sample (not per layer×head×sample),
    then extracts entropy from the cached attention weights.

    Returns:
        List of [num_layers][num_heads] entropy values, or None if
        the backend doesn't support attention extraction.
    """
    samples = probe.get_samples(count=min(config.batch_size, 4))

    # Probe whether attention extraction is supported
    try:
        _, probe_attn = backend.forward_with_attention(samples[0].full_text)
    except (NotImplementedError, RuntimeError):
        return None

    if not probe_attn:
        return None

    total_layers = len(probe_attn)
    num_heads = probe_attn[0].shape[0]

    # Accumulate entropy: [layers][heads] summed across samples
    entropy_accum: list[list[float]] = [
        [0.0] * num_heads for _ in range(total_layers)
    ]

    # One forward pass per sample — O(S) instead of O(L × H × S)
    with _progress_bar() as progress:
        task = progress.add_task(
            f"Attention entropy ({len(samples)} samples, {total_layers}L×{num_heads}H)",
            total=len(samples),
        )

        for sample_idx, sample in enumerate(samples):
            if sample_idx == 0:
                # Reuse the probe result for the first sample
                attn_weights = probe_attn
            else:
                _, attn_weights = backend.forward_with_attention(sample.full_text)

            for layer_idx in range(total_layers):
                # attn_weights[layer] is (num_heads, seq_len, seq_len)
                layer_attn = attn_weights[layer_idx]
                for head_idx in range(num_heads):
                    # Entropy of attention distribution for last token
                    last_token_attn = layer_attn[head_idx, -1]  # (seq_len,)
                    ent = -torch.sum(
                        last_token_attn * torch.log(last_token_attn.clamp(min=1e-12))
                    ).item()
                    entropy_accum[layer_idx][head_idx] += ent

            progress.update(task, advance=1)

    # Average across samples
    num_samples = len(samples)
    all_entropies: list[list[float]] = [
        [v / num_samples for v in layer_ents]
        for layer_ents in entropy_accum
    ]

    return all_entropies


def run_map(
    backend: Backend,
    probe: Probe,
    config: NeuroScanConfig,
) -> NeuroReport:
    """Run complete neuroanatomy analysis: ablation + logit lens + labeling.

    This is the main entry point for `neuro-scan map`.

    Returns:
        Complete NeuroReport with all analysis results.
    """
    start_time = time.time()
    total_layers = backend.get_total_layers()

    # Step 1: Ablation scan
    baseline_score, baseline_unc, ablation_results = run_ablation_scan(
        backend, probe, config
    )

    # Step 2: Logit lens (only if backend supports it)
    logit_lens_trajectory: list[list[LogitLensStep]] = []
    try:
        logit_lens_trajectory = run_logit_lens(backend, probe, config)
    except NotImplementedError:
        pass

    # Step 3: Attention entropy (optional)
    attention_entropy = run_attention_entropy(backend, probe, config)

    # Step 4: Block Influence (single forward pass)
    block_influence = _compute_block_influence(backend, probe)

    # Step 5: Auto-label layers
    layer_labels = label_layers(
        total_layers=total_layers,
        ablation_results=ablation_results,
        logit_lens_trajectory=logit_lens_trajectory,
        top_k=config.top_k_layers,
    )

    # Step 6: Identify top important layers
    sorted_by_impact = sorted(
        ablation_results,
        key=lambda r: abs(r.score_delta),
        reverse=True,
    )
    top_important = [r.layer_idx for r in sorted_by_impact[: config.top_k_layers]]

    total_time = time.time() - start_time

    return NeuroReport(
        model_path=config.model_path,
        probe_name=config.probe_name,
        total_layers=total_layers,
        baseline_score=baseline_score,
        baseline_uncertainty=baseline_unc,
        ablation_results=ablation_results,
        logit_lens_trajectory=logit_lens_trajectory,
        attention_entropy=attention_entropy,
        layer_labels=layer_labels,
        top_important_layers=top_important,
        total_time_seconds=total_time,
        block_influence=block_influence,
        metadata={
            "samples_used": config.batch_size,
            "probe": config.probe_name,
        },
    )


# --- Internal helpers ---


def _evaluate_baseline(
    backend: Backend,
    samples: list,
    score_token_ids: list[int],
    score_values: list[float],
    tokenizer=None,
) -> list[ScoreResult]:
    """Evaluate baseline (no ablation) on all samples."""
    results = []
    for sample in samples:
        logits = backend.forward(text=sample.full_text)
        result = score_from_logits(
            logits=logits,
            score_token_ids=score_token_ids,
            score_values=score_values,
            correct_answer=sample.correct_answer,
            tokenizer=tokenizer,
        )
        results.append(result)
    return results


def _evaluate_with_ablation(
    backend: Backend,
    samples: list,
    ablated_layers: list[int],
    score_token_ids: list[int],
    score_values: list[float],
    tokenizer=None,
) -> list[ScoreResult]:
    """Evaluate with specified layers ablated on all samples."""
    results = []
    for sample in samples:
        logits = backend.forward_with_ablation(
            text=sample.full_text,
            ablated_layers=ablated_layers,
        )
        result = score_from_logits(
            logits=logits,
            score_token_ids=score_token_ids,
            score_values=score_values,
            correct_answer=sample.correct_answer,
            tokenizer=tokenizer,
        )
        results.append(result)
    return results


def _compute_block_influence(
    backend: Backend,
    probe: Probe,
) -> list[float] | None:
    """Compute Block Influence scores from a single forward pass.

    BI(layer_i) = 1 - cos_sim(h_{i-1}, h_i) where h is the residual
    stream at the last token position.

    Uses the first probe sample for efficiency (one forward pass).

    Returns:
        List of BI scores (length = num_layers - 1), or None if the
        backend doesn't support hidden state extraction.
    """
    from neuro_scan.similarity import compute_block_influence

    samples = probe.get_samples(count=1)
    if not samples:
        return None

    try:
        _, hidden_states = backend.forward_with_hidden_states(samples[0].full_text)
    except NotImplementedError:
        return None

    return compute_block_influence(hidden_states)


def _progress_bar() -> Progress:
    """Create a standard progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    )
