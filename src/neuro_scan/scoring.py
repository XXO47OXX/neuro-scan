"""Logit distribution scoring — reused from layer-scan with neuro-scan extensions.

Core scoring from logit distributions, plus entropy and top-k token utilities
for logit lens analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F  # noqa: N812

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScoreResult:
    """Result of scoring a single sample from logits.

    Args:
        expected_score: Weighted average score from the probability distribution.
        uncertainty: Variance of the score distribution.
        probabilities: Probability of each score value in the restricted set.
        raw_logits: The raw logit values for the restricted token set.
        log_odds: Log-odds of the correct answer token vs all other score tokens.
            None if correct_answer was not provided.
        is_correct: Whether argmax of restricted logits == correct_answer.
            None if correct_answer was not provided.
        coverage: Sum of probabilities for score tokens in the full vocab
            distribution. High coverage (>0.5) means the model is genuinely
            choosing among score tokens; low coverage (<0.1) means digits
            are noise.
        full_vocab_entropy: Entropy of the full vocabulary distribution (nats).
        full_vocab_log_odds: log P(correct) - log P(full_vocab_argmax) from
            the full vocabulary. None if not computable.
        top_token: The argmax token from the full vocabulary distribution.
        top_token_prob: Probability of the top token in the full vocab.
    """

    expected_score: float
    uncertainty: float
    probabilities: list[float]
    raw_logits: list[float]
    log_odds: float | None = None
    is_correct: bool | None = None
    coverage: float = 0.0
    full_vocab_entropy: float = 0.0
    full_vocab_log_odds: float | None = None
    top_token: str | None = None
    top_token_prob: float = 0.0


def score_from_logits(
    logits: torch.Tensor,
    score_token_ids: list[int],
    score_values: list[float] | None = None,
    correct_answer: int | None = None,
    tokenizer=None,
) -> ScoreResult:
    """Compute expected score from logit distribution over a restricted token set.

    Args:
        logits: Raw logits tensor of shape (vocab_size,) for the scoring position.
        score_token_ids: Token IDs corresponding to score values.
        score_values: Numeric values for each token. Defaults to [0, 1, ..., len-1].
        correct_answer: The correct digit value (e.g. 7 for digit "7").
            If provided, computes log-odds and accuracy metrics.
        tokenizer: Optional tokenizer for decoding the full-vocab top token.
            If provided, enables coverage, entropy, and top_token diagnostics.

    Returns:
        ScoreResult with expected score, uncertainty, and distribution details.
    """
    if score_values is None:
        score_values = list(range(len(score_token_ids)))

    if len(score_token_ids) != len(score_values):
        raise ValueError(
            f"score_token_ids ({len(score_token_ids)}) and "
            f"score_values ({len(score_values)}) must have equal length"
        )

    if len(score_token_ids) == 0:
        raise ValueError("score_token_ids must not be empty")

    # Full-vocab diagnostics (before restricting)
    coverage = 0.0
    full_vocab_entropy = 0.0
    full_vocab_log_odds_val = None
    top_token = None
    top_token_prob_val = 0.0

    logits_float = logits.float()
    # Clamp full logits for stability
    if torch.isinf(logits_float).any():
        max_finite = logits_float[torch.isfinite(logits_float)]
        clamp_val = max_finite.max().item() + 100.0 if len(max_finite) > 0 else 1e6
        logits_float = torch.clamp(logits_float, min=-clamp_val, max=clamp_val)

    full_probs = F.softmax(logits_float, dim=0)

    # Coverage: sum of probabilities for score tokens in full vocab
    score_ids_tensor = torch.tensor(score_token_ids, device=full_probs.device)
    coverage = float(full_probs[score_ids_tensor].sum().item())

    # Full-vocab entropy
    log_probs_full = torch.log(full_probs.clamp(min=1e-12))
    full_vocab_entropy = float(-torch.sum(full_probs * log_probs_full).item())

    # Top token from full vocab
    top_id = int(torch.argmax(full_probs).item())
    top_token_prob_val = float(full_probs[top_id].item())
    if tokenizer is not None:
        top_token = tokenizer.decode([top_id])

    # Full-vocab log-odds: log P(correct_digit) - log P(full_vocab_argmax)
    if correct_answer is not None:
        correct_idx_in_scores = None
        for idx, val in enumerate(score_values):
            if int(val) == correct_answer:
                correct_idx_in_scores = idx
                break
        if correct_idx_in_scores is not None:
            correct_token_id = score_token_ids[correct_idx_in_scores]
            log_full_probs = F.log_softmax(logits_float, dim=0)
            log_p_correct_full = log_full_probs[correct_token_id].item()
            log_p_top_full = log_full_probs[top_id].item()
            full_vocab_log_odds_val = log_p_correct_full - log_p_top_full

    # Restricted scoring
    restricted_logits = logits[score_token_ids].float()

    # Clamp +inf to prevent NaN from softmax
    if torch.isinf(restricted_logits).any():
        max_finite = restricted_logits[torch.isfinite(restricted_logits)]
        clamp_val = max_finite.max().item() + 100.0 if len(max_finite) > 0 else 1e6
        restricted_logits = torch.clamp(restricted_logits, max=clamp_val)

    probs = F.softmax(restricted_logits, dim=0)

    probs_np = probs.detach().cpu().numpy()
    values_np = np.array(score_values, dtype=np.float64)

    expected = float(np.dot(values_np, probs_np))
    variance = float(np.dot((values_np - expected) ** 2, probs_np))

    # Compute log-odds and accuracy if correct_answer is provided
    log_odds_val = None
    is_correct_val = None
    if correct_answer is not None:
        log_odds_val, is_correct_val = _compute_log_odds(
            restricted_logits, score_values, correct_answer
        )

    return ScoreResult(
        expected_score=expected,
        uncertainty=variance,
        probabilities=probs_np.tolist(),
        raw_logits=restricted_logits.detach().cpu().tolist(),
        log_odds=log_odds_val,
        is_correct=is_correct_val,
        coverage=coverage,
        full_vocab_entropy=full_vocab_entropy,
        full_vocab_log_odds=full_vocab_log_odds_val,
        top_token=top_token,
        top_token_prob=top_token_prob_val,
    )


def _compute_log_odds(
    restricted_logits: torch.Tensor,
    score_values: list[float],
    correct_answer: int,
) -> tuple[float | None, bool | None]:
    """Compute log-odds of correct answer and argmax accuracy.

    log_odds = log P(correct) - log(sum P(incorrect))

    Returns:
        Tuple of (log_odds, is_correct). Returns (None, None) if
        correct_answer is not found in score_values.
    """
    correct_idx = None
    for idx, val in enumerate(score_values):
        if int(val) == correct_answer:
            correct_idx = idx
            break

    if correct_idx is None:
        return None, None

    log_softmax = F.log_softmax(restricted_logits, dim=0)
    log_p_correct = log_softmax[correct_idx].item()

    other_logits = torch.cat([
        restricted_logits[:correct_idx],
        restricted_logits[correct_idx + 1:],
    ])
    if len(other_logits) > 0:
        log_p_other_sum = torch.logsumexp(other_logits, dim=0).item()
        log_odds = log_p_correct - log_p_other_sum
    else:
        log_odds = float("inf")

    argmax_idx = int(torch.argmax(restricted_logits).item())
    is_correct = argmax_idx == correct_idx

    return float(log_odds), is_correct


def get_digit_token_ids(tokenizer) -> tuple[list[int], list[float]]:
    """Get token IDs for digits 0-9 from a tokenizer.

    Returns:
        Tuple of (token_ids, score_values) for digits 0-9.

    Raises:
        ValueError: If any digit encodes to multiple tokens.
    """
    token_ids = []
    score_values = []
    multi_token_digits = []

    for digit in range(10):
        token_id = tokenizer.encode(str(digit), add_special_tokens=False)
        if len(token_id) == 1:
            token_ids.append(token_id[0])
            score_values.append(float(digit))
        else:
            multi_token_digits.append((digit, token_id))

    if multi_token_digits:
        details = ", ".join(f"'{d}' -> {ids}" for d, ids in multi_token_digits)
        logger.warning(
            "Tokenizer encodes some digits as multiple tokens: %s. "
            "Digit-based scoring is not compatible with this tokenizer.",
            details,
        )
        raise ValueError(
            f"Digits encode to multiple tokens: {details}. "
            "This tokenizer is not compatible with digit-based scoring. "
            "Consider using a custom probe with --probe custom --custom-probe <path> "
            "that maps to single-token score values."
        )

    return token_ids, score_values


@dataclass(frozen=True)
class AggregateResult:
    """Aggregated scoring result across multiple samples.

    Args:
        mean_score: Mean expected score (legacy metric).
        mean_uncertainty: Mean variance of score distributions.
        mean_log_odds: Mean log-odds of correct answer. None if no samples
            have correct_answer annotations.
        accuracy: Fraction of samples where argmax == correct_answer.
            None if no samples have correct_answer annotations.
    """

    mean_score: float
    mean_uncertainty: float
    mean_log_odds: float | None = None
    accuracy: float | None = None
    mean_coverage: float | None = None


def aggregate_scores(results: list[ScoreResult]) -> tuple[float, float]:
    """Aggregate multiple ScoreResults into a single score and uncertainty.

    Args:
        results: List of ScoreResult from individual probe samples.

    Returns:
        Tuple of (mean_score, mean_uncertainty).
    """
    if not results:
        return 0.0, 0.0

    scores = [r.expected_score for r in results]
    uncertainties = [r.uncertainty for r in results]

    return float(np.mean(scores)), float(np.mean(uncertainties))


def aggregate_scores_full(results: list[ScoreResult]) -> AggregateResult:
    """Aggregate scores with full metrics including log-odds and accuracy.

    Args:
        results: List of ScoreResult from individual probe samples.

    Returns:
        AggregateResult with all metrics.
    """
    if not results:
        return AggregateResult(mean_score=0.0, mean_uncertainty=0.0)

    scores = [r.expected_score for r in results]
    uncertainties = [r.uncertainty for r in results]

    log_odds_values = [r.log_odds for r in results if r.log_odds is not None]
    correct_values = [r.is_correct for r in results if r.is_correct is not None]

    mean_log_odds = float(np.mean(log_odds_values)) if log_odds_values else None
    accuracy = float(np.mean(correct_values)) if correct_values else None

    # Coverage diagnostic
    coverage_values = [r.coverage for r in results]
    mean_coverage = float(np.mean(coverage_values)) if coverage_values else None

    return AggregateResult(
        mean_score=float(np.mean(scores)),
        mean_uncertainty=float(np.mean(uncertainties)),
        mean_log_odds=mean_log_odds,
        accuracy=accuracy,
        mean_coverage=mean_coverage,
    )


# --- neuro-scan extensions ---


def entropy_from_logits(logits: torch.Tensor) -> float:
    """Compute entropy of the full vocabulary distribution.

    Args:
        logits: Raw logits of shape (vocab_size,).

    Returns:
        Entropy in nats. Higher = more uncertain distribution.
    """
    logits = logits.float()
    # Clamp for numerical stability
    if torch.isinf(logits).any():
        max_finite = logits[torch.isfinite(logits)]
        clamp_val = max_finite.max().item() + 100.0 if len(max_finite) > 0 else 1e6
        logits = torch.clamp(logits, min=-clamp_val, max=clamp_val)

    probs = F.softmax(logits, dim=0)
    # Avoid log(0) by clamping
    log_probs = torch.log(probs.clamp(min=1e-12))
    entropy = -torch.sum(probs * log_probs).item()
    return float(entropy)


def top_k_tokens(
    logits: torch.Tensor,
    tokenizer,
    k: int = 5,
) -> list[tuple[str, float]]:
    """Get top-k tokens and their probabilities from logits.

    Args:
        logits: Raw logits of shape (vocab_size,).
        tokenizer: Tokenizer with decode method.
        k: Number of top tokens to return.

    Returns:
        List of (token_str, probability) tuples, sorted descending.
    """
    probs = F.softmax(logits.float(), dim=0)
    top_probs, top_ids = torch.topk(probs, min(k, len(probs)))

    results = []
    for prob, token_id in zip(top_probs.tolist(), top_ids.tolist()):
        token_str = tokenizer.decode([token_id])
        results.append((token_str, prob))

    return results
