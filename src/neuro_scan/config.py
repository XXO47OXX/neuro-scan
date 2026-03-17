"""Configuration data classes for neuro-scan neuroanatomy analysis."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AblationResult:
    """Result of ablating a single layer.

    Args:
        layer_idx: Index of the ablated layer.
        score: Score with this layer ablated.
        score_delta: Change vs baseline (baseline - ablated_score).
        uncertainty: Variance of the score estimate.
        log_odds: Mean log-odds of correct answer with layer ablated.
            None if probes lack correct_answer annotations.
        accuracy: Accuracy with layer ablated. None if N/A.
    """

    layer_idx: int
    score: float
    score_delta: float
    uncertainty: float
    log_odds: float | None = None
    accuracy: float | None = None


@dataclass(frozen=True)
class LogitLensStep:
    """One layer's logit lens projection for a single sample.

    Args:
        layer_idx: Layer index.
        top_token: Most likely token decoded from this layer's hidden state.
        top_token_prob: Probability of the top token.
        target_token_prob: Probability of the expected answer token at this layer.
        entropy: Entropy of the vocabulary distribution at this layer.
    """

    layer_idx: int
    top_token: str
    top_token_prob: float
    target_token_prob: float
    entropy: float


@dataclass(frozen=True)
class NeuroScanConfig:
    """Configuration for a neuro-scan run.

    Args:
        model_path: Path to the model directory or HuggingFace model ID.
        probe_name: Name of the evaluation probe to use.
        batch_size: Number of probe samples per evaluation.
        output_dir: Directory for results output.
        backend: Backend name (transformers, exllamav2).
        dtype: Model dtype string.
        device: Device string for model loading.
        top_k_layers: Number of most important layers to highlight.
        logit_lens_top_k: Number of top tokens to show per layer in logit lens.
    """

    model_path: str
    probe_name: str = "math"
    batch_size: int = 16
    output_dir: str = "./results"
    custom_probe_path: str | None = None
    backend: str = "transformers"
    dtype: str = "float16"
    device: str = "auto"
    top_k_layers: int = 10
    logit_lens_top_k: int = 5
    tuned_lens_path: str | None = None
    extra_metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.top_k_layers < 1:
            raise ValueError(f"top_k_layers must be >= 1, got {self.top_k_layers}")
        if self.logit_lens_top_k < 1:
            raise ValueError(f"logit_lens_top_k must be >= 1, got {self.logit_lens_top_k}")


@dataclass(frozen=True)
class PromptRepeatConfig:
    """Configuration for the prompt repetition experiment.

    Args:
        model_path: Path to the model.
        probe_name: Probe to use.
        repeat_counts: List of repetition counts to test (e.g., [1, 2, 3, 4]).
        batch_size: Number of probe samples.
    """

    model_path: str
    probe_name: str = "math"
    repeat_counts: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    batch_size: int = 16
    backend: str = "transformers"
    dtype: str = "float16"
    device: str = "auto"
    custom_probe_path: str | None = None


@dataclass
class NeuroReport:
    """Complete neuroanatomy report for a model.

    Args:
        model_path: Path to the analyzed model.
        probe_name: Probe used for evaluation.
        total_layers: Number of decoder layers.
        baseline_score: Score with no ablation.
        ablation_results: Per-layer ablation sensitivity results.
        logit_lens_trajectory: Per-sample logit lens trajectory.
        attention_entropy: Per-layer, per-head attention entropy (optional).
        layer_labels: Auto-labeled layer functions.
        top_important_layers: Top-k layers by ablation sensitivity.
        total_time_seconds: Wall-clock time for the full analysis.
        block_influence: Per-layer BI score (1 - cos_sim(input, output)).
            None if not computed. Index i = BI for layer i.
        metadata: Additional info.
    """

    model_path: str
    probe_name: str
    total_layers: int
    baseline_score: float
    baseline_uncertainty: float
    ablation_results: list[AblationResult]
    logit_lens_trajectory: list[list[LogitLensStep]]
    attention_entropy: list[list[float]] | None
    layer_labels: dict[int, str]
    top_important_layers: list[int]
    total_time_seconds: float
    block_influence: list[float] | None = None
    metadata: dict[str, object] = field(default_factory=dict)
