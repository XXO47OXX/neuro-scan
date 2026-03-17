from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AblationResult:
    """Result of ablating a single layer."""

    layer_idx: int
    score: float
    score_delta: float
    uncertainty: float
    log_odds: float | None = None
    accuracy: float | None = None


@dataclass(frozen=True)
class LogitLensStep:
    """One layer's logit lens projection for a single sample."""

    layer_idx: int
    top_token: str
    top_token_prob: float
    target_token_prob: float
    entropy: float


@dataclass(frozen=True)
class NeuroScanConfig:
    """Configuration for a neuro-scan run."""

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
    """Configuration for the prompt repetition experiment."""

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
    """Complete neuroanatomy report for a model."""

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
