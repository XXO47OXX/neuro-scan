from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeSample:
    """A single evaluation sample."""

    prompt: str
    scoring_suffix: str = ""
    expected_score: float = 0.0
    correct_answer: int | None = None
    metadata: dict[str, str] | None = None

    @property
    def full_text(self) -> str:
        """Complete text up to the scoring position."""
        return self.prompt + self.scoring_suffix


class Probe(ABC):
    """Abstract base class for evaluation probes.

    A probe defines:
    1. A set of test samples (prompts + expected answers)
    2. How to interpret logits at the scoring position
    3. Metadata about what capability is being tested
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this probe (e.g., 'math', 'eq', 'json')."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this probe measures."""

    @abstractmethod
    def get_samples(self, count: int | None = None) -> list[ProbeSample]:
        """Return evaluation samples.

        Args:
            count: Max number of samples. None = all available samples.

        Returns:
            List of ProbeSample instances.
        """

    @abstractmethod
    def get_score_token_ids(self, tokenizer) -> tuple[list[int], list[float]]:
        """Return token IDs and their numeric values for scoring.

        Args:
            tokenizer: The model's tokenizer.

        Returns:
            Tuple of (token_ids, score_values).
        """

    def validate(self, tokenizer) -> bool:
        """Validate that this probe is compatible with the given tokenizer.

        Args:
            tokenizer: The model's tokenizer.

        Returns:
            True if compatible, raises ValueError otherwise.
        """
        token_ids, values = self.get_score_token_ids(tokenizer)
        if len(token_ids) < 2:
            raise ValueError(
                f"Probe '{self.name}' requires at least 2 score tokens, "
                f"got {len(token_ids)}"
            )
        if len(token_ids) != len(values):
            raise ValueError(
                f"Probe '{self.name}': token_ids ({len(token_ids)}) and "
                f"values ({len(values)}) must have equal length"
            )
        return True
