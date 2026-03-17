"""Custom probe loader — load user-defined probes from JSON files."""

from __future__ import annotations

import json
from pathlib import Path

from neuro_scan.probes.base import Probe, ProbeSample
from neuro_scan.scoring import get_digit_token_ids


class CustomProbe(Probe):
    """Probe loaded from a JSON file.

    Expected JSON format:
    {
        "name": "my_probe",
        "description": "What this probe measures",
        "scoring": "digits",
        "samples": [
            {
                "prompt": "Rate from 0-9...",
                "expected_score": 7.0,
                "metadata": {"category": "..."}
            }
        ]
    }
    """

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Probe file not found: {path}")

        with open(path) as f:
            data = json.load(f)

        self._name: str = data["name"]
        self._description: str = data.get("description", f"Custom probe from {path.name}")
        self._scoring: str = data.get("scoring", "digits")
        self._samples: list[dict] = data["samples"]

        if not self._samples:
            raise ValueError(f"Probe file {path} contains no samples")

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def get_samples(self, count: int | None = None) -> list[ProbeSample]:
        samples = [
            ProbeSample(
                prompt=s["prompt"],
                scoring_suffix=s.get("scoring_suffix", ""),
                expected_score=s.get("expected_score", 0.0),
                correct_answer=s.get("correct_answer"),
                metadata=s.get("metadata"),
            )
            for s in self._samples
        ]
        if count is not None:
            samples = samples[:count]
        return samples

    def get_score_token_ids(self, tokenizer) -> tuple[list[int], list[float]]:
        if self._scoring == "digits":
            return get_digit_token_ids(tokenizer)
        raise ValueError(
            f"Custom scoring mode '{self._scoring}' not yet supported. "
            "Use 'digits' for 0-9 digit scoring."
        )
