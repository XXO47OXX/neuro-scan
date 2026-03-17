"""Emotional intelligence probe based on EQ-Bench methodology.

Tests the model's ability to understand and reason about emotions,
social situations, and human psychology. This capability maps to
different layer circuits than mathematical reasoning.
"""

from __future__ import annotations

from neuro_scan.probes.base import Probe, ProbeSample
from neuro_scan.scoring import get_digit_token_ids

_EQ_SAMPLES = [
    {
        "prompt": (
            "A friend cancels plans last minute for the third time this month. "
            "Rate from 0 (not at all frustrated) to 9 (extremely frustrated) "
            "how a typical person would feel.\n"
            "Answer: "
        ),
        "expected": 7.0,
        "category": "frustration",
    },
    {
        "prompt": (
            "Someone receives unexpected praise from a usually critical boss. "
            "Rate from 0 (suspicious) to 9 (purely happy) how they would feel.\n"
            "Answer: "
        ),
        "expected": 5.0,
        "category": "mixed_emotions",
    },
    {
        "prompt": (
            "A child shows you their drawing, clearly proud. You can see it's not "
            "technically good. Rate from 0 (be honest) to 9 (be encouraging) "
            "the socially appropriate response.\n"
            "Answer: "
        ),
        "expected": 8.0,
        "category": "social_judgment",
    },
    {
        "prompt": (
            "Two coworkers are in a heated argument. One says 'I'm fine' in a "
            "flat tone. Rate from 0 (they are fine) to 9 (they are clearly not fine) "
            "the sarcasm level.\n"
            "Answer: "
        ),
        "expected": 8.0,
        "category": "sarcasm_detection",
    },
    {
        "prompt": (
            "A person loses their job but says 'Maybe this is a blessing in disguise.' "
            "Rate from 0 (pure denial) to 9 (genuine optimism) their emotional state.\n"
            "Answer: "
        ),
        "expected": 4.0,
        "category": "coping",
    },
    {
        "prompt": (
            "At a party, someone stands alone looking at their phone. "
            "Rate from 0 (perfectly comfortable) to 9 (socially anxious) "
            "the most likely emotional state.\n"
            "Answer: "
        ),
        "expected": 6.0,
        "category": "social_reading",
    },
    {
        "prompt": (
            "A student gets a 92/100 on a test they studied weeks for. Their friend "
            "who barely studied got 95. Rate from 0 (purely happy) to 9 (envious) "
            "the student's likely feeling.\n"
            "Answer: "
        ),
        "expected": 6.0,
        "category": "comparison",
    },
    {
        "prompt": (
            "Someone apologizes by saying 'I'm sorry you feel that way.' "
            "Rate from 0 (genuine apology) to 9 (non-apology) the sincerity.\n"
            "Answer: "
        ),
        "expected": 8.0,
        "category": "sincerity",
    },
    {
        "prompt": (
            "A person laughs at something sad. Rate from 0 (inappropriate) to 9 "
            "(common coping mechanism) how normal this response is.\n"
            "Answer: "
        ),
        "expected": 6.0,
        "category": "coping",
    },
    {
        "prompt": (
            "After a breakup, someone immediately starts dating again. "
            "Rate from 0 (healthy moving on) to 9 (avoidance behavior) "
            "the psychological assessment.\n"
            "Answer: "
        ),
        "expected": 7.0,
        "category": "psychology",
    },
    {
        "prompt": (
            "A teenager rolls their eyes when asked to do chores. "
            "Rate from 0 (rebellious) to 9 (normal developmental behavior) "
            "the appropriate interpretation.\n"
            "Answer: "
        ),
        "expected": 7.0,
        "category": "development",
    },
    {
        "prompt": (
            "A manager uses the phrase 'Let's circle back on this.' "
            "Rate from 0 (genuine interest) to 9 (polite dismissal) "
            "the likely intent.\n"
            "Answer: "
        ),
        "expected": 7.0,
        "category": "workplace",
    },
]


class EqProbe(Probe):
    """Emotional intelligence probe using 0-9 scoring."""

    @property
    def name(self) -> str:
        return "eq"

    @property
    def description(self) -> str:
        return (
            "Emotional intelligence probe. Tests understanding of emotions, "
            "social cues, sarcasm, coping mechanisms, and interpersonal dynamics. "
            "Maps to different layer circuits than mathematical reasoning."
        )

    def get_samples(self, count: int | None = None) -> list[ProbeSample]:
        samples = [
            ProbeSample(
                prompt=s["prompt"],
                expected_score=s["expected"],
                correct_answer=int(s["expected"]),
                metadata={"category": s["category"]},
            )
            for s in _EQ_SAMPLES
        ]
        if count is not None:
            samples = samples[:count]
        return samples

    def get_score_token_ids(self, tokenizer) -> tuple[list[int], list[float]]:
        return get_digit_token_ids(tokenizer)
