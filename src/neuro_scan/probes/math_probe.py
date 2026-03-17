from __future__ import annotations

from neuro_scan.probes.base import Probe, ProbeSample
from neuro_scan.scoring import get_digit_token_ids

_MATH_SAMPLES = [
    {
        "prompt": (
            "Estimate the result of 347 x 28. "
            "Rate your confidence from 0-9 that the answer is between 9700 and 9750.\n"
            "Answer: "
        ),
        "expected": 7.0,
        "category": "arithmetic",
    },
    {
        "prompt": (
            "Without calculating exactly, is 2^17 closer to 100000 or 150000? "
            "Rate from 0 (definitely 100000) to 9 (definitely 150000).\n"
            "Answer: "
        ),
        "expected": 3.0,
        "category": "powers",
    },
    {
        "prompt": (
            "A triangle has sides of length 5, 12, and 13. "
            "Rate from 0-9 how likely this is a right triangle.\n"
            "Answer: "
        ),
        "expected": 9.0,
        "category": "geometry",
    },
    {
        "prompt": (
            "If log2(x) = 10, rate from 0-9 how close x is to 1000.\n"
            "Answer: "
        ),
        "expected": 7.0,
        "category": "logarithm",
    },
    {
        "prompt": (
            "Rate from 0-9 how confident you are that sqrt(144) + sqrt(169) > 25.\n"
            "Answer: "
        ),
        "expected": 1.0,
        "category": "roots",
    },
    {
        "prompt": (
            "A store offers 30% off, then an additional 20% off the reduced price. "
            "Rate from 0-9 how close the total discount is to 50%.\n"
            "Answer: "
        ),
        "expected": 4.0,
        "category": "percentage",
    },
    {
        "prompt": (
            "How many prime numbers are there between 1 and 20? "
            "Rate from 0 (fewer than 6) to 9 (more than 10).\n"
            "Answer: "
        ),
        "expected": 5.0,
        "category": "primes",
    },
    {
        "prompt": (
            "If you flip a fair coin 10 times, rate from 0-9 how likely "
            "you are to get exactly 5 heads.\n"
            "Answer: "
        ),
        "expected": 3.0,
        "category": "probability",
    },
    {
        "prompt": (
            "The sum of interior angles of a hexagon is ___. "
            "Rate from 0-9 how confident you are that it's 720 degrees.\n"
            "Answer: "
        ),
        "expected": 9.0,
        "category": "geometry",
    },
    {
        "prompt": (
            "Rate from 0-9: The derivative of x^3 + 2x^2 at x=1 equals 7.\n"
            "Answer: "
        ),
        "expected": 9.0,
        "category": "calculus",
    },
    {
        "prompt": (
            "Estimate: 999 x 1001 is closest to which value? "
            "Rate from 0 (around 990000) to 9 (around 1000000).\n"
            "Answer: "
        ),
        "expected": 9.0,
        "category": "arithmetic",
    },
    {
        "prompt": (
            "Rate from 0-9: In the Fibonacci sequence (1,1,2,3,5,8,13,...), "
            "the ratio of consecutive terms approaches 1.618.\n"
            "Answer: "
        ),
        "expected": 9.0,
        "category": "sequences",
    },
    {
        "prompt": (
            "Rate from 0-9 your confidence that the integral of 1/x from 1 to e equals 1.\n"
            "Answer: "
        ),
        "expected": 9.0,
        "category": "calculus",
    },
    {
        "prompt": (
            "A group of 23 people is in a room. "
            "Rate from 0-9 how likely at least two share a birthday.\n"
            "Answer: "
        ),
        "expected": 5.0,
        "category": "probability",
    },
    {
        "prompt": (
            "Rate from 0-9: The series 1 + 1/2 + 1/4 + 1/8 + ... converges to 2.\n"
            "Answer: "
        ),
        "expected": 9.0,
        "category": "series",
    },
    {
        "prompt": (
            "Rate from 0-9 how confident you are that "
            "the number 91 is prime.\n"
            "Answer: "
        ),
        "expected": 1.0,
        "category": "primes",
    },
]


class MathProbe(Probe):
    """Math reasoning probe using 0-9 digit scoring."""

    @property
    def name(self) -> str:
        return "math"

    @property
    def description(self) -> str:
        return (
            "Mathematical estimation and reasoning probe. Tests arithmetic, "
            "geometry, calculus, probability, and number theory. Scored on "
            "a 0-9 scale via logit distribution over digit tokens."
        )

    def get_samples(self, count: int | None = None) -> list[ProbeSample]:
        samples = [
            ProbeSample(
                prompt=s["prompt"],
                expected_score=s["expected"],
                correct_answer=int(s["expected"]),
                metadata={"category": s["category"]},
            )
            for s in _MATH_SAMPLES
        ]
        if count is not None:
            samples = samples[:count]
        return samples

    def get_score_token_ids(self, tokenizer) -> tuple[list[int], list[float]]:
        return get_digit_token_ids(tokenizer)
