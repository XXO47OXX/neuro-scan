"""JSON format compliance probe.

Tests the model's ability to follow structured output format instructions,
which is critical for production LLM pipelines.
"""

from __future__ import annotations

from neuro_scan.probes.base import Probe, ProbeSample
from neuro_scan.scoring import get_digit_token_ids

_JSON_SAMPLES = [
    {
        "prompt": (
            'Given the text: "Apache/2.4.41 (Ubuntu) Server"\n'
            "Extract the following fields as valid JSON:\n"
            '{"server": "...", "version": "...", "os": "..."}\n\n'
            "Rate from 0-9 how confident you are in producing valid JSON output.\n"
            "Answer: "
        ),
        "expected": 8.0,
        "category": "simple_extraction",
    },
    {
        "prompt": (
            "Convert this key-value data to a JSON array:\n"
            "Name: Alice, Age: 30\n"
            "Name: Bob, Age: 25\n"
            "Name: Carol, Age: 35\n\n"
            "Rate from 0-9 your confidence in outputting syntactically valid JSON.\n"
            "Answer: "
        ),
        "expected": 8.0,
        "category": "array_format",
    },
    {
        "prompt": (
            "Given nested data:\n"
            "Company: Acme Corp\n"
            "  Department: Engineering\n"
            "    Team Lead: Alice\n"
            "    Members: Bob, Carol\n"
            "  Department: Marketing\n"
            "    Team Lead: Dave\n\n"
            "Produce nested JSON. "
            "Rate from 0-9 your confidence in valid nested JSON.\n"
            "Answer: "
        ),
        "expected": 6.0,
        "category": "nested_structure",
    },
    {
        "prompt": (
            "Extract structured data from this network banner:\n"
            '<html><head><title>D-Link DCS-930L</title></head></html>\n'
            "Required JSON format:\n"
            '{"brand": "...", "product": "...", "type": "camera"}\n\n'
            "Important: Only extract what is explicitly in the banner.\n"
            "Rate from 0-9 confidence in grounded JSON extraction.\n"
            "Answer: "
        ),
        "expected": 7.0,
        "category": "grounded_extraction",
    },
    {
        "prompt": (
            "Given this ambiguous text:\n"
            '"Welcome to our website - Powered by nginx"\n\n'
            "Extract ALL of these fields (use null if not found):\n"
            '{"brand": "...", "product": "...", "version": "...", '
            '"os": "...", "device_type": "..."}\n\n'
            "Rate from 0-9 how well you can handle null/missing fields.\n"
            "Answer: "
        ),
        "expected": 6.0,
        "category": "null_handling",
    },
    {
        "prompt": (
            "Parse this multi-line server response into JSON:\n\n"
            "HTTP/1.1 200 OK\n"
            "Server: Microsoft-IIS/10.0\n"
            "X-Powered-By: ASP.NET\n"
            "Content-Type: text/html\n\n"
            '{"server": "...", "version": "...", "framework": "...", '
            '"content_type": "..."}\n\n'
            "Rate from 0-9 your confidence in complete, valid JSON.\n"
            "Answer: "
        ),
        "expected": 8.0,
        "category": "header_parsing",
    },
    {
        "prompt": (
            "Text contains special characters that need JSON escaping:\n"
            'Path: C:\\Users\\admin\\Desktop\n'
            'Message: He said "hello" and left\n'
            "Tab:\there\n\n"
            "Produce valid JSON with properly escaped strings.\n"
            "Rate from 0-9 confidence in correct JSON escaping.\n"
            "Answer: "
        ),
        "expected": 5.0,
        "category": "escaping",
    },
    {
        "prompt": (
            "Generate a JSON response that follows this EXACT schema:\n"
            "{\n"
            '  "results": [{"id": int, "name": str, "score": float}],\n'
            '  "total": int,\n'
            '  "page": int\n'
            "}\n\n"
            "Sample data: 3 results, page 1.\n"
            "Rate from 0-9 your confidence in schema-compliant JSON.\n"
            "Answer: "
        ),
        "expected": 7.0,
        "category": "schema_compliance",
    },
    {
        "prompt": (
            "CRITICAL: Output ONLY valid JSON, no markdown, no explanation.\n"
            "Input: The Raspberry Pi 4 Model B runs Debian 11\n"
            'Output format: {"device": "...", "os": "...", "os_version": "..."}\n\n'
            "Rate from 0-9 your ability to output ONLY JSON with no extra text.\n"
            "Answer: "
        ),
        "expected": 7.0,
        "category": "strict_output",
    },
    {
        "prompt": (
            "Parse this compound string into structured JSON:\n"
            '"nas-_-ds415play-_-synology"\n\n'
            "Hint: Fields are separated by '-_-'.\n"
            '{"type": "...", "product": "...", "brand": "..."}\n\n'
            "Rate from 0-9 confidence in correctly parsing compound strings.\n"
            "Answer: "
        ),
        "expected": 6.0,
        "category": "compound_parsing",
    },
]


class JsonProbe(Probe):
    """JSON format compliance probe for production LLM pipelines."""

    @property
    def name(self) -> str:
        return "json"

    @property
    def description(self) -> str:
        return (
            "JSON format compliance probe. Tests extraction, escaping, "
            "null handling, schema compliance, and strict output formatting. "
            "Critical for detecting IFEval-like regressions from layer duplication."
        )

    def get_samples(self, count: int | None = None) -> list[ProbeSample]:
        samples = [
            ProbeSample(
                prompt=s["prompt"],
                expected_score=s["expected"],
                correct_answer=int(s["expected"]),
                metadata={"category": s["category"]},
            )
            for s in _JSON_SAMPLES
        ]
        if count is not None:
            samples = samples[:count]
        return samples

    def get_score_token_ids(self, tokenizer) -> tuple[list[int], list[float]]:
        return get_digit_token_ids(tokenizer)
