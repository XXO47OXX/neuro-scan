"""Pre-computed neuroanatomy report fetch from HuggingFace Hub."""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

DATASET_ID = "XXO47OXX/neuro-scan-results"


def _normalize_model_id(model_id: str) -> str:
    """Normalize model ID for fuzzy matching."""
    return model_id.strip().lower().replace("_", "-")


def fetch_results(model_id: str, probe: str = "math") -> dict | None:
    """Fetch pre-computed neuroanatomy report from HF Hub dataset.

    Args:
        model_id: HuggingFace model ID (e.g. "Qwen/Qwen2-7B").
        probe: Probe name to look up (default: "math").

    Returns:
        Matching record dict, or None if not found.

    Raises:
        ImportError: If the ``datasets`` package is not installed.
    """
    try:
        import datasets as _datasets_mod
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for fetch.\n"
            "Install it with: pip install neuro-scan[lookup]"
        ) from None

    logger.info("Fetching results from %s ...", DATASET_ID)

    try:
        ds = _datasets_mod.load_dataset(DATASET_ID, split="train")
    except Exception as exc:
        logger.warning("Could not load dataset %s: %s", DATASET_ID, exc)
        return None

    needle = _normalize_model_id(model_id)

    for row in ds:
        row_id = _normalize_model_id(row.get("model_id", ""))
        row_probe = row.get("probe", "").strip().lower()
        if row_id == needle and row_probe == probe.strip().lower():
            return dict(row)

    # Fuzzy: try matching the last segment (e.g. "Qwen2-7B")
    short_needle = needle.rsplit("/", 1)[-1]
    for row in ds:
        row_id = _normalize_model_id(row.get("model_id", ""))
        row_short = row_id.rsplit("/", 1)[-1]
        row_probe = row.get("probe", "").strip().lower()
        if row_short == short_needle and row_probe == probe.strip().lower():
            return dict(row)

    return None


def format_fetch_result(record: dict) -> str:
    """Format a fetch result for console display.

    Args:
        record: A record dict from :func:`fetch_results`.

    Returns:
        Formatted multi-line string.
    """
    lines: list[str] = []
    lines.append(f"Model:    {record.get('model_id', 'unknown')}")
    lines.append(f"Probe:    {record.get('probe', 'unknown')}")
    lines.append(f"Scanned:  {record.get('scan_date', 'unknown')}")
    lines.append(f"Version:  {record.get('neuro_scan_version', 'unknown')}")
    lines.append(f"Layers:   {record.get('total_layers', '?')}")
    lines.append(f"Baseline: {record.get('baseline_score', '?')}")

    top_layers = record.get("top_important_layers")
    if top_layers:
        if isinstance(top_layers, str):
            try:
                top_layers = json.loads(top_layers)
            except (json.JSONDecodeError, TypeError):
                top_layers = []
        lines.append(f"Top layers: {top_layers}")

    layer_labels = record.get("layer_labels")
    if layer_labels:
        if isinstance(layer_labels, str):
            try:
                layer_labels = json.loads(layer_labels)
            except (json.JSONDecodeError, TypeError):
                layer_labels = {}

        if isinstance(layer_labels, dict):
            reasoning = [
                k for k, v in layer_labels.items() if v == "reasoning"
            ]
            if reasoning:
                lines.append(f"Reasoning layers: {reasoning}")

    report_url = record.get("report_url")
    if report_url:
        lines.append(f"Full report: {report_url}")

    return "\n".join(lines)
