from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuro_scan.config import NeuroReport

logger = logging.getLogger(__name__)


def export_json(report: NeuroReport, output_path: str | Path) -> Path:
    """Export the full neuro-scan report as JSON.

    Args:
        report: The neuro-scan report.
        output_path: Path for the output JSON file.

    Returns:
        Path to the generated JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "model": report.model_path,
        "probe": report.probe_name,
        "total_layers": report.total_layers,
        "baseline_score": report.baseline_score,
        "baseline_uncertainty": report.baseline_uncertainty,
        "scan_time_seconds": report.total_time_seconds,
        "ablation_results": [
            {
                "layer_idx": r.layer_idx,
                "score": r.score,
                "score_delta": r.score_delta,
                "uncertainty": r.uncertainty,
                "log_odds": r.log_odds,
                "accuracy": r.accuracy,
            }
            for r in report.ablation_results
        ],
        "top_important_layers": report.top_important_layers,
        "layer_labels": {str(k): v for k, v in report.layer_labels.items()},
        "logit_lens_trajectory": [
            [
                {
                    "layer_idx": step.layer_idx,
                    "top_token": step.top_token,
                    "top_token_prob": step.top_token_prob,
                    "target_token_prob": step.target_token_prob,
                    "entropy": step.entropy,
                }
                for step in trajectory
            ]
            for trajectory in report.logit_lens_trajectory
        ],
        "attention_entropy": report.attention_entropy,
        "block_influence": report.block_influence,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Report JSON saved to %s", output_path)
    return output_path


def export_csv(report: NeuroReport, output_path: str | Path) -> Path:
    """Export ablation results as CSV for spreadsheet analysis.

    Args:
        report: The neuro-scan report.
        output_path: Path for the output CSV file.

    Returns:
        Path to the generated CSV file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["layer_idx,score,score_delta,uncertainty,log_odds,accuracy,label"]

    for r in report.ablation_results:
        label = report.layer_labels.get(r.layer_idx, "unknown")
        log_odds_str = f"{r.log_odds:.6f}" if r.log_odds is not None else ""
        accuracy_str = f"{r.accuracy:.6f}" if r.accuracy is not None else ""
        lines.append(
            f"{r.layer_idx},{r.score:.6f},{r.score_delta:.6f},"
            f"{r.uncertainty:.6f},{log_odds_str},{accuracy_str},{label}"
        )

    output_path.write_text("\n".join(lines) + "\n")
    logger.info("CSV saved to %s", output_path)

    return output_path
