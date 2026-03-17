"""Example: Run full neuroanatomy map on Llama-3-8B.

Usage:
    python examples/full_map_llama3.py
"""

from pathlib import Path

from neuro_scan.backends.transformers_backend import TransformersBackend
from neuro_scan.config import NeuroScanConfig
from neuro_scan.export import export_csv, export_json
from neuro_scan.probes.math_probe import MathProbe
from neuro_scan.scanner import run_map
from neuro_scan.visualization import (
    generate_ablation_chart,
    generate_attention_heatmap,
    generate_logit_lens_heatmap,
    generate_summary_text,
)

MODEL_PATH = "meta-llama/Meta-Llama-3-8B"
OUTPUT_DIR = Path("./results/llama3")

backend = TransformersBackend()
backend.load(MODEL_PATH, dtype="bfloat16")

config = NeuroScanConfig(
    model_path=MODEL_PATH,
    probe_name="math",
    batch_size=8,
    top_k_layers=10,
    output_dir=str(OUTPUT_DIR),
)
probe = MathProbe()

report = run_map(backend, probe, config)

# Print summary
print(generate_summary_text(report))

# Generate all visualizations
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
generate_ablation_chart(report, OUTPUT_DIR / "ablation.html")
generate_logit_lens_heatmap(report, OUTPUT_DIR / "logit_lens.html")
generate_attention_heatmap(report, OUTPUT_DIR / "attention.html")
export_json(report, OUTPUT_DIR / "report.json")
export_csv(report, OUTPUT_DIR / "ablation.csv")

print(f"\nAll outputs saved to {OUTPUT_DIR}")

backend.cleanup()
