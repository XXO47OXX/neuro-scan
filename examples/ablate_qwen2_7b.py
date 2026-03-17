"""Example: Run ablation scan on Qwen2-7B.

Usage:
    python examples/ablate_qwen2_7b.py
"""

from neuro_scan.backends.transformers_backend import TransformersBackend
from neuro_scan.config import NeuroScanConfig
from neuro_scan.probes.math_probe import MathProbe
from neuro_scan.scanner import run_ablation_scan
from neuro_scan.visualization import generate_ablation_chart, generate_summary_text

MODEL_PATH = "Qwen/Qwen2-7B"

backend = TransformersBackend()
backend.load(MODEL_PATH, dtype="float16")

config = NeuroScanConfig(
    model_path=MODEL_PATH,
    probe_name="math",
    batch_size=8,
    top_k_layers=10,
)
probe = MathProbe()

baseline, unc, results = run_ablation_scan(backend, probe, config)
print(f"Baseline: {baseline:.4f} (±{unc:.4f})")

# Find top layers
sorted_results = sorted(results, key=lambda r: abs(r.score_delta), reverse=True)
for i, r in enumerate(sorted_results[:10], 1):
    print(f"  #{i}: layer {r.layer_idx} delta={r.score_delta:+.4f}")

backend.cleanup()
