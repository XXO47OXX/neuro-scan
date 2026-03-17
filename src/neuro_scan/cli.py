from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(
    name="neuro-scan",
    help="LLM Neuroanatomy Explorer — map what each transformer layer does",
    add_completion=False,
)
console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_probe(probe_name: str, custom_probe_path: str | None = None):
    """Load the specified probe by name."""
    from neuro_scan.probes.custom import CustomProbe
    from neuro_scan.probes.eq_probe import EqProbe
    from neuro_scan.probes.json_probe import JsonProbe
    from neuro_scan.probes.math_probe import MathProbe

    builtin_probes = {
        "math": MathProbe,
        "eq": EqProbe,
        "json": JsonProbe,
    }

    if probe_name == "custom":
        if not custom_probe_path:
            console.print("[red]--custom-probe required when --probe=custom[/red]")
            raise typer.Exit(1)
        return CustomProbe(custom_probe_path)

    if probe_name in builtin_probes:
        return builtin_probes[probe_name]()

    console.print(f"[red]Unknown probe: {probe_name}[/red]")
    console.print(f"Available probes: {', '.join(builtin_probes.keys())}, custom")
    raise typer.Exit(1)


def _load_backend(backend_name: str):
    """Load the specified inference backend."""
    if backend_name == "transformers":
        from neuro_scan.backends.transformers_backend import TransformersBackend

        return TransformersBackend()
    elif backend_name == "exllamav2":
        from neuro_scan.backends.exllamav2 import ExLlamaV2Backend

        return ExLlamaV2Backend()
    elif backend_name == "vllm":
        from neuro_scan.backends.vllm_backend import VLLMBackend

        return VLLMBackend()
    else:
        console.print(f"[red]Unknown backend: {backend_name}[/red]")
        console.print("Available backends: transformers, exllamav2, vllm")
        raise typer.Exit(1)


# Common CLI options
_model_option = typer.Option(..., "--model", "-m", help="Model path or HuggingFace ID")
_probe_option = typer.Option("math", "--probe", "-p", help="Probe: math, eq, json, custom")
_backend_option = typer.Option("transformers", "--backend", "-b", help="Backend: transformers, exllamav2")
_batch_option = typer.Option(16, "--batch-size", help="Samples per evaluation")
_output_option = typer.Option("./results", "--output", "-o", help="Output directory")
_custom_probe_option = typer.Option(None, "--custom-probe", help="Path to custom probe JSON")
_dtype_option = typer.Option("float16", "--dtype", help="Model dtype: float16, bfloat16, float32")
_verbose_option = typer.Option(False, "--verbose", "-v", help="Verbose logging")
_top_k_option = typer.Option(10, "--top-k", "-k", help="Number of top layers to highlight")


def _init_backend_and_probe(
    model: str,
    probe: str,
    backend: str,
    custom_probe: str | None,
    dtype: str,
    verbose: bool,
):
    """Common setup: logging, backend, probe loading and validation."""
    _setup_logging(verbose)

    probe_instance = _load_probe(probe, custom_probe)
    backend_instance = _load_backend(backend)

    console.print("[bold cyan]Loading model...[/bold cyan]")
    backend_instance.load(model, dtype=dtype)

    total_layers = backend_instance.get_total_layers()
    console.print(f"[green]Model loaded: {total_layers} layers[/green]")

    probe_instance.validate(backend_instance.get_tokenizer())
    console.print(f"[green]Probe '{probe}' validated[/green]")

    return backend_instance, probe_instance


@app.command()
def ablate(
    model: str = _model_option,
    probe: str = _probe_option,
    backend: str = _backend_option,
    batch_size: int = _batch_option,
    output: str = _output_option,
    custom_probe: str = _custom_probe_option,
    dtype: str = _dtype_option,
    verbose: bool = _verbose_option,
    top_k: int = _top_k_option,
) -> None:
    """Run layer ablation analysis — find which layers matter most."""
    from neuro_scan.config import NeuroScanConfig
    from neuro_scan.scanner import run_ablation_scan
    from neuro_scan.visualization import generate_ablation_chart

    backend_instance, probe_instance = _init_backend_and_probe(
        model, probe, backend, custom_probe, dtype, verbose
    )

    config = NeuroScanConfig(
        model_path=model,
        probe_name=probe,
        batch_size=batch_size,
        output_dir=output,
        backend=backend,
        dtype=dtype,
        top_k_layers=top_k,
        custom_probe_path=custom_probe,
    )

    console.print(Panel("[bold]Ablation Scan[/bold]", title="neuro-scan"))

    baseline_score, baseline_unc, results = run_ablation_scan(
        backend_instance, probe_instance, config
    )

    # Build a minimal report for visualization
    from neuro_scan.config import NeuroReport
    from neuro_scan.labeler import label_layers

    labels = label_layers(
        total_layers=backend_instance.get_total_layers(),
        ablation_results=results,
        top_k=top_k,
    )
    sorted_by_impact = sorted(results, key=lambda r: abs(r.score_delta), reverse=True)
    top_layers = [r.layer_idx for r in sorted_by_impact[:top_k]]

    report = NeuroReport(
        model_path=model,
        probe_name=probe,
        total_layers=backend_instance.get_total_layers(),
        baseline_score=baseline_score,
        baseline_uncertainty=baseline_unc,
        ablation_results=results,
        logit_lens_trajectory=[],
        attention_entropy=None,
        layer_labels=labels,
        top_important_layers=top_layers,
        total_time_seconds=0.0,
    )

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    chart_path = generate_ablation_chart(report, output_dir / "ablation.html")
    console.print(f"\n[green]Ablation chart: {chart_path}[/green]")

    from neuro_scan.visualization import generate_summary_text

    console.print(generate_summary_text(report))

    backend_instance.cleanup()


@app.command(name="logit-lens")
def logit_lens(
    model: str = _model_option,
    probe: str = _probe_option,
    backend: str = _backend_option,
    batch_size: int = _batch_option,
    output: str = _output_option,
    custom_probe: str = _custom_probe_option,
    dtype: str = _dtype_option,
    verbose: bool = _verbose_option,
    top_k: int = typer.Option(5, "--top-k", "-k", help="Top-k tokens per layer"),
    tuned_lens_path: str = typer.Option(None, "--tuned-lens", help="Path to tuned lens .safetensors"),
) -> None:
    """Run logit lens analysis — see what each layer predicts."""
    from neuro_scan.config import NeuroScanConfig
    from neuro_scan.scanner import run_logit_lens
    from neuro_scan.visualization import generate_logit_lens_heatmap

    backend_instance, probe_instance = _init_backend_and_probe(
        model, probe, backend, custom_probe, dtype, verbose
    )

    config = NeuroScanConfig(
        model_path=model,
        probe_name=probe,
        batch_size=batch_size,
        output_dir=output,
        backend=backend,
        dtype=dtype,
        logit_lens_top_k=top_k,
        custom_probe_path=custom_probe,
        tuned_lens_path=tuned_lens_path,
    )

    # Load tuned lens if provided
    loaded_tuned_lens = None
    if tuned_lens_path:
        from neuro_scan.tuned_lens import TunedLens
        console.print(f"[cyan]Loading tuned lens from {tuned_lens_path}...[/cyan]")
        loaded_tuned_lens = TunedLens.load(tuned_lens_path)

    console.print(Panel("[bold]Logit Lens Analysis[/bold]", title="neuro-scan"))

    trajectories = run_logit_lens(
        backend_instance, probe_instance, config, tuned_lens=loaded_tuned_lens
    )

    # Build minimal report for visualization
    from neuro_scan.config import NeuroReport

    report = NeuroReport(
        model_path=model,
        probe_name=probe,
        total_layers=backend_instance.get_total_layers(),
        baseline_score=0.0,
        baseline_uncertainty=0.0,
        ablation_results=[],
        logit_lens_trajectory=trajectories,
        attention_entropy=None,
        layer_labels={},
        top_important_layers=[],
        total_time_seconds=0.0,
    )

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    heatmap_path = generate_logit_lens_heatmap(report, output_dir / "logit_lens.html")
    console.print(f"\n[green]Logit lens heatmap: {heatmap_path}[/green]")

    backend_instance.cleanup()


@app.command()
def attention(
    model: str = _model_option,
    probe: str = _probe_option,
    backend: str = _backend_option,
    batch_size: int = _batch_option,
    output: str = _output_option,
    custom_probe: str = _custom_probe_option,
    dtype: str = _dtype_option,
    verbose: bool = _verbose_option,
) -> None:
    """Run attention entropy analysis (experimental)."""
    from neuro_scan.config import NeuroScanConfig
    from neuro_scan.scanner import run_attention_entropy
    from neuro_scan.visualization import generate_attention_heatmap

    backend_instance, probe_instance = _init_backend_and_probe(
        model, probe, backend, custom_probe, dtype, verbose
    )

    config = NeuroScanConfig(
        model_path=model,
        probe_name=probe,
        batch_size=batch_size,
        output_dir=output,
        backend=backend,
        dtype=dtype,
        custom_probe_path=custom_probe,
    )

    console.print(
        Panel("[bold]Attention Entropy Analysis[/bold] (experimental)", title="neuro-scan")
    )

    entropy_data = run_attention_entropy(backend_instance, probe_instance, config)

    if entropy_data is None:
        console.print("[yellow]Backend does not support attention extraction.[/yellow]")
        backend_instance.cleanup()
        return

    from neuro_scan.config import NeuroReport

    report = NeuroReport(
        model_path=model,
        probe_name=probe,
        total_layers=backend_instance.get_total_layers(),
        baseline_score=0.0,
        baseline_uncertainty=0.0,
        ablation_results=[],
        logit_lens_trajectory=[],
        attention_entropy=entropy_data,
        layer_labels={},
        top_important_layers=[],
        total_time_seconds=0.0,
    )

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    heatmap_path = generate_attention_heatmap(report, output_dir / "attention.html")
    console.print(f"\n[green]Attention heatmap: {heatmap_path}[/green]")

    backend_instance.cleanup()


@app.command(name="map")
def map_command(
    model: str = _model_option,
    probe: str = _probe_option,
    backend: str = _backend_option,
    batch_size: int = _batch_option,
    output: str = _output_option,
    custom_probe: str = _custom_probe_option,
    dtype: str = _dtype_option,
    verbose: bool = _verbose_option,
    top_k: int = _top_k_option,
) -> None:
    """Run complete neuroanatomy map (ablation + logit lens + labeling)."""
    from neuro_scan.config import NeuroScanConfig
    from neuro_scan.export import export_csv, export_json
    from neuro_scan.scanner import run_map
    from neuro_scan.visualization import (
        generate_ablation_chart,
        generate_attention_heatmap,
        generate_entropy_profile_chart,
        generate_logit_lens_heatmap,
        generate_summary_text,
    )

    backend_instance, probe_instance = _init_backend_and_probe(
        model, probe, backend, custom_probe, dtype, verbose
    )

    config = NeuroScanConfig(
        model_path=model,
        probe_name=probe,
        batch_size=batch_size,
        output_dir=output,
        backend=backend,
        dtype=dtype,
        top_k_layers=top_k,
        custom_probe_path=custom_probe,
    )

    console.print(
        Panel(
            f"[bold]neuro-scan v0.2.1[/bold]\n"
            f"Model: {model}\n"
            f"Probe: {probe}\n"
            f"Backend: {backend}",
            title="Full Neuroanatomy Map",
        )
    )

    report = run_map(backend_instance, probe_instance, config)

    # Generate all outputs
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(generate_summary_text(report))

    chart_path = generate_ablation_chart(report, output_dir / "ablation.html")
    console.print(f"\n[green]Ablation chart: {chart_path}[/green]")

    lens_path = generate_logit_lens_heatmap(report, output_dir / "logit_lens.html")
    console.print(f"[green]Logit lens: {lens_path}[/green]")

    attn_path = generate_attention_heatmap(report, output_dir / "attention.html")
    console.print(f"[green]Attention: {attn_path}[/green]")

    entropy_path = generate_entropy_profile_chart(report, output_dir / "entropy_profile.html")
    console.print(f"[green]Entropy profile: {entropy_path}[/green]")

    json_path = export_json(report, output_dir / "report.json")
    console.print(f"[green]Report JSON: {json_path}[/green]")

    csv_path = export_csv(report, output_dir / "ablation.csv")
    console.print(f"[green]Ablation CSV: {csv_path}[/green]")

    backend_instance.cleanup()
    console.print("[dim]Done.[/dim]")


@app.command()
def circuit(
    model: str = _model_option,
    probe: str = _probe_option,
    backend: str = _backend_option,
    batch_size: int = _batch_option,
    output: str = _output_option,
    custom_probe: str = _custom_probe_option,
    dtype: str = _dtype_option,
    verbose: bool = _verbose_option,
    top_k_pairs: int = typer.Option(10, "--top-k-pairs", help="Number of top layers to consider"),
    strategy: str = typer.Option("fast", "--strategy", help="Strategy: fast, thorough, exhaustive"),
) -> None:
    """Detect synergistic and redundant layer circuits."""
    from neuro_scan.circuit import CircuitConfig, run_circuit_detection
    from neuro_scan.config import NeuroScanConfig
    from neuro_scan.scanner import run_ablation_scan

    backend_instance, probe_instance = _init_backend_and_probe(
        model, probe, backend, custom_probe, dtype, verbose
    )

    config = NeuroScanConfig(
        model_path=model,
        probe_name=probe,
        batch_size=batch_size,
        output_dir=output,
        backend=backend,
        dtype=dtype,
        custom_probe_path=custom_probe,
    )

    console.print(Panel("[bold]Circuit Detection[/bold]", title="neuro-scan"))

    # Step 1: Run ablation scan first
    console.print("[bold cyan]Running ablation scan...[/bold cyan]")
    baseline_score, _, ablation_results = run_ablation_scan(
        backend_instance, probe_instance, config
    )

    # Step 2: Run circuit detection
    circuit_config = CircuitConfig(
        top_k_pairs=top_k_pairs,
        strategy=strategy,
    )

    report = run_circuit_detection(
        backend=backend_instance,
        probe=probe_instance,
        ablation_results=ablation_results,
        baseline_score=baseline_score,
        circuit_config=circuit_config,
    )

    # Display results
    console.print(f"\n[bold]Circuit Detection Results ({report.strategy})[/bold]")
    console.print(f"  Pairs tested: {len(report.candidate_pairs)}")
    console.print(f"  Total evaluations: {report.total_evals}")
    console.print(f"  Time: {report.total_time_seconds:.1f}s")

    if report.synergistic_pairs:
        console.print("\n[bold green]Synergistic Pairs (cooperating layers):[/bold green]")
        for r in report.synergistic_pairs[:10]:
            console.print(
                f"  layers ({r.layer_i}, {r.layer_j}): "
                f"interaction={r.interaction_effect:+.4f} "
                f"(joint_delta={r.joint_delta:.4f}, "
                f"sum_individual={r.individual_delta_i + r.individual_delta_j:.4f})"
            )

    if report.redundant_pairs:
        console.print("\n[bold yellow]Redundant Pairs (overlapping layers):[/bold yellow]")
        for r in report.redundant_pairs[:10]:
            console.print(
                f"  layers ({r.layer_i}, {r.layer_j}): "
                f"interaction={r.interaction_effect:+.4f} "
                f"(joint_delta={r.joint_delta:.4f}, "
                f"sum_individual={r.individual_delta_i + r.individual_delta_j:.4f})"
            )

    # Save results
    import json

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_data = {
        "strategy": report.strategy,
        "top_k": report.top_k,
        "total_pairs": len(report.candidate_pairs),
        "total_evals": report.total_evals,
        "time_seconds": report.total_time_seconds,
        "synergistic": [
            {
                "layer_i": r.layer_i,
                "layer_j": r.layer_j,
                "interaction_effect": r.interaction_effect,
                "joint_delta": r.joint_delta,
                "individual_delta_i": r.individual_delta_i,
                "individual_delta_j": r.individual_delta_j,
            }
            for r in report.synergistic_pairs
        ],
        "redundant": [
            {
                "layer_i": r.layer_i,
                "layer_j": r.layer_j,
                "interaction_effect": r.interaction_effect,
                "joint_delta": r.joint_delta,
                "individual_delta_i": r.individual_delta_i,
                "individual_delta_j": r.individual_delta_j,
            }
            for r in report.redundant_pairs
        ],
        "all_interactions": [
            {
                "layer_i": r.layer_i,
                "layer_j": r.layer_j,
                "interaction_effect": r.interaction_effect,
                "interaction_type": r.interaction_type,
                "joint_delta": r.joint_delta,
            }
            for r in report.interactions
        ],
    }

    results_path = output_dir / "circuit.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    console.print(f"\n[green]Circuit report: {results_path}[/green]")

    backend_instance.cleanup()


@app.command(name="prompt-repeat")
def prompt_repeat(
    model: str = _model_option,
    probe: str = _probe_option,
    backend: str = _backend_option,
    batch_size: int = _batch_option,
    output: str = _output_option,
    custom_probe: str = _custom_probe_option,
    dtype: str = _dtype_option,
    verbose: bool = _verbose_option,
    repeat_counts: str = typer.Option(
        "1,2,3,4", "--repeat-counts", "-r",
        help="Comma-separated repetition counts to test",
    ),
) -> None:
    """Run prompt repetition experiment (Concept C).

    Tests the hypothesis: does repeating a prompt N times approximate
    the effect of duplicating K transformer layers?
    """
    _setup_logging(verbose)

    counts = [int(x.strip()) for x in repeat_counts.split(",")]

    probe_instance = _load_probe(probe, custom_probe)
    backend_instance = _load_backend(backend)

    console.print("[bold cyan]Loading model...[/bold cyan]")
    backend_instance.load(model, dtype=dtype)

    total_layers = backend_instance.get_total_layers()
    console.print(f"[green]Model loaded: {total_layers} layers[/green]")

    probe_instance.validate(backend_instance.get_tokenizer())

    console.print(
        Panel(
            f"[bold]Prompt Repetition Experiment[/bold]\n"
            f"Model: {model}\n"
            f"Probe: {probe}\n"
            f"Repeat counts: {counts}",
            title="neuro-scan",
        )
    )

    from neuro_scan.scoring import aggregate_scores, score_from_logits

    token_ids, score_values = probe_instance.get_score_token_ids(
        backend_instance.get_tokenizer()
    )
    samples = probe_instance.get_samples(count=batch_size)

    results_table: dict[int, float] = {}

    for n in counts:
        scores = []
        for sample in samples:
            # Repeat the prompt N times
            repeated_text = (sample.full_text + "\n") * n
            repeated_text = repeated_text.rstrip("\n")

            logits = backend_instance.forward(repeated_text)
            result = score_from_logits(
                logits=logits,
                score_token_ids=token_ids,
                score_values=score_values,
            )
            scores.append(result)

        mean_score, _ = aggregate_scores(scores)
        results_table[n] = mean_score
        console.print(f"  Repeat x{n}: score = {mean_score:.4f}")

    # Summary
    console.print("\n[bold]Results:[/bold]")
    baseline = results_table.get(1, 0.0)
    for n, score in sorted(results_table.items()):
        delta = score - baseline
        console.print(f"  x{n}: {score:.4f} (delta from x1: {delta:+.4f})")

    # Save results
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    import json

    results_data = {
        "model": model,
        "probe": probe,
        "repeat_results": {str(k): v for k, v in results_table.items()},
        "baseline_x1": baseline,
    }
    results_path = output_dir / "prompt_repeat.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    console.print(f"\n[green]Results: {results_path}[/green]")

    backend_instance.cleanup()
    console.print("[dim]Done.[/dim]")


@app.command()
def compare(
    reports: list[str] = typer.Argument(
        ..., help="Paths to neuro-scan report.json files (2 or more)",
    ),
    output: str = _output_option,
    verbose: bool = _verbose_option,
) -> None:
    """Compare neuroanatomy across multiple models."""
    from neuro_scan.compare import (
        generate_comparison_text,
        run_comparison,
    )

    _setup_logging(verbose)

    if len(reports) < 2:
        console.print("[red]Need at least 2 report files to compare[/red]")
        raise typer.Exit(1)

    console.print(
        Panel(
            f"[bold]Model Comparison[/bold]\n"
            f"Reports: {len(reports)}",
            title="neuro-scan",
        )
    )

    report = run_comparison(reports)

    console.print(generate_comparison_text(report))

    # Save results
    import json

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_data = {
        "models": [
            {
                "name": m.model_name,
                "total_layers": m.total_layers,
                "baseline_score": m.baseline_score,
                "reasoning_layers": m.reasoning_layers,
                "reasoning_fraction": m.reasoning_fraction,
                "top_important_layers": m.top_important_layers,
            }
            for m in report.models
        ],
        "similarity_matrix": report.similarity_matrix.tolist(),
        "shared_reasoning_layers": report.shared_reasoning_layers,
        "model_rankings": [
            {"model": name, "mean_sensitivity": sens}
            for name, sens in report.model_rankings
        ],
    }

    results_path = output_dir / "comparison.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    console.print(f"\n[green]Comparison results: {results_path}[/green]")


@app.command(name="cross-probe")
def cross_probe_cmd(
    model: str = _model_option,
    probe_list: str = typer.Option(
        "math,eq,json", "--probes",
        help="Comma-separated probe names",
    ),
    backend: str = _backend_option,
    batch_size: int = _batch_option,
    output: str = _output_option,
    dtype: str = _dtype_option,
    verbose: bool = _verbose_option,
    top_k: int = _top_k_option,
) -> None:
    """Compare layer importance across multiple probes."""
    import json

    from neuro_scan.config import NeuroScanConfig
    from neuro_scan.cross_probe import run_cross_probe_analysis

    _setup_logging(verbose)

    probe_names = [p.strip() for p in probe_list.split(",")]
    probe_instances = [_load_probe(name) for name in probe_names]

    backend_instance = _load_backend(backend)
    console.print("[bold cyan]Loading model...[/bold cyan]")
    backend_instance.load(model, dtype=dtype)

    total_layers = backend_instance.get_total_layers()
    console.print(f"[green]Model loaded: {total_layers} layers[/green]")

    for p in probe_instances:
        p.validate(backend_instance.get_tokenizer())

    config = NeuroScanConfig(
        model_path=model,
        probe_name=",".join(probe_names),
        batch_size=batch_size,
        output_dir=output,
        backend=backend,
        dtype=dtype,
        top_k_layers=top_k,
    )

    console.print(
        Panel(
            f"[bold]Cross-Probe Analysis[/bold]\n"
            f"Model: {model}\n"
            f"Probes: {', '.join(probe_names)}",
            title="neuro-scan",
        )
    )

    report = run_cross_probe_analysis(
        backend_instance, probe_instances, config, top_k=top_k
    )

    # Display results
    console.print(f"\nAnalysis time: {report.total_time_seconds:.1f}s\n")

    for pr in report.per_probe:
        console.print(
            f"[bold]{pr.probe_name}[/bold]: baseline={pr.baseline_score:.4f}, "
            f"top layers={pr.top_layers}"
        )

    if report.universal_layers:
        console.print(
            f"\n[bold green]Universal layers[/bold green] "
            f"(top-{top_k} for ALL probes): {report.universal_layers}"
        )
    else:
        console.print(
            f"\n[yellow]No universal layers found in top-{top_k}[/yellow]"
        )

    if report.probe_specific_layers:
        console.print("\n[bold]Probe-specific layers:[/bold]")
        for probe_name, layers in report.probe_specific_layers.items():
            console.print(f"  {probe_name} only: {layers}")

    # Correlation matrix
    if len(report.probe_names) > 1:
        console.print("\n[bold]Probe correlation matrix:[/bold]")
        header = "         " + "  ".join(f"{n:>8s}" for n in report.probe_names)
        console.print(header)
        for i, name in enumerate(report.probe_names):
            row = f"  {name:>6s}"
            for j in range(len(report.probe_names)):
                row += f"  {report.correlation_matrix[i, j]:8.4f}"
            console.print(row)

    # Save results
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_data = {
        "model": model,
        "probes": report.probe_names,
        "total_layers": report.total_layers,
        "universal_layers": report.universal_layers,
        "probe_specific_layers": report.probe_specific_layers,
        "correlation_matrix": report.correlation_matrix.tolist(),
        "per_probe": [
            {
                "probe": pr.probe_name,
                "baseline": pr.baseline_score,
                "top_layers": pr.top_layers,
                "ablation_deltas": pr.ablation_deltas,
            }
            for pr in report.per_probe
        ],
    }

    results_path = output_dir / "cross_probe.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    console.print(f"\n[green]Results: {results_path}[/green]")

    backend_instance.cleanup()


@app.command()
def calibrate(
    model: str = _model_option,
    output_lens: str = typer.Option(..., "--output", "-o", help="Output path for .safetensors lens file"),
    backend: str = _backend_option,
    dtype: str = _dtype_option,
    verbose: bool = _verbose_option,
    steps: int = typer.Option(250, "--steps", help="SGD training steps per layer"),
    num_texts: int = typer.Option(32, "--texts", help="Number of calibration texts"),
) -> None:
    """Train a tuned lens for improved logit lens analysis.

    Learns per-layer affine probes that project intermediate hidden
    states to final-layer-quality logits, reducing early-layer bias.
    """
    _setup_logging(verbose)

    from neuro_scan.tuned_lens import TunedLens

    backend_instance = _load_backend(backend)

    console.print("[bold cyan]Loading model...[/bold cyan]")
    backend_instance.load(model, dtype=dtype)

    total_layers = backend_instance.get_total_layers()
    console.print(f"[green]Model loaded: {total_layers} layers[/green]")

    console.print(
        Panel(
            f"[bold]Tuned Lens Calibration[/bold]\n"
            f"Model: {model}\n"
            f"Steps: {steps}\n"
            f"Calibration texts: {num_texts}",
            title="neuro-scan",
        )
    )

    # Generate calibration texts from built-in probes
    calibration_texts = []
    from neuro_scan.probes.math_probe import MathProbe
    from neuro_scan.probes.eq_probe import EqProbe
    from neuro_scan.probes.json_probe import JsonProbe

    for probe_cls in [MathProbe, EqProbe, JsonProbe]:
        p = probe_cls()
        for sample in p.get_samples():
            calibration_texts.append(sample.full_text)
            if len(calibration_texts) >= num_texts:
                break
        if len(calibration_texts) >= num_texts:
            break

    console.print(f"[cyan]Using {len(calibration_texts)} calibration texts[/cyan]")

    lens = TunedLens.train(
        backend=backend_instance,
        calibration_texts=calibration_texts,
        steps=steps,
    )

    lens.save(output_lens)
    console.print(f"\n[green]Tuned lens saved to {output_lens}[/green]")

    backend_instance.cleanup()
    console.print("[dim]Done.[/dim]")


@app.command()
def fetch(
    model: str = _model_option,
    probe: str = _probe_option,
    output_path: str = typer.Option(None, "--output", "-o", help="Save report.json to this path"),
    verbose: bool = _verbose_option,
) -> None:
    """Fetch pre-computed neuroanatomy report from HuggingFace Hub."""
    _setup_logging(verbose)

    from neuro_scan.fetch import fetch_results, format_fetch_result

    console.print(
        Panel(
            f"[bold]Pre-computed Fetch[/bold]\n"
            f"Model: {model}\n"
            f"Probe: {probe}",
            title="neuro-scan",
        )
    )

    console.print("[bold cyan]Searching HuggingFace Hub...[/bold cyan]")
    record = fetch_results(model, probe)

    if record is None:
        console.print(f"[yellow]No pre-computed report found for {model} / {probe}[/yellow]")
        console.print("[dim]Tip: run 'neuro-scan map' to generate your own report[/dim]")
        raise typer.Exit(1)

    console.print(format_fetch_result(record))

    if output_path:
        import json as json_mod

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json_mod.dump(record, f, indent=2)
        console.print(f"\n[green]Report saved to {out}[/green]")


@app.command()
def probes() -> None:
    """List available evaluation probes."""
    from neuro_scan.probes.eq_probe import EqProbe
    from neuro_scan.probes.json_probe import JsonProbe
    from neuro_scan.probes.math_probe import MathProbe

    console.print("[bold]Available Probes:[/bold]\n")

    for probe_cls in [MathProbe, EqProbe, JsonProbe]:
        p = probe_cls()
        samples = p.get_samples()
        console.print(f"  [cyan]{p.name}[/cyan]")
        console.print(f"    {p.description}")
        console.print(f"    Samples: {len(samples)}")
        console.print()

    console.print("  [cyan]custom[/cyan]")
    console.print("    Load from JSON file with --custom-probe <path>")


@app.command()
def version() -> None:
    """Show version."""
    from neuro_scan import __version__

    console.print(f"neuro-scan v{__version__}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
