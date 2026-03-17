from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from neuro_scan.labeler import get_label_color

if TYPE_CHECKING:
    from neuro_scan.circuit import CircuitReport
    from neuro_scan.config import NeuroReport
    from neuro_scan.cross_probe import CrossProbeReport

logger = logging.getLogger(__name__)


def generate_ablation_chart(
    report: NeuroReport,
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    """Generate an interactive ablation sensitivity bar chart.

    X-axis: layer index (0..N-1)
    Y-axis: score_delta (baseline - ablated score)
    Color: auto-labeled layer function
    Stars: top-k most important layers

    Args:
        report: The neuro-scan report.
        output_path: Path for the output HTML file.
        title: Custom title.

    Returns:
        Path to the generated HTML file.
    """
    import plotly.graph_objects as go

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if title is None:
        model_name = Path(report.model_path).name
        title = f"Layer Ablation Sensitivity — {model_name} / {report.probe_name}"

    layers = list(range(report.total_layers))
    deltas = [r.score_delta for r in report.ablation_results]
    colors = [get_label_color(report.layer_labels.get(i, "")) for i in layers]
    labels = [report.layer_labels.get(i, "unknown") for i in layers]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=layers,
            y=deltas,
            marker_color=colors,
            text=labels,
            hovertemplate=(
                "Layer %{x}<br>"
                "Score delta: %{y:.4f}<br>"
                "Function: %{text}<br>"
                "<extra></extra>"
            ),
        )
    )

    # Mark top-k layers with stars
    for rank, layer_idx in enumerate(report.top_important_layers, 1):
        delta = report.ablation_results[layer_idx].score_delta
        fig.add_trace(
            go.Scatter(
                x=[layer_idx],
                y=[delta],
                mode="markers+text",
                marker={"size": 14, "color": "gold", "symbol": "star"},
                text=[f"#{rank}"],
                textposition="top center",
                name=f"Top {rank}: layer {layer_idx} (delta={delta:.4f})",
                showlegend=True,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Layer Index",
        yaxis_title="Score Delta (baseline - ablated)",
        width=1000,
        height=500,
        showlegend=True,
        xaxis={"dtick": max(1, report.total_layers // 20)},
    )

    # Baseline annotation
    fig.add_annotation(
        text=f"Baseline: {report.baseline_score:.4f}",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        font={"size": 12, "color": "gray"},
    )

    fig.write_html(str(output_path), include_plotlyjs="cdn")
    logger.info("Ablation chart saved to %s", output_path)

    return output_path


def generate_logit_lens_heatmap(
    report: NeuroReport,
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    """Generate a logit lens trajectory heatmap.

    X-axis: layer index
    Y-axis: sample index
    Color: target token probability at each layer

    Args:
        report: The neuro-scan report.
        output_path: Path for the output HTML file.
        title: Custom title.

    Returns:
        Path to the generated HTML file.
    """
    import plotly.graph_objects as go

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if title is None:
        model_name = Path(report.model_path).name
        title = f"Logit Lens Trajectory — {model_name} / {report.probe_name}"

    trajectories = report.logit_lens_trajectory
    if not trajectories:
        # Create empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No logit lens data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font={"size": 20},
        )
        fig.update_layout(title=title, width=1000, height=500)
        fig.write_html(str(output_path), include_plotlyjs="cdn")
        return output_path

    # Build probability matrix
    num_samples = len(trajectories)
    num_layers = len(trajectories[0]) if trajectories else 0

    prob_matrix = np.zeros((num_samples, num_layers))
    hover_text = []

    for s_idx, trajectory in enumerate(trajectories):
        row_hover = []
        for l_idx, step in enumerate(trajectory):
            prob_matrix[s_idx, l_idx] = step.target_token_prob
            row_hover.append(
                f"Layer {step.layer_idx}<br>"
                f"Top token: {step.top_token} ({step.top_token_prob:.3f})<br>"
                f"Target prob: {step.target_token_prob:.3f}<br>"
                f"Entropy: {step.entropy:.2f}"
            )
        hover_text.append(row_hover)

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=prob_matrix,
            x=list(range(num_layers)),
            y=[f"Sample {i}" for i in range(num_samples)],
            colorscale="Viridis",
            colorbar={"title": "Target Token Prob"},
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
        )
    )

    # Mark emergence layers per sample
    for s_idx, trajectory in enumerate(trajectories):
        for step in trajectory:
            if step.target_token_prob > 0.1:
                fig.add_trace(
                    go.Scatter(
                        x=[step.layer_idx],
                        y=[f"Sample {s_idx}"],
                        mode="markers",
                        marker={"size": 8, "color": "red", "symbol": "diamond"},
                        name=f"Emergence (sample {s_idx}, layer {step.layer_idx})",
                        showlegend=False,
                        hovertemplate=(
                            f"Emergence at layer {step.layer_idx}<br>"
                            f"Target prob: {step.target_token_prob:.3f}<extra></extra>"
                        ),
                    )
                )
                break  # Only mark first emergence

    fig.update_layout(
        title=title,
        xaxis_title="Layer Index",
        yaxis_title="Probe Sample",
        width=1000,
        height=max(400, num_samples * 30 + 200),
    )

    fig.write_html(str(output_path), include_plotlyjs="cdn")
    logger.info("Logit lens heatmap saved to %s", output_path)

    return output_path


def generate_attention_heatmap(
    report: NeuroReport,
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    """Generate an attention entropy heatmap.

    X-axis: layer index
    Y-axis: attention head index
    Color: entropy (high = diffuse attention, low = focused)

    Args:
        report: The neuro-scan report.
        output_path: Path for the output HTML file.
        title: Custom title.

    Returns:
        Path to the generated HTML file.
    """
    import plotly.graph_objects as go

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if title is None:
        model_name = Path(report.model_path).name
        title = f"Attention Entropy — {model_name} / {report.probe_name}"

    entropy_data = report.attention_entropy
    if not entropy_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No attention data available (backend may not support attention extraction)",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font={"size": 16},
        )
        fig.update_layout(title=title, width=1000, height=500)
        fig.write_html(str(output_path), include_plotlyjs="cdn")
        return output_path

    # Build entropy matrix: [num_heads x num_layers]
    num_layers = len(entropy_data)
    num_heads = len(entropy_data[0]) if entropy_data else 0

    entropy_matrix = np.zeros((num_heads, num_layers))
    for l_idx, head_entropies in enumerate(entropy_data):
        for h_idx, ent in enumerate(head_entropies):
            entropy_matrix[h_idx, l_idx] = ent

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=entropy_matrix,
            x=list(range(num_layers)),
            y=[f"Head {i}" for i in range(num_heads)],
            colorscale="RdYlBu_r",  # Red = high entropy, Blue = low
            colorbar={"title": "Entropy (nats)"},
            hovertemplate=(
                "Layer %{x}, Head %{y}<br>"
                "Entropy: %{z:.3f}<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Layer Index",
        yaxis_title="Attention Head",
        width=1000,
        height=max(400, num_heads * 15 + 200),
    )

    fig.write_html(str(output_path), include_plotlyjs="cdn")
    logger.info("Attention heatmap saved to %s", output_path)

    return output_path


def generate_entropy_profile_chart(
    report: NeuroReport,
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    """Generate an entropy profile across layers from logit lens data.

    Shows how the full-vocab entropy changes across layers — high entropy
    in early layers (model uncertain) decreasing toward final layers.
    """
    import plotly.graph_objects as go

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if title is None:
        model_name = Path(report.model_path).name
        title = f"Entropy Profile — {model_name} / {report.probe_name}"

    trajectories = report.logit_lens_trajectory
    if not trajectories:
        fig = go.Figure()
        fig.add_annotation(
            text="No logit lens data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font={"size": 20},
        )
        fig.update_layout(title=title, width=1000, height=500)
        fig.write_html(str(output_path), include_plotlyjs="cdn")
        return output_path

    num_layers = len(trajectories[0])

    # Compute mean and std entropy per layer
    entropy_per_layer = np.zeros((len(trajectories), num_layers))
    for s_idx, traj in enumerate(trajectories):
        for l_idx, step in enumerate(traj):
            entropy_per_layer[s_idx, l_idx] = step.entropy

    mean_entropy = entropy_per_layer.mean(axis=0)
    std_entropy = entropy_per_layer.std(axis=0)
    layers = list(range(num_layers))

    fig = go.Figure()

    # Confidence band
    fig.add_trace(
        go.Scatter(
            x=layers + layers[::-1],
            y=list(mean_entropy + std_entropy) + list((mean_entropy - std_entropy)[::-1]),
            fill="toself",
            fillcolor="rgba(99, 110, 250, 0.2)",
            line={"color": "rgba(255,255,255,0)"},
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Mean line
    fig.add_trace(
        go.Scatter(
            x=layers,
            y=list(mean_entropy),
            mode="lines+markers",
            name="Mean entropy",
            line={"color": "#636EFA", "width": 2},
            marker={"size": 4},
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Layer Index",
        yaxis_title="Entropy (nats)",
        width=1000,
        height=500,
    )

    fig.write_html(str(output_path), include_plotlyjs="cdn")
    logger.info("Entropy profile chart saved to %s", output_path)

    return output_path


def generate_interaction_heatmap(
    report: "CircuitReport",
    total_layers: int,
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    """Generate an interaction effect heatmap from circuit detection.

    Shows synergistic (red) and redundant (blue) layer pairs.
    """
    import plotly.graph_objects as go

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if title is None:
        title = f"Layer Interaction Effects ({report.strategy})"

    matrix = np.full((total_layers, total_layers), np.nan)
    for r in report.interactions:
        matrix[r.layer_i, r.layer_j] = r.interaction_effect
        matrix[r.layer_j, r.layer_i] = r.interaction_effect

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=matrix,
            x=list(range(total_layers)),
            y=list(range(total_layers)),
            colorscale="RdBu_r",
            zmid=0,
            colorbar={"title": "Interaction Effect"},
            hovertemplate=(
                "Layer %{x} × Layer %{y}<br>"
                "Interaction: %{z:.4f}<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Layer Index",
        yaxis_title="Layer Index",
        width=800,
        height=800,
    )

    fig.write_html(str(output_path), include_plotlyjs="cdn")
    logger.info("Interaction heatmap saved to %s", output_path)

    return output_path


def generate_cross_probe_chart(
    report: "CrossProbeReport",
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    """Generate a grouped bar chart comparing ablation deltas across probes.

    X-axis: layer index (0..N-1)
    Y-axis: ablation delta (baseline - ablated score)
    Groups: one bar color per probe

    Args:
        report: The cross-probe analysis report.
        output_path: Path for the output HTML file.
        title: Custom title.

    Returns:
        Path to the generated HTML file.
    """
    import plotly.graph_objects as go

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if title is None:
        title = f"Cross-Probe Layer Sensitivity — {', '.join(report.probe_names)}"

    layers = list(range(report.total_layers))

    # Color palette for probes
    colors = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA",
        "#FFA15A", "#19D3F3", "#FF6692", "#B6E880",
    ]

    fig = go.Figure()

    for i, pr in enumerate(report.per_probe):
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Bar(
                x=layers,
                y=pr.ablation_deltas,
                name=pr.probe_name,
                marker_color=color,
                opacity=0.7,
                hovertemplate=(
                    f"Probe: {pr.probe_name}<br>"
                    "Layer %{x}<br>"
                    "Delta: %{y:.4f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    # Mark universal layers
    if report.universal_layers:
        for layer_idx in report.universal_layers:
            fig.add_vline(
                x=layer_idx, line_dash="dash",
                line_color="gold", line_width=1,
                annotation_text="U",
                annotation_position="top",
            )

    fig.update_layout(
        title=title,
        xaxis_title="Layer Index",
        yaxis_title="Ablation Delta (baseline - ablated)",
        barmode="group",
        width=1200,
        height=500,
        showlegend=True,
        xaxis={"dtick": max(1, report.total_layers // 20)},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )

    fig.write_html(str(output_path), include_plotlyjs="cdn")
    logger.info("Cross-probe chart saved to %s", output_path)

    return output_path


def generate_summary_text(report: NeuroReport) -> str:
    """Generate a text summary of neuroanatomy results.

    Args:
        report: The neuro-scan report.

    Returns:
        Formatted text summary.
    """
    lines = [
        "=" * 60,
        "NEURO-SCAN RESULTS",
        "=" * 60,
        f"Model: {report.model_path}",
        f"Probe: {report.probe_name}",
        f"Total layers: {report.total_layers}",
        f"Scan time: {report.total_time_seconds:.1f}s",
        "",
        f"Baseline score: {report.baseline_score:.4f} "
        f"(+/-{report.baseline_uncertainty:.4f})",
        "",
        "TOP IMPORTANT LAYERS (by ablation sensitivity):",
        "-" * 60,
    ]

    for rank, layer_idx in enumerate(report.top_important_layers, 1):
        result = report.ablation_results[layer_idx]
        label = report.layer_labels.get(layer_idx, "unknown")
        lines.append(
            f"  #{rank}: layer {layer_idx:3d} "
            f"(delta={result.score_delta:+.4f}) "
            f"[{label}]"
        )

    lines.extend(["", "LAYER FUNCTION DISTRIBUTION:", "-" * 60])

    # Count labels
    label_counts: dict[str, int] = {}
    for label in report.layer_labels.values():
        label_counts[label] = label_counts.get(label, 0) + 1

    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = count / report.total_layers * 100
        lines.append(f"  {label:25s}: {count:3d} layers ({pct:.1f}%)")

    # Block Influence summary
    if report.block_influence:
        bi = report.block_influence
        lines.extend(["", "BLOCK INFLUENCE (ShortGPT BI metric):", "-" * 60])
        lines.append(f"  Mean BI: {np.mean(bi):.4f}")
        lines.append(f"  Max  BI: {np.max(bi):.4f} (layer {int(np.argmax(bi)) + 1})")
        lines.append(f"  Min  BI: {np.min(bi):.4f} (layer {int(np.argmin(bi)) + 1})")
        # Top 5 highest BI layers
        sorted_bi = sorted(enumerate(bi), key=lambda x: x[1], reverse=True)
        lines.append("  Top-5 highest BI (most transformative):")
        for rank, (idx, score) in enumerate(sorted_bi[:5], 1):
            lines.append(f"    #{rank}: layer {idx + 1:3d} (BI={score:.4f})")
        lines.append("  Top-5 lowest BI (most removable):")
        for rank, (idx, score) in enumerate(sorted_bi[-5:], 1):
            lines.append(f"    #{rank}: layer {idx + 1:3d} (BI={score:.4f})")

    lines.extend(["", "=" * 60])
    return "\n".join(lines)
