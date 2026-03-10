"""Standard plots for all experiment phases."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.2)

CATEGORY_LABELS = {
    "a": "A: Artist-named",
    "b": "B: Style-matched",
    "c": "C: Generic",
    "d": "D: Out-of-distribution",
}
CATEGORY_COLORS = {
    "a": "#e74c3c",
    "b": "#f39c12",
    "c": "#3498db",
    "d": "#95a5a6",
}


def save_fig(fig, name: str, output_dir: str = "results/plots"):
    """Save figure as both PNG and PDF."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_signal_vs_steps(
    detections_by_step: dict[int, pd.DataFrame],
    metric: str = "ghost_confidence",
    title: str = "Ghost Signal vs Training Steps",
    output_name: str = "signal_vs_steps",
    output_dir: str = "results/plots",
):
    """Plot detection metric across training checkpoints, broken down by prompt category."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for cat in ["a", "b", "c", "d"]:
        steps = sorted(detections_by_step.keys())
        means = []
        stds = []
        for step in steps:
            df = detections_by_step[step]
            cat_df = df[df["prompt_category"] == cat]
            means.append(cat_df[metric].mean())
            stds.append(cat_df[metric].std())
        means = np.array(means)
        stds = np.array(stds)
        ax.plot(steps, means, "o-", label=CATEGORY_LABELS[cat], color=CATEGORY_COLORS[cat])
        ax.fill_between(steps, means - stds, means + stds, alpha=0.15, color=CATEGORY_COLORS[cat])

    ax.set_xlabel("Training Step")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.legend()
    save_fig(fig, output_name, output_dir)


def plot_detection_rate_vs_steps(
    detections_by_step: dict[int, pd.DataFrame],
    output_name: str = "detection_rate_vs_steps",
    output_dir: str = "results/plots",
):
    """Plot detection rate across training checkpoints by prompt category."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for cat in ["a", "b", "c", "d"]:
        steps = sorted(detections_by_step.keys())
        rates = []
        for step in steps:
            df = detections_by_step[step]
            cat_df = df[df["prompt_category"] == cat]
            rates.append(cat_df["detected"].mean())
        ax.plot(steps, rates, "o-", label=CATEGORY_LABELS[cat], color=CATEGORY_COLORS[cat])

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Detection Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Detection Rate vs Training Steps")
    ax.legend()
    save_fig(fig, output_name, output_dir)


def plot_correlation_histogram(
    watermarked_scores: np.ndarray,
    null_scores: np.ndarray,
    title: str = "Ghost Confidence Distribution",
    output_name: str = "correlation_histogram",
    output_dir: str = "results/plots",
):
    """Overlaid histogram: watermarked vs null distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(null_scores, bins=50, alpha=0.6, label="Null (unwatermarked)", color="#3498db", density=True)
    ax.hist(watermarked_scores, bins=50, alpha=0.6, label="Watermarked", color="#e74c3c", density=True)
    ax.set_xlabel("Ghost Confidence")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    save_fig(fig, output_name, output_dir)


def plot_dilution_curve(
    ratios: list[float],
    detection_rates: list[float],
    mean_confidences: list[float],
    std_confidences: list[float],
    output_name: str = "dilution_curve",
    output_dir: str = "results/plots",
):
    """Detection rate and mean confidence vs watermarked ratio."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = "#e74c3c"
    ax1.set_xlabel("Watermarked Ratio")
    ax1.set_ylabel("Detection Rate", color=color1)
    ax1.plot(ratios, detection_rates, "o-", color=color1, label="Detection Rate")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(-0.05, 1.05)

    ax2 = ax1.twinx()
    color2 = "#3498db"
    ax2.set_ylabel("Mean Ghost Confidence", color=color2)
    means = np.array(mean_confidences)
    stds = np.array(std_confidences)
    ax2.plot(ratios, means, "s--", color=color2, label="Mean Confidence")
    ax2.fill_between(ratios, means - stds, means + stds, alpha=0.15, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    fig.suptitle("Dilution Series: Signal Survival")
    fig.tight_layout()
    save_fig(fig, output_name, output_dir)


def plot_attack_survival_matrix(
    df: pd.DataFrame,
    output_name: str = "attack_survival",
    output_dir: str = "results/plots",
):
    """Heatmap of detection rates across attack types."""
    pivot = df.pivot_table(
        values="detected",
        index="attack_type",
        columns="attack_param",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax, vmin=0, vmax=1)
    ax.set_title("Post-Generation Attack Survival (Detection Rate)")
    save_fig(fig, output_name, output_dir)
