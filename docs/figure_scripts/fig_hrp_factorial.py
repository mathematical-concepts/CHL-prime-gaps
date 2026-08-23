"""H/R/P factorial ranking and effect figures."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from release_assets import MODEL_DISPLAY, SourceSpec, read_csv_checked
from figure_scripts.common import annotate_bars, plt, save_figure


def build(outputs_root: Path, output_dir: Path, formats: list[str]) -> tuple[list[Path], list[Path]]:
    model_source = SourceSpec("A10", "chl2_factor_model_summary.csv").resolve(outputs_root)
    effect_source = SourceSpec("A10", "chl2_factorial_effects_summary.csv").resolve(outputs_root)
    models = read_csv_checked(
        model_source,
        ["filter", "model", "rank_by_loglik", "loglik_per_event", "H", "R", "P"],
    )
    effects = read_csv_checked(
        effect_source,
        ["filter", "effect", "pooled_effect_loglik_per_event"],
    )
    outputs: list[Path] = []

    sub = models[models["filter"].astype(str) == "ALL"].sort_values("rank_by_loglik")
    if len(sub) != 8:
        raise ValueError(f"Expected eight H/R/P vertices in ALL, found {len(sub)}")
    labels = [MODEL_DISPLAY.get(str(x), str(x)) for x in sub["model"]]
    x = np.arange(len(labels), dtype=float)
    # Plot relative to the best model so small differences remain visible.
    best = float(sub["loglik_per_event"].max())
    losses = best - sub["loglik_per_event"].astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    bars = ax.bar(x, losses)
    ax.set_xticks(x, labels, rotation=32, ha="right")
    ax.set_ylabel("Log-likelihood loss per event relative to the best vertex")
    ax.set_title("H/R/P factorial model ranking in the ALL stratum")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    outputs.extend(save_figure(fig, output_dir, "fig_hrp_model_ranking", formats))

    order = ["H", "R", "P", "H:R", "H:P", "R:P", "H:R:P"]
    esub = effects[effects["filter"].astype(str) == "ALL"].set_index("effect")
    missing = [name for name in order if name not in esub.index]
    if missing:
        raise ValueError(f"Missing H/R/P factorial effects: {missing}")
    values = np.asarray([float(esub.loc[name, "pooled_effect_loglik_per_event"]) for name in order])
    fig, ax = plt.subplots(figsize=(9.2, 5.5))
    bars = ax.bar(np.arange(len(order)), values)
    lower = min(0.0, float(values.min()))
    upper = max(0.0, float(values.max()))
    span = max(upper - lower, 1e-12)
    ax.set_ylim(lower - 0.18 * span, upper + 0.14 * span)
    annotate_bars(ax, bars, digits=3)
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xticks(np.arange(len(order)), order)
    ax.set_ylabel("Factorial effect on log-likelihood per event")
    ax.set_title("H/R/P main effects and interactions in the ALL stratum")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    outputs.extend(save_figure(fig, output_dir, "fig_hrp_factorial_effects", formats))
    return outputs, [model_source, effect_source]
