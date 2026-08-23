"""Orientation-lift diagonal and KL comparison figures."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from release_assets import SourceSpec, read_csv_checked
from figure_scripts.common import annotate_bars, plt, save_figure


def build(outputs_root: Path, output_dir: Path, formats: list[str]) -> tuple[list[Path], list[Path]]:
    source = SourceSpec("A08", "chl2_os_old_direct_vs_oriented.csv").resolve(outputs_root)
    df = read_csv_checked(
        source,
        [
            "block",
            "q",
            "emp_diag",
            "old_diag_model",
            "oriented_diag_model",
            "old_kl",
            "oriented_kl",
            "old_model_label",
        ],
    )
    sub = df[df["block"].astype(str) == "ALL"].sort_values("q")
    if sub.empty:
        raise ValueError("Orientation-lift comparison has no ALL aggregate")
    labels = set(sub["old_model_label"].astype(str))
    if labels != {"absolute_prime_os_previous_gap_conditioned"}:
        raise ValueError(f"Unexpected old-model label(s): {sorted(labels)}")

    q = sub["q"].astype(int).to_numpy()
    x = np.arange(len(q), dtype=float)
    width = 0.24
    outputs: list[Path] = []

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    b1 = ax.bar(x - width, sub["emp_diag"].astype(float), width, label="Empirical")
    b2 = ax.bar(x, sub["old_diag_model"].astype(float), width, label="Previous-gap control")
    b3 = ax.bar(x + width, sub["oriented_diag_model"].astype(float), width, label="Orientation lift")
    ax.set_xticks(x, [str(v) for v in q])
    ax.set_xlabel(r"Prime modulus $q$")
    ax.set_ylabel("Mean diagonal probability")
    ax.set_title("Absolute-prime modular diagnostic before and after orientation lift")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    outputs.extend(save_figure(fig, output_dir, "fig_orientation_lift_diagonal", formats))

    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    b1 = ax.bar(x - width / 2.0, sub["old_kl"].astype(float), width, label="Previous-gap control")
    b2 = ax.bar(x + width / 2.0, sub["oriented_kl"].astype(float), width, label="Orientation lift")
    ax.set_xticks(x, [str(v) for v in q])
    ax.set_xlabel(r"Prime modulus $q$")
    ax.set_ylabel("Weighted KL divergence")
    ax.set_yscale("log")
    ax.set_title("Orientation lift reduces modular KL divergence")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    outputs.extend(save_figure(fig, output_dir, "fig_orientation_lift_kl", formats))
    return outputs, [source]
