"""Gap-length-filter scale-wave figures."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from release_assets import SourceSpec, read_csv_checked
from figure_scripts.common import G2_FILTER_PLOT, plt, save_figure


def build(outputs_root: Path, output_dir: Path, formats: list[str]) -> tuple[list[Path], list[Path]]:
    source = SourceSpec("A05", "chl4d2_scale_wave_summary.csv").resolve(outputs_root)
    df = read_csv_checked(
        source,
        [
            "filter",
            "D3_empirical_gap_population",
            "D3_chl2_gap_population",
            "D3_residual_gap_population",
            "empirical_events",
        ],
    )
    order = [name for name in G2_FILTER_PLOT if name in set(df["filter"].astype(str))]
    sub = df.set_index("filter").loc[order].reset_index()
    labels = [G2_FILTER_PLOT[name] for name in order]
    x = np.arange(len(labels), dtype=float)
    outputs: list[Path] = []

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.plot(x, sub["D3_empirical_gap_population"].astype(float), marker="o", label="Empirical")
    ax.plot(x, sub["D3_chl2_gap_population"].astype(float), marker="s", label="CHL2")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xticks(x, labels, rotation=28, ha="right")
    ax.set_ylabel(r"$D_3$ from the gap-residue population")
    ax.set_title(r"Scale wave across filters of the candidate gap $g_2$")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    outputs.extend(save_figure(fig, output_dir, "fig_scale_wave_d3", formats))

    fig, ax = plt.subplots(figsize=(10.5, 5.3))
    bars = ax.bar(x, sub["D3_residual_gap_population"].astype(float))
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xticks(x, labels, rotation=28, ha="right")
    ax.set_ylabel(r"$D_3^{emp}-D_3^{CHL2}$")
    ax.set_title(r"Residual scale wave by $g_2$ filter")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    outputs.extend(save_figure(fig, output_dir, "fig_scale_wave_residual", formats))
    return outputs, [source]
