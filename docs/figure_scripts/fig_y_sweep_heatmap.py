"""Truncation-horizon Y-sweep heatmap."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.colors import TwoSlopeNorm

from release_assets import SourceSpec, read_csv_checked
from figure_scripts.common import STRATUM_PLOT, plt, save_figure

CHL1 = "CHL1_ratio_only_cond_eta"
CHL2 = "CHL2_path_excl_cond_eta"


def build(outputs_root: Path, output_dir: Path, formats: list[str]) -> tuple[list[Path], list[Path]]:
    source = SourceSpec("A04", "chl2_y_sweep_gains.csv").resolve(outputs_root)
    df = read_csv_checked(
        source,
        ["Y", "filter", "model", "baseline", "delta_loglik_model_minus_baseline"],
    )
    df = df[(df["model"] == CHL2) & (df["baseline"] == CHL1)].copy()
    if df.empty:
        raise ValueError("No CHL2-vs-CHL1 rows found in the Y-sweep gains")

    y_values = sorted(int(v) for v in df["Y"].unique())
    filters = [name for name in STRATUM_PLOT if name in set(df["filter"].astype(str))]
    matrix = np.full((len(filters), len(y_values)), np.nan, dtype=float)
    for i, filt in enumerate(filters):
        for j, y in enumerate(y_values):
            hit = df[(df["filter"].astype(str) == filt) & (df["Y"].astype(int) == y)]
            if not hit.empty:
                matrix[i, j] = float(hit.iloc[0]["delta_loglik_model_minus_baseline"])

    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        raise ValueError("The Y-sweep heatmap contains no finite values")
    vmin = float(finite.min())
    vmax = float(finite.max())
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax) if vmin < 0.0 < vmax else None

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", norm=norm, interpolation="nearest")
    ax.set_xticks(np.arange(len(y_values)), [str(y) for y in y_values])
    ax.set_yticks(np.arange(len(filters)), [STRATUM_PLOT[f] for f in filters])
    ax.set_xlabel(r"Truncation horizon $Y$")
    ax.set_ylabel("Likelihood stratum")
    ax.set_title("CHL2 minus CHL1 across truncation horizons")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                red, green, blue, _ = image.cmap(image.norm(value))
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                text_color = "white" if luminance < 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{value:.2e}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                )
    fig.colorbar(image, ax=ax, label=r"$\Delta\ell$ per event")
    fig.tight_layout()
    return save_figure(fig, output_dir, "fig_y_sweep_heatmap", formats), [source]
