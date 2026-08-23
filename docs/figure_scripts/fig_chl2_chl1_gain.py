"""CHL2-vs-CHL1 block-stability figure."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from release_assets import SourceSpec, read_csv_checked
from figure_scripts.common import STRATUM_PLOT, annotate_bars, plt, save_figure

CHL1 = "CHL1_ratio_only_cond_eta"
CHL2 = "CHL2_path_excl_cond_eta"


def build(outputs_root: Path, output_dir: Path, formats: list[str]) -> tuple[list[Path], list[Path]]:
    source = SourceSpec("A02", "chl2_metrics_by_block.csv").resolve(outputs_root)
    df = read_csv_checked(source, ["filter_name", "model", "block", "n_events", "loglik_sum"])
    df = df[df["model"].isin([CHL1, CHL2])].copy()

    order = [name for name in STRATUM_PLOT if name in set(df["filter_name"].astype(str))]
    labels: list[str] = []
    means: list[float] = []
    stds: list[float] = []
    for filt in order:
        vals: list[float] = []
        sub = df[df["filter_name"].astype(str) == filt]
        for _, grp in sub.groupby("block"):
            by_model = grp.set_index("model")
            if CHL1 not in by_model.index or CHL2 not in by_model.index:
                continue
            n = float(by_model.loc[CHL2, "n_events"])
            if n > 0:
                vals.append(float((by_model.loc[CHL2, "loglik_sum"] - by_model.loc[CHL1, "loglik_sum"]) / n))
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        labels.append(STRATUM_PLOT[filt])
        means.append(float(arr.mean()))
        stds.append(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0)

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4)
    annotate_bars(ax, bars, digits=3)
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xticks(x, labels, rotation=28, ha="right")
    ax.set_ylabel(r"CHL2 $-$ CHL1 log-likelihood per event")
    ax.set_title("CHL2 gain over CHL1 across DS1 strata")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return save_figure(fig, output_dir, "fig_chl2_chl1_gain", formats), [source]
