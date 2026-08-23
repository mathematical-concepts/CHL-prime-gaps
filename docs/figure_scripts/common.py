"""Common plotting helpers for CHL2 v2.0.0 release figures."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


STRATUM_PLOT = {
    "ALL": r"$S_{all}$",
    "LOW_ONLY_LE58": r"$S_{dense}$",
    "MID_59_120": r"$S_{trans}$",
    "MID_121_240": r"$S_{121:240}$",
    "MID_121_400": r"$S_{121:400}$",
    "NO_58": r"$S_{>58}$",
    "NO_120": r"$S_{>120}$",
    "NO_240": r"$S_{>240}$",
}

G2_FILTER_PLOT = {
    "ALL": r"$g_2$: all",
    "LOW_ONLY_LE58": r"$g_2\leq58$",
    "MID_59_120": r"$59\leq g_2\leq120$",
    "MID_121_240": r"$121\leq g_2\leq240$",
    "MID_121_400": r"$121\leq g_2\leq400$",
    "NO_58": r"$g_2>58$",
    "NO_120": r"$g_2>120$",
    "NO_240": r"$g_2>240$",
}


def parse_formats(text: str) -> list[str]:
    formats = [x.strip().lower() for x in text.split(",") if x.strip()]
    allowed = {"svg", "pdf", "png"}
    bad = sorted(set(formats) - allowed)
    if bad:
        raise ValueError(f"Unsupported figure formats: {bad}; allowed={sorted(allowed)}")
    if not formats:
        raise ValueError("At least one figure format is required")
    return formats


def save_figure(fig, output_dir: Path, stem: str, formats: Sequence[str]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        kwargs = {"bbox_inches": "tight"}
        if fmt == "png":
            kwargs["dpi"] = 180
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def annotate_bars(ax, bars, *, digits: int = 4) -> None:
    for bar in bars:
        height = float(bar.get_height())
        ax.annotate(
            f"{height:.{digits}g}",
            xy=(bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, 3 if height >= 0 else -12),
            textcoords="offset points",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=8,
        )
