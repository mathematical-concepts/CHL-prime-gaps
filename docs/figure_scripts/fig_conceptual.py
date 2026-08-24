"""Deterministic conceptual diagrams for the CHL2 v2.0.0 paper."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from figure_scripts.common import plt, save_figure


def _box(ax, xy: tuple[float, float], text: str) -> None:
    x, y = xy
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.45", "alpha": 0.12, "linewidth": 1.0},
        transform=ax.transAxes,
    )


def _arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str | None = None,
    *,
    label_y: float | None = None,
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "linewidth": 1.2},
    )
    if label:
        xm = (start[0] + end[0]) / 2.0
        ym = (start[1] + end[1]) / 2.0
        y = ym + 0.055 if label_y is None else label_y
        ax.text(xm, y, label, ha="center", va="bottom", fontsize=8.5, transform=ax.transAxes)


def _construction_chain(output_dir: Path, formats: list[str]) -> list[Path]:
    fig, ax = plt.subplots(figsize=(12.5, 3.8))
    ax.set_axis_off()
    xs = [0.08, 0.29, 0.50, 0.71, 0.92]
    labels = [
        "Static Hardy--Littlewood\ntriple weight",
        "Conditional ratio\n$R_Y(g_2\\mid g_1)$",
        "No-interior survival\n$P=e^{-\\Omega_Y^{\\rm path}}$",
        "Row-normalized\nCHL2 gap kernel",
        "Valid-edge\norientation lift",
    ]
    for x, label in zip(xs, labels):
        _box(ax, (x, 0.56), label)
    arrow_labels = ["condition", "enforce consecutivity", "anchor and normalize", "diagnostic projection"]
    for i, label in enumerate(arrow_labels):
        _arrow(
            ax,
            (xs[i] + 0.08, 0.56),
            (xs[i + 1] - 0.08, 0.56),
            label,
            label_y=0.75,
        )
    ax.text(
        0.5,
        0.13,
        "Scientific outputs follow the same order: kernel audits -> modular diagnostics -> tables/figures -> TeX/PDF.",
        ha="center",
        va="center",
        fontsize=9.5,
        transform=ax.transAxes,
    )
    return save_figure(fig, output_dir, "fig_construction_chain", formats)


def _tuple_anatomy(output_dir: Path, formats: list[str]) -> list[Path]:
    fig, ax = plt.subplots(figsize=(11.5, 3.6))
    ax.set_axis_off()
    left, mid, right = 0.10, 0.40, 0.90
    ax.plot([left, right], [0.52, 0.52], linewidth=1.2, transform=ax.transAxes)
    for x, marker, label in [
        (left, "o", "$p_n$\n(position $0$)"),
        (mid, "o", "$p_{n+1}$\n(position $g_1$)"),
        (right, "o", "$p_{n+2}$\n(position $g_1+g_2$)"),
    ]:
        ax.plot([x], [0.52], marker=marker, markersize=8, transform=ax.transAxes)
        ax.text(x, 0.34, label, ha="center", va="top", fontsize=10, transform=ax.transAxes)
    interior = np.linspace(mid + 0.08, right - 0.08, 6)
    ax.plot(interior, np.full_like(interior, 0.52), linestyle="None", marker="|", markersize=13, transform=ax.transAxes)
    ax.text(
        (mid + right) / 2,
        0.72,
        "possible interior prime locations $g_1+u$",
        ha="center",
        va="center",
        fontsize=10,
        transform=ax.transAxes,
    )
    ax.annotate(
        "$g_1$ observed",
        xy=((left + mid) / 2, 0.52),
        xytext=((left + mid) / 2, 0.82),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        arrowprops={"arrowstyle": "-[,widthB=5.0,lengthB=0.8", "linewidth": 1.0},
    )
    ax.annotate(
        "$g_2$ candidate must contain no prime",
        xy=((mid + right) / 2, 0.52),
        xytext=((mid + right) / 2, 0.12),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        arrowprops={"arrowstyle": "-[,widthB=8.5,lengthB=0.8", "linewidth": 1.0},
    )
    return save_figure(fig, output_dir, "fig_tuple_anatomy", formats)


def _repro_pipeline(output_dir: Path, formats: list[str]) -> list[Path]:
    fig, ax = plt.subplots(figsize=(11.8, 4.5))
    ax.set_axis_off()
    top = [(0.12, 0.70), (0.37, 0.70), (0.62, 0.70), (0.87, 0.70)]
    labels = [
        "Public scripts and\nconfiguration",
        "Local DS1 and\naudit outputs",
        "Generated tables and\nvector figures",
        "TeX source and\nrelease PDF",
    ]
    for xy, label in zip(top, labels):
        _box(ax, xy, label)
    for i in range(3):
        _arrow(ax, (top[i][0] + 0.095, top[i][1]), (top[i + 1][0] - 0.095, top[i + 1][1]))
    lower = [(0.22, 0.25), (0.50, 0.25), (0.78, 0.25)]
    lower_labels = [
        "Tests and scientific gates",
        "SHA-256 manifests and\nGit provenance",
        "PDF preflight and\nvisual review",
    ]
    for xy, label in zip(lower, lower_labels):
        _box(ax, xy, label)
    _arrow(ax, (0.37, 0.62), (0.27, 0.35))
    _arrow(ax, (0.62, 0.62), (0.50, 0.35))
    _arrow(ax, (0.87, 0.62), (0.78, 0.35))
    ax.text(
        0.5,
        0.04,
        "Generated data and outputs remain outside Git; the repository versions the producers and the paper source.",
        ha="center",
        va="bottom",
        fontsize=9.5,
        transform=ax.transAxes,
    )
    return save_figure(fig, output_dir, "fig_repro_pipeline", formats)


def build(outputs_root: Path, output_dir: Path, formats: list[str]) -> tuple[list[Path], list[Path]]:
    del outputs_root
    outputs: list[Path] = []
    outputs.extend(_construction_chain(output_dir, formats))
    outputs.extend(_tuple_anatomy(output_dir, formats))
    outputs.extend(_repro_pipeline(output_dir, formats))
    return outputs, []
