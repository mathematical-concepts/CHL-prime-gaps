#!/usr/bin/env python3
"""Generate CHL2 v2.0.0 LaTeX table fragments from local audit outputs.

The script never embeds scientific constants.  Every displayed number is read
from the canonical ``outputs/v2_release_*`` directories and every generated
fragment records the SHA-256 of its source CSV(s).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

DOCS_DIR = Path(__file__).resolve().parent
if str(DOCS_DIR) not in sys.path:
    sys.path.insert(0, str(DOCS_DIR))

from release_assets import (  # noqa: E402
    G2_FILTER_TEX,
    as_bool,
    MODEL_DISPLAY,
    STRATUM_TEX,
    SourceSpec,
    fmt_count_ratio,
    fmt_fixed,
    fmt_gain,
    fmt_integer,
    fmt_share,
    output_record,
    portable_path,
    read_csv_checked,
    raw_tex,
    render_tabular,
    tex_escape,
    sha256_file,
    source_record,
    write_json,
    write_text,
)


CHL1 = "CHL1_ratio_only_cond_eta"
CHL2 = "CHL2_path_excl_cond_eta"
PRINCIPAL_FILTER_ORDER = list(STRATUM_TEX)
G2_FILTER_ORDER = list(G2_FILTER_TEX)


class TableBuildError(RuntimeError):
    pass


def _ordered_values(values: Sequence[str], order: Sequence[str]) -> list[str]:
    present = set(map(str, values))
    return [name for name in order if name in present] + sorted(present - set(order))


def _write_table(output_dir: Path, name: str, text: str) -> Path:
    return write_text(output_dir / f"{name}.tex", text)


def build_main_likelihood(outputs_root: Path, output_dir: Path) -> tuple[Path, list[Path]]:
    src = SourceSpec("A02", "chl2_conditional_summary.csv").resolve(outputs_root)
    df = read_csv_checked(
        src,
        ["filter", "model", "n_events", "loglik_per_event", "conditional_kl"],
    )
    keep = df[df["model"].isin([CHL1, CHL2])].copy()
    if set(keep["model"]) != {CHL1, CHL2}:
        raise TableBuildError("A02 summary must contain both CHL1 and CHL2 path models")
    rows: list[list[object]] = []
    for filt in _ordered_values(keep["filter"].astype(str).unique(), PRINCIPAL_FILTER_ORDER):
        sub = keep[keep["filter"].astype(str) == filt].set_index("model")
        if CHL1 not in sub.index or CHL2 not in sub.index:
            continue
        n_events = float(sub.loc[CHL2, "n_events"])
        ll1 = float(sub.loc[CHL1, "loglik_per_event"])
        ll2 = float(sub.loc[CHL2, "loglik_per_event"])
        rows.append([
            STRATUM_TEX.get(filt, filt),
            fmt_integer(n_events),
            fmt_fixed(ll1, 9),
            fmt_fixed(ll2, 9),
            fmt_gain(ll2 - ll1, 6),
            fmt_gain(sub.loc[CHL2, "conditional_kl"], 6),
        ])
    text = render_tabular(
        headers=["Stratum", "Events", "CHL1 loglik/event", "CHL2 loglik/event", raw_tex(r"$\Delta\ell$"), "CHL2 KL"],
        rows=rows,
        align="lrrrrr",
        sources=[src],
    )
    return _write_table(output_dir, "table_main_likelihood", text), [src]


def build_block_stability(outputs_root: Path, output_dir: Path) -> tuple[Path, list[Path]]:
    src = SourceSpec("A02", "chl2_metrics_by_block.csv").resolve(outputs_root)
    df = read_csv_checked(src, ["filter_name", "model", "block", "n_events", "loglik_sum"])
    df = df[df["model"].isin([CHL1, CHL2])].copy()
    rows: list[list[object]] = []
    for filt in _ordered_values(df["filter_name"].astype(str).unique(), PRINCIPAL_FILTER_ORDER):
        sub = df[df["filter_name"].astype(str) == filt]
        metrics: list[float] = []
        for block, grp in sub.groupby("block"):
            by_model = grp.set_index("model")
            if CHL1 not in by_model.index or CHL2 not in by_model.index:
                continue
            n = float(by_model.loc[CHL2, "n_events"])
            if n <= 0:
                continue
            metrics.append(float((by_model.loc[CHL2, "loglik_sum"] - by_model.loc[CHL1, "loglik_sum"]) / n))
        if not metrics:
            continue
        arr = np.asarray(metrics, dtype=float)
        rows.append([
            STRATUM_TEX.get(filt, filt),
            fmt_gain(arr.mean(), 6),
            fmt_gain(arr.std(ddof=1) if len(arr) > 1 else 0.0, 6),
            fmt_gain(arr.min(), 6),
            fmt_gain(arr.max(), 6),
            fmt_count_ratio(int((arr > 0).sum()), len(arr)),
        ])
    text = render_tabular(
        headers=["Stratum", raw_tex(r"Unweighted block mean $\Delta\ell$"), "Block SD", "Minimum", "Maximum", "Positive blocks"],
        rows=rows,
        align="lrrrrr",
        sources=[src],
    )
    return _write_table(output_dir, "table_chl2_chl1_block_stability", text), [src]


def build_y_sweep(outputs_root: Path, output_dir: Path) -> tuple[Path, list[Path]]:
    src = SourceSpec("A04", "chl2_y_sweep_gains.csv").resolve(outputs_root)
    df = read_csv_checked(
        src,
        ["Y", "filter", "model", "baseline", "delta_loglik_model_minus_baseline"],
    )
    sub = df[
        (df["model"] == CHL2)
        & (df["baseline"] == CHL1)
    ].copy()
    if sub.empty:
        raise TableBuildError("A04 gains contain no CHL2-vs-CHL1 rows")
    y_values = sorted(int(v) for v in sub["Y"].unique())
    rows: list[list[object]] = []
    for filt in _ordered_values(sub["filter"].astype(str).unique(), PRINCIPAL_FILTER_ORDER):
        sf = sub[sub["filter"].astype(str) == filt].set_index("Y")
        vals = [float(sf.loc[y, "delta_loglik_model_minus_baseline"]) if y in sf.index else float("nan") for y in y_values]
        finite = np.asarray([v for v in vals if math.isfinite(v)], dtype=float)
        if finite.size == 0:
            continue
        rows.append([
            STRATUM_TEX.get(filt, filt),
            *[fmt_gain(v, 5) if math.isfinite(v) else "--" for v in vals],
            fmt_gain(finite.min(), 5),
            fmt_count_ratio(int((finite > 0).sum()), len(finite)),
        ])
    text = render_tabular(
        headers=["Stratum", *[raw_tex(rf"$Y={y}$") for y in y_values], "Minimum", raw_tex(r"Positive $Y$")],
        rows=rows,
        align="l" + "r" * (len(y_values) + 2),
        sources=[src],
    )
    return _write_table(output_dir, "table_y_sweep", text), [src]


def build_memory_irreducibility(outputs_root: Path, output_dir: Path) -> tuple[Path, list[Path]]:
    src = SourceSpec("A02", "chl2_memory_irreducibility.csv").resolve(outputs_root)
    df = read_csv_checked(
        src,
        ["filter", "order1_model", "baseline", "n_events", "delta_loglik_per_event_total", "block_z", "n_blocks"],
    )
    sub = df[(df["filter"] == "ALL") & (df["order1_model"] == CHL2)].copy()
    preferred = [
        "noPhi_order0_eta",
        "HL2_order0_eta",
        "HL2_gap_excl_order0_eta",
        "Cramer_order0_exp",
        "Cramer_Granville_order0_exp",
        "Cramer_Granville_gap_excl_order0_exp",
    ]
    rows: list[list[object]] = []
    by_baseline = {str(r["baseline"]): r for _, r in sub.iterrows()}
    for baseline in preferred:
        if baseline not in by_baseline:
            continue
        r = by_baseline[baseline]
        rows.append([
            MODEL_DISPLAY.get(baseline, baseline),
            fmt_gain(r["delta_loglik_per_event_total"], 6),
            fmt_fixed(r["block_z"], 3),
            fmt_integer(r["n_events"]),
            str(int(r["n_blocks"])),
        ])
    text = render_tabular(
        headers=["Order-zero baseline", raw_tex(r"CHL2 gain $\Delta\ell$"), raw_tex(r"Block $z$"), "Events", "Blocks"],
        rows=rows,
        align="lrrrr",
        sources=[src],
    )
    return _write_table(output_dir, "table_memory_irreducibility", text), [src]


def build_scale_wave(outputs_root: Path, output_dir: Path) -> tuple[Path, list[Path]]:
    src = SourceSpec("A05", "chl4d2_scale_wave_summary.csv").resolve(outputs_root)
    df = read_csv_checked(
        src,
        [
            "filter",
            "empirical_events",
            "event_share_vs_ALL",
            "empirical_diag_probability",
            "D3_empirical_gap_population",
            "D3_chl2_gap_population",
            "D3_residual_gap_population",
        ],
    )
    rows: list[list[object]] = []
    for filt in _ordered_values(df["filter"].astype(str).unique(), G2_FILTER_ORDER):
        r = df[df["filter"].astype(str) == filt].iloc[0]
        rows.append([
            G2_FILTER_TEX.get(filt, filt),
            fmt_integer(r["empirical_events"]),
            fmt_share(r["event_share_vs_ALL"], 6),
            fmt_fixed(r["empirical_diag_probability"], 6),
            fmt_fixed(r["D3_empirical_gap_population"], 6),
            fmt_fixed(r["D3_chl2_gap_population"], 6),
            fmt_fixed(r["D3_residual_gap_population"], 6),
        ])
    text = render_tabular(
        headers=[raw_tex(r"$g_2$ filter"), "Events", "Mass share", "Emp. diagonal", raw_tex(r"$D_3^{\rm emp}$"), raw_tex(r"$D_3^{\rm CHL2}$"), "Residual"],
        rows=rows,
        align="lrrrrrr",
        sources=[src],
    )
    return _write_table(output_dir, "table_scale_wave", text), [src]


def build_modular_orientation_lift(outputs_root: Path, output_dir: Path) -> tuple[Path, list[Path]]:
    src = SourceSpec("A08", "chl2_os_old_direct_vs_oriented.csv").resolve(outputs_root)
    df = read_csv_checked(
        src,
        [
            "block",
            "q",
            "emp_diag",
            "old_diag_model",
            "oriented_diag_model",
            "old_kl",
            "oriented_kl",
            "delta_kl_old_minus_oriented",
            "old_model_label",
            "oriented_model_label",
        ],
    )
    sub = df[df["block"].astype(str) == "ALL"].sort_values("q")
    if sub.empty:
        raise TableBuildError("A08 comparison has no block=ALL rows")
    labels = set(sub["old_model_label"].astype(str))
    if labels != {"absolute_prime_os_previous_gap_conditioned"}:
        raise TableBuildError(f"Unexpected A08 old-model labels: {sorted(labels)}")
    rows = [
        [
            str(int(r.q)),
            fmt_fixed(r.emp_diag, 6),
            fmt_fixed(r.old_diag_model, 6),
            fmt_fixed(r.oriented_diag_model, 6),
            fmt_gain(r.old_kl, 6),
            fmt_gain(r.oriented_kl, 6),
            fmt_gain(r.delta_kl_old_minus_oriented, 6),
        ]
        for r in sub.itertuples(index=False)
    ]
    text = render_tabular(
        headers=[raw_tex(r"$q$"), "Emp. diagonal", "Previous-gap control", "Orientation lift", "Control KL", "Lift KL", raw_tex(r"$\Delta\mathrm{KL}$")],
        rows=rows,
        align="rrrrrrr",
        sources=[src],
    )
    return _write_table(output_dir, "table_modular_orientation_lift", text), [src]


def build_chl4_residual(outputs_root: Path, output_dir: Path) -> tuple[Path, list[Path]]:
    q3_src = SourceSpec("A07", "chl4_q3_diagonal_scalar.csv").resolve(outputs_root)
    stab_src = SourceSpec("A07", "chl4_block_stability.csv").resolve(outputs_root)
    q3 = read_csv_checked(
        q3_src,
        [
            "block",
            "D_empirical",
            "D_chl2",
            "D_residual_emp_minus_chl2",
            "diagonal_probability_empirical",
            "diagonal_probability_chl2",
            "wrong_sign_chl2",
        ],
    ).sort_values("block")
    stability = read_csv_checked(stab_src, ["diagnostic", "n_blocks", "mean", "min", "max"])
    q3_summary = stability[stability["diagnostic"] == "q3_D_residual"]
    comment = ""
    if not q3_summary.empty:
        r = q3_summary.iloc[0]
        comment = (
            f"% q3 residual summary: mean={float(r['mean']):.12g}; "
            f"min={float(r['min']):.12g}; max={float(r['max']):.12g}; "
            f"n_blocks={int(r['n_blocks'])}\n"
        )
    rows = [
        [
            str(r.block),
            fmt_fixed(r.D_empirical, 6),
            fmt_fixed(r.D_chl2, 6),
            fmt_fixed(r.D_residual_emp_minus_chl2, 6),
            fmt_fixed(r.diagonal_probability_empirical, 6),
            fmt_fixed(r.diagonal_probability_chl2, 6),
            "yes" if as_bool(r.wrong_sign_chl2) else "no",
        ]
        for r in q3.itertuples(index=False)
    ]
    text = comment + render_tabular(
        headers=["Block", r"$D_3^{\rm emp}$", r"$D_3^{\rm control}$", "Residual", "Emp. diagonal", "Control diagonal", "Wrong sign"],
        rows=rows,
        align="lrrrrrl",
        sources=[q3_src, stab_src],
    )
    return _write_table(output_dir, "table_chl4_q3_residual", text), [q3_src, stab_src]


def build_hrp_models(outputs_root: Path, output_dir: Path) -> tuple[Path, list[Path]]:
    src = SourceSpec("A10", "chl2_factor_model_summary.csv").resolve(outputs_root)
    df = read_csv_checked(
        src,
        ["filter", "model", "H", "R", "P", "n_events", "loglik_per_event", "conditional_kl", "rank_by_loglik"],
    )
    sub = df[df["filter"].astype(str) == "ALL"].sort_values("rank_by_loglik")
    if len(sub) != 8:
        raise TableBuildError(f"A10 ALL cube must contain 8 models, found {len(sub)}")
    rows = [
        [
            str(int(r.rank_by_loglik)),
            MODEL_DISPLAY.get(str(r.model), str(r.model)),
            str(int(r.H)),
            str(int(r.R)),
            str(int(r.P)),
            fmt_fixed(r.loglik_per_event, 9),
            fmt_gain(r.conditional_kl, 6),
        ]
        for r in sub.itertuples(index=False)
    ]
    text = render_tabular(
        headers=["Rank", "Model", r"$H$", r"$R$", r"$P$", "Loglik/event", "Conditional KL"],
        rows=rows,
        align="rlrrrrr",
        sources=[src],
    )
    return _write_table(output_dir, "table_hrp_model_ranking", text), [src]


def build_hrp_effects(outputs_root: Path, output_dir: Path) -> tuple[Path, list[Path]]:
    src = SourceSpec("A10", "chl2_factorial_effects_summary.csv").resolve(outputs_root)
    df = read_csv_checked(
        src,
        [
            "filter",
            "effect",
            "pooled_effect_loglik_per_event",
            "n_blocks",
            "positive_blocks",
            "negative_blocks",
            "all_blocks_same_nonzero_sign",
        ],
    )
    order = ["H", "R", "P", "H:R", "H:P", "R:P", "H:R:P"]
    sub = df[df["filter"].astype(str) == "ALL"].set_index("effect")
    rows: list[list[object]] = []
    for effect in order:
        if effect not in sub.index:
            raise TableBuildError(f"A10 ALL factorial table is missing effect {effect}")
        r = sub.loc[effect]
        rows.append([
            rf"${effect.replace(':', r'\!:\!')}$",
            fmt_gain(r["pooled_effect_loglik_per_event"], 6),
            fmt_count_ratio(r["positive_blocks"], r["n_blocks"]),
            fmt_count_ratio(r["negative_blocks"], r["n_blocks"]),
            "yes" if as_bool(r["all_blocks_same_nonzero_sign"]) else "no",
        ])
    text = render_tabular(
        headers=["Effect", "Factorial effect (nats/event)", "Positive blocks", "Negative blocks", "Stable sign"],
        rows=rows,
        align="lrrrr",
        sources=[src],
    )
    return _write_table(output_dir, "table_hrp_factorial_effects", text), [src]


def build_hrp_context(outputs_root: Path, output_dir: Path) -> tuple[Path, list[Path]]:
    src = SourceSpec("A10", "chl2_factor_context_effects_summary.csv").resolve(outputs_root)
    df = read_csv_checked(
        src,
        [
            "filter",
            "factor",
            "context",
            "model_low",
            "model_high",
            "pooled_delta_loglik_per_event",
            "n_blocks",
            "positive_blocks",
            "negative_blocks",
        ],
    )
    wanted = [
        ("P", "H=0;R=1"),
        ("R", "H=0;P=1"),
        ("H", "R=1;P=0"),
        ("H", "R=1;P=1"),
    ]
    sub = df[df["filter"].astype(str) == "ALL"]
    rows: list[list[object]] = []
    for factor, context in wanted:
        hit = sub[(sub["factor"] == factor) & (sub["context"] == context)]
        if hit.empty:
            raise TableBuildError(f"A10 context table is missing {factor} | {context}")
        r = hit.iloc[0]
        rows.append([
            rf"${factor}\mid({context.replace(';', ',')})$",
            raw_tex(f"{tex_escape(MODEL_DISPLAY.get(str(r['model_low']), str(r['model_low'])))} $\\to$ {tex_escape(MODEL_DISPLAY.get(str(r['model_high']), str(r['model_high'])))}"),
            fmt_gain(r["pooled_delta_loglik_per_event"], 6),
            fmt_count_ratio(r["positive_blocks"], r["n_blocks"]),
            fmt_count_ratio(r["negative_blocks"], r["n_blocks"]),
        ])
    text = render_tabular(
        headers=["Contextual addition", "Comparison", raw_tex(r"$\Delta\ell$"), "Positive blocks", "Negative blocks"],
        rows=rows,
        align="llrrr",
        sources=[src],
    )
    return _write_table(output_dir, "table_hrp_context_effects", text), [src]


BUILDERS: dict[str, Callable[[Path, Path], tuple[Path, list[Path]]]] = {
    "main_likelihood": build_main_likelihood,
    "block_stability": build_block_stability,
    "y_sweep": build_y_sweep,
    "memory_irreducibility": build_memory_irreducibility,
    "scale_wave": build_scale_wave,
    "modular_orientation_lift": build_modular_orientation_lift,
    "chl4_residual": build_chl4_residual,
    "hrp_models": build_hrp_models,
    "hrp_effects": build_hrp_effects,
    "hrp_context": build_hrp_context,
}


def parse_only(text: str | None) -> list[str]:
    if not text:
        return list(BUILDERS)
    requested = [item.strip() for item in text.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(BUILDERS))
    if unknown:
        raise ValueError(f"Unknown table builder(s): {unknown}; choices={sorted(BUILDERS)}")
    return requested


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", default="outputs", help="Repository outputs directory")
    parser.add_argument(
        "--output-dir",
        default="outputs/v2_release_paper_assets/tables",
        help="Generated LaTeX fragment directory",
    )
    parser.add_argument("--only", default=None, help=f"Comma-separated subset: {','.join(BUILDERS)}")
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if any selected audit output is missing or has an incompatible schema",
    )
    args = parser.parse_args(argv)

    outputs_root = Path(args.outputs_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = parse_only(args.only)

    produced: list[Path] = []
    all_sources: dict[str, Path] = {}
    skipped: list[dict[str, str]] = []
    for key in selected:
        try:
            output, sources = BUILDERS[key](outputs_root, output_dir)
        except (FileNotFoundError, ValueError, TableBuildError) as exc:
            if args.strict:
                raise
            skipped.append({"table": key, "reason": str(exc)})
            print(f"[tables] SKIP {key}: {exc}", file=sys.stderr)
            continue
        produced.append(output)
        for source in sources:
            all_sources[source.as_posix()] = source
        print(f"[tables] wrote {output}")

    manifest_path = output_dir / "tables_manifest.json"
    manifest = {
        "schema": "chl2-paper-tables@1",
        "producer": portable_path(Path(__file__)),
        "producer_sha256": sha256_file(Path(__file__)),
        "outputs_root": portable_path(outputs_root),
        "selected": selected,
        "strict": bool(args.strict),
        "sources": [source_record(p) for p in sorted(all_sources.values(), key=lambda x: x.as_posix())],
        "outputs": [output_record(p) for p in sorted(produced, key=lambda x: x.as_posix())],
        "skipped": skipped,
    }
    write_json(manifest_path, manifest)
    print(f"[tables] wrote {manifest_path}")
    if args.strict and len(produced) != len(selected):  # defensive
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
