#!/usr/bin/env python3
"""
CHL4-D3: Orientation Lift Audit for modulo-3 prime-residue transitions.

This audit tests whether the apparent modulo-3 failure of the CHL2-induced
Oliver--Soundararajan diagnostic comes from the gap-population model itself, or
from the way a marginal gap-residue population is lifted to a row-wise prime
residue transition matrix.

For q=3, reduced residues are {1,2}. Let

    p_j = P(g mod 3 = j),  j = 0,1,2.

The oriented lift is

    T(1,1) ∝ p0/2,   T(1,2) ∝ p1,
    T(2,2) ∝ p0/2,   T(2,1) ∝ p2,

with row-wise normalization. The factor p0/2 is essential: the diagonal branch
is shared between the two previous-prime rows.

A deliberately invalid naive lift is also reported:

    T(1,1) ∝ p0,     T(1,2) ∝ p1,
    T(2,2) ∝ p0,     T(2,1) ∝ p2,

which tends to manufacture diagonal persistence.

Inputs
------
- gap-residue CSV from CHL4-D2: columns containing block, filter, gap_mod3,
  empirical_share and model_share (or empirical_count/model_expected).
- optional empirical and CHL2 direct transfer matrices from CHL4-A.

Outputs
-------
- chl4d3_orientation_lift_matrices.csv
- chl4d3_orientation_lift_summary.csv
- chl4d3_direct_vs_orientation_lift.csv
- chl4d3_orientation_defect_summary.csv
- chl4d3_interpretacion.md
- chl4d3_runtime_telemetry.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def d3_from_probs(t11: float, t12: float, t21: float, t22: float, eps: float = 1e-15) -> float:
    return float(math.log(((t11 + eps) * (t22 + eps)) / ((t12 + eps) * (t21 + eps))))


def normalize_pair(x: float, y: float, eps: float = 1e-15) -> Tuple[float, float]:
    z = x + y
    if z <= eps:
        return 0.5, 0.5
    return float(x / z), float(y / z)


def weighted_diag_from_matrix_rows(rows: Sequence[Mapping[str, float]], prob_key: str, row_count_key: Optional[str] = None) -> float:
    num = 0.0
    den = 0.0
    for r in rows:
        b = int(r["from_residue_b"])
        a = int(r["to_residue_a"])
        p = float(r[prob_key])
        w = float(r.get(row_count_key, 1.0)) if row_count_key else 1.0
        if a == b:
            num += w * p
        den += w * p
    if den <= 0:
        return float("nan")
    return float(num / den)


def q3_orientation_matrix_from_gap_shares(p0: float, p1: float, p2: float, kind: str = "orientation") -> Dict[Tuple[int, int], float]:
    """Build q=3 row-wise transition matrix from gap residue shares.

    kind="orientation" uses p0/2 on each diagonal row.
    kind="naive_invalid" uses p0 on each diagonal row, intentionally wrong.
    """
    if kind not in {"orientation", "naive_invalid"}:
        raise ValueError(f"unknown lift kind {kind}")
    diag_mass = p0 / 2.0 if kind == "orientation" else p0
    t11, t12 = normalize_pair(diag_mass, p1)
    t22, t21 = normalize_pair(diag_mass, p2)
    return {
        (1, 1): t11,
        (1, 2): t12,
        (2, 1): t21,
        (2, 2): t22,
    }


def matrix_summary(matrix: Mapping[Tuple[int, int], float], row_weights: Optional[Mapping[int, float]] = None) -> Dict[str, float]:
    t11 = float(matrix.get((1, 1), 0.0))
    t12 = float(matrix.get((1, 2), 0.0))
    t21 = float(matrix.get((2, 1), 0.0))
    t22 = float(matrix.get((2, 2), 0.0))
    d3 = d3_from_probs(t11, t12, t21, t22)
    if row_weights:
        w1 = float(row_weights.get(1, 1.0))
        w2 = float(row_weights.get(2, 1.0))
        diag = (w1 * t11 + w2 * t22) / max(w1 + w2, 1e-15)
    else:
        diag = 0.5 * (t11 + t22)
    return {
        "T11": t11,
        "T12": t12,
        "T21": t21,
        "T22": t22,
        "D3": d3,
        "diagonal_probability": float(diag),
    }


def load_gap_population(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"block", "filter", "gap_mod3"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"gap residue CSV missing columns: {missing}")
    if "empirical_share" not in df.columns:
        if "empirical_count" not in df.columns:
            raise ValueError("need empirical_share or empirical_count")
        df["empirical_share"] = df.groupby(["block", "filter"])["empirical_count"].transform(lambda x: x / x.sum())
    if "model_share" not in df.columns:
        if "model_expected" not in df.columns:
            raise ValueError("need model_share or model_expected")
        df["model_share"] = df.groupby(["block", "filter"])["model_expected"].transform(lambda x: x / x.sum())
    df["gap_mod3"] = df["gap_mod3"].astype(int)
    return df


def make_orientation_outputs(gap_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    matrix_rows: List[dict] = []
    summary_rows: List[dict] = []
    for (block, filt), g in gap_df.groupby(["block", "filter"], sort=True):
        shares = {}
        for source, share_col in [("empirical", "empirical_share"), ("model", "model_share")]:
            p = {int(r.gap_mod3): float(getattr(r, share_col)) for r in g.itertuples()}
            p0, p1, p2 = p.get(0, 0.0), p.get(1, 0.0), p.get(2, 0.0)
            # Some rows may have very small rounding drift.
            s = p0 + p1 + p2
            if s > 0:
                p0, p1, p2 = p0 / s, p1 / s, p2 / s
            for lift_kind in ["orientation", "naive_invalid"]:
                mat = q3_orientation_matrix_from_gap_shares(p0, p1, p2, lift_kind)
                # Natural row masses induced by the unnormalized lift.
                if lift_kind == "orientation":
                    row_weights = {1: p0 / 2.0 + p1, 2: p0 / 2.0 + p2}
                else:
                    row_weights = {1: p0 + p1, 2: p0 + p2}
                summ = matrix_summary(mat, row_weights=row_weights)
                summary_rows.append({
                    "block": block,
                    "filter": filt,
                    "source": source,
                    "lift_kind": lift_kind,
                    "p0_gap_mod3_diag": p0,
                    "p1_gap_mod3_branch_1": p1,
                    "p2_gap_mod3_branch_2": p2,
                    **summ,
                })
                for (b, a), prob in sorted(mat.items()):
                    matrix_rows.append({
                        "block": block,
                        "filter": filt,
                        "source": source,
                        "lift_kind": lift_kind,
                        "from_residue_b": b,
                        "to_residue_a": a,
                        "probability": prob,
                        "row_weight": row_weights[b],
                    })
    return pd.DataFrame(matrix_rows), pd.DataFrame(summary_rows)


def normalize_direct_empirical_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    rename = {}
    if "from_residue_b_prev_prime" in d.columns:
        rename["from_residue_b_prev_prime"] = "from_residue_b"
    if "to_residue_a_current_prime" in d.columns:
        rename["to_residue_a_current_prime"] = "to_residue_a"
    if "empirical_prob" in d.columns:
        rename["empirical_prob"] = "empirical_probability"
    d = d.rename(columns=rename)
    if "filter" not in d.columns:
        d["filter"] = "ALL"
    if "block" not in d.columns:
        d["block"] = "ALL"
    return d


def normalize_direct_model_columns(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    d = df.copy()
    rename = {}
    if "from_residue_b_prev_prime" in d.columns:
        rename["from_residue_b_prev_prime"] = "from_residue_b"
    if "to_residue_a_current_prime" in d.columns:
        rename["to_residue_a_current_prime"] = "to_residue_a"
    if "model_prob" in d.columns:
        rename["model_prob"] = "model_probability"
    d = d.rename(columns=rename)
    if "model" in d.columns:
        d = d[d["model"] == model_name].copy()
    if "filter" not in d.columns:
        d["filter"] = "ALL"
    if "block" not in d.columns:
        d["block"] = "ALL"
    return d


def direct_matrix_summary(df: pd.DataFrame, source: str, prob_col: str) -> pd.DataFrame:
    rows: List[dict] = []
    d = df[df.get("q", 3) == 3].copy() if "q" in df.columns else df.copy()
    for (block, filt), g in d.groupby(["block", "filter"], sort=True):
        mat = {}
        row_weights = {}
        for r in g.itertuples(index=False):
            b = int(getattr(r, "from_residue_b"))
            a = int(getattr(r, "to_residue_a"))
            mat[(b, a)] = float(getattr(r, prob_col))
            if hasattr(r, "row_count"):
                row_weights[b] = float(getattr(r, "row_count"))
        if all(k in mat for k in [(1, 1), (1, 2), (2, 1), (2, 2)]):
            summ = matrix_summary(mat, row_weights=row_weights if row_weights else None)
            rows.append({
                "block": block,
                "filter": filt,
                "source": source,
                "direct_kind": "direct_os_matrix",
                **summ,
            })
    return pd.DataFrame(rows)


def compare_direct_vs_lift(direct_summary: pd.DataFrame, lift_summary: pd.DataFrame, lift_kind: str = "orientation") -> pd.DataFrame:
    if direct_summary.empty:
        return pd.DataFrame()
    lift = lift_summary[lift_summary["lift_kind"] == lift_kind].copy()
    rows: List[dict] = []
    for r in direct_summary.itertuples(index=False):
        m = lift[(lift["block"] == r.block) & (lift["filter"] == r.filter) & (lift["source"] == r.source)]
        if m.empty and r.filter == "ALL":
            # fallback: if no exact ALL for aggregate only
            m = lift[(lift["block"] == r.block) & (lift["filter"] == "ALL") & (lift["source"] == r.source)]
        if m.empty:
            continue
        lr = m.iloc[0]
        rows.append({
            "block": r.block,
            "filter": r.filter,
            "source": r.source,
            "lift_kind": lift_kind,
            "D3_direct": r.D3,
            "D3_lift": float(lr["D3"]),
            "delta_D3_direct_minus_lift": r.D3 - float(lr["D3"]),
            "diagonal_direct": r.diagonal_probability,
            "diagonal_lift": float(lr["diagonal_probability"]),
            "delta_diagonal_direct_minus_lift": r.diagonal_probability - float(lr["diagonal_probability"]),
        })
    return pd.DataFrame(rows)


def summarize_defects(comparison: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame()
    rows = []
    for (source, filt, lift_kind), g in comparison.groupby(["source", "filter", "lift_kind"], sort=True):
        rows.append({
            "source": source,
            "filter": filt,
            "lift_kind": lift_kind,
            "n_blocks": int(g["block"].nunique()),
            "mean_delta_D3_direct_minus_lift": float(g["delta_D3_direct_minus_lift"].mean()),
            "std_delta_D3_direct_minus_lift": float(g["delta_D3_direct_minus_lift"].std(ddof=1)) if len(g) > 1 else 0.0,
            "mean_delta_diagonal_direct_minus_lift": float(g["delta_diagonal_direct_minus_lift"].mean()),
            "std_delta_diagonal_direct_minus_lift": float(g["delta_diagonal_direct_minus_lift"].std(ddof=1)) if len(g) > 1 else 0.0,
        })
    return pd.DataFrame(rows)


def write_interpretation(outdir: Path, lift_summary: pd.DataFrame, comparison: pd.DataFrame, defects: pd.DataFrame) -> None:
    """Write a Markdown interpretation file.

    Important: the headline comparison must use only the oriented lift.
    The audit also computes a deliberately invalid naive lift as a negative
    control; mixing both lift kinds in the same headline table gives an
    apparently contradictory result.
    """
    lines = []
    lines.append("# CHL4-D3 Orientation Lift Audit")
    lines.append("")
    lines.append("This audit tests whether the modulo $3$ Oliver--Soundararajan discrepancy comes from the gap population itself or from the row-wise lift from gap residues to prime-residue transition matrices.")
    lines.append("")
    lines.append("For $q=3$, the oriented lift is:")
    lines.append("")
    lines.append("$$T(1,1) \\propto p_0/2,\\quad T(1,2) \\propto p_1,$$")
    lines.append("$$T(2,2) \\propto p_0/2,\\quad T(2,1) \\propto p_2,$$")
    lines.append("")
    lines.append("where $p_j=P(g\\equiv j \\pmod 3)$. The factor $p_0/2$ is the key orientation correction: the diagonal branch is shared across the two previous-prime residue rows.")
    lines.append("")

    if not comparison.empty:
        oriented_all = comparison[(comparison["filter"] == "ALL") & (comparison["lift_kind"] == "orientation")].copy()
        if not oriented_all.empty:
            lines.append("## Direct OS matrix versus oriented lift, ALL")
            lines.append("")
            keep = ["block", "source", "D3_direct", "D3_lift", "delta_D3_direct_minus_lift", "diagonal_direct", "diagonal_lift"]
            # Prefer a true aggregate ALL row if present. Otherwise report block means.
            agg = oriented_all[oriented_all["block"] == "ALL"].copy()
            if agg.empty:
                agg = oriented_all.groupby("source", as_index=False).agg({
                    "D3_direct": "mean",
                    "D3_lift": "mean",
                    "delta_D3_direct_minus_lift": "mean",
                    "diagonal_direct": "mean",
                    "diagonal_lift": "mean",
                })
                agg.insert(0, "block", "B01-B10 mean")
            # Normalize source labels for readability without changing CSVs.
            agg = agg[keep].copy()
            agg["source"] = agg["source"].replace({
                "empirical": "empirical_direct_transfer",
                "model": "chl2_direct_transfer",
            })
            lines.append(agg.to_markdown(index=False))
            lines.append("")

        naive_all = comparison[(comparison["filter"] == "ALL") & (comparison["lift_kind"] == "naive_invalid")].copy()
        if not naive_all.empty:
            lines.append("## Negative control: invalid naive lift, ALL")
            lines.append("")
            lines.append("The naive lift is intentionally wrong: it assigns the full $p_0$ diagonal mass to each row instead of sharing it as $p_0/2$. It is included only to show how a false diagonal persistence can be manufactured.")
            lines.append("")
            keep_naive = ["block", "source", "D3_lift", "diagonal_lift"]
            naive = naive_all[naive_all["block"] == "ALL"].copy()
            if naive.empty:
                naive = naive_all.groupby("source", as_index=False).agg({
                    "D3_lift": "mean",
                    "diagonal_lift": "mean",
                })
                naive.insert(0, "block", "B01-B10 mean")
            naive = naive[keep_naive].copy()
            naive["source"] = naive["source"].replace({
                "empirical": "empirical_gap_population",
                "model": "chl2_gap_population",
            })
            lines.append(naive.to_markdown(index=False))
            lines.append("")

    if not defects.empty:
        lines.append("## Defect summary")
        lines.append("")
        # Show oriented lift first, then the invalid lift if present.
        d = defects[defects["filter"] == "ALL"].copy()
        if not d.empty:
            d["source"] = d["source"].replace({
                "empirical": "empirical_direct_transfer",
                "model": "chl2_direct_transfer",
            })
            order = {"orientation": 0, "naive_invalid": 1}
            d["_order"] = d["lift_kind"].map(order).fillna(99)
            d = d.sort_values(["_order", "source"]).drop(columns=["_order"])
            lines.append(d.to_markdown(index=False))
            lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("If the empirical oriented lift agrees with the empirical direct matrix but the CHL2 direct OS matrix disagrees with the CHL2 oriented lift, then the failure is not in the marginal population of $g \\bmod 3$. It is in the lifting rule that converts gap-residue mass into a row-wise transition matrix.")
    lines.append("")
    lines.append("This is the expected outcome if CHL2 has already learned the correct modulo $3$ gap population but the diagnostic matrix assigns the off-diagonal branches to rows incorrectly.")
    (outdir / "chl4d3_interpretacion.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="CHL4-D3 orientation lift audit for q=3.")
    ap.add_argument("--gap-residue-csv", required=True, help="CHL4-D2 gap residue population CSV.")
    ap.add_argument("--empirical-matrix-csv", default=None, help="CHL4-A empirical transfer matrices CSV.")
    ap.add_argument("--model-matrix-csv", default=None, help="CHL4-A CHL2 model transfer matrices CSV.")
    ap.add_argument("--model-name", default="CHL2_path_excl_cond_eta")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    gap_df = load_gap_population(Path(args.gap_residue_csv))
    lift_mats, lift_summary = make_orientation_outputs(gap_df)

    direct_summaries = []
    if args.empirical_matrix_csv:
        emp = normalize_direct_empirical_columns(pd.read_csv(args.empirical_matrix_csv))
        direct_summaries.append(direct_matrix_summary(emp, "empirical", "empirical_probability"))
    if args.model_matrix_csv:
        mod = normalize_direct_model_columns(pd.read_csv(args.model_matrix_csv), args.model_name)
        direct_summaries.append(direct_matrix_summary(mod, "model", "model_probability"))
    direct_summary = pd.concat(direct_summaries, ignore_index=True) if direct_summaries else pd.DataFrame()

    comparisons = []
    for kind in ["orientation", "naive_invalid"]:
        comparisons.append(compare_direct_vs_lift(direct_summary, lift_summary, kind))
    comparison = pd.concat([c for c in comparisons if not c.empty], ignore_index=True) if any(not c.empty for c in comparisons) else pd.DataFrame()
    defects = summarize_defects(comparison)

    lift_mats.to_csv(outdir / "chl4d3_orientation_lift_matrices.csv", index=False)
    lift_summary.to_csv(outdir / "chl4d3_orientation_lift_summary.csv", index=False)
    direct_summary.to_csv(outdir / "chl4d3_direct_matrix_summary.csv", index=False)
    comparison.to_csv(outdir / "chl4d3_direct_vs_orientation_lift.csv", index=False)
    defects.to_csv(outdir / "chl4d3_orientation_defect_summary.csv", index=False)
    write_interpretation(outdir, lift_summary, comparison, defects)

    telemetry = {
        "script": "chl4d3_orientation_lift_audit.py",
        "elapsed_seconds": time.perf_counter() - t0,
        "python": sys.version,
        "platform": platform.platform(),
        "argv": sys.argv,
        "gap_residue_csv": str(args.gap_residue_csv),
        "empirical_matrix_csv": str(args.empirical_matrix_csv),
        "model_matrix_csv": str(args.model_matrix_csv),
        "model_name": args.model_name,
        "n_gap_rows": int(len(gap_df)),
        "n_lift_matrix_rows": int(len(lift_mats)),
        "n_lift_summary_rows": int(len(lift_summary)),
        "n_direct_summary_rows": int(len(direct_summary)),
        "n_comparison_rows": int(len(comparison)),
        "n_defect_rows": int(len(defects)),
    }
    (outdir / "chl4d3_runtime_telemetry.json").write_text(json.dumps(telemetry, indent=2), encoding="utf-8")
    config = vars(args).copy()
    config.update({"q": 3, "audit": "CHL4-D3 orientation lift"})
    (outdir / "chl4d3_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"[CHL4-D3] wrote outputs to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
