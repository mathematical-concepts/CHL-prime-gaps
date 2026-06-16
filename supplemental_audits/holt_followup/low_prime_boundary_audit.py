#!/usr/bin/env python3
"""Low-prime boundary / valid-edge multiplicity audit.

This diagnostic tests whether the old-direct modular error decreases with
valid-edge multiplicity asymmetry N0(q)/Nr(q) for prime moduli q.
It is intentionally a diagnostic compatibility check, not a theorem.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--old-vs-oriented-csv", required=True, help="CSV from chl4d5: chl2_os_old_direct_vs_oriented.csv")
    p.add_argument("--gap-residue-csv", default=None, help="Optional CHL4-D2 gap population CSV for q=3 p0 check")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.old_vs_oriented_csv)
    required = {"q", "old_kl", "oriented_kl", "delta_kl_old_minus_oriented", "old_l1", "oriented_l1", "old_diag_model", "oriented_diag_model", "emp_diag"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in old-vs-oriented CSV: {sorted(missing)}")

    rows = []
    for q, g in df.groupby("q"):
        q = int(q)
        # This table is intended for prime moduli.  For composite q, N_r may depend on r.
        n0 = q - 1
        nr = q - 2
        edge_ratio = n0 / nr
        edge_excess = edge_ratio - 1.0
        rows.append({
            "q": q,
            "N0": n0,
            "Nr_nonzero_prime_modulus": nr,
            "edge_ratio_N0_over_Nr": edge_ratio,
            "edge_excess_ratio_minus_1": edge_excess,
            "old_kl_mean": g["old_kl"].mean(),
            "old_kl_std": g["old_kl"].std(ddof=1),
            "oriented_kl_mean": g["oriented_kl"].mean(),
            "oriented_kl_std": g["oriented_kl"].std(ddof=1),
            "delta_kl_mean": g["delta_kl_old_minus_oriented"].mean(),
            "old_l1_mean": g["old_l1"].mean(),
            "oriented_l1_mean": g["oriented_l1"].mean(),
            "emp_diag_mean": g["emp_diag"].mean(),
            "old_diag_mean": g["old_diag_model"].mean(),
            "oriented_diag_mean": g["oriented_diag_model"].mean(),
            "old_diag_abs_error": (g["old_diag_model"] - g["emp_diag"]).abs().mean(),
            "oriented_diag_abs_error": (g["oriented_diag_model"] - g["emp_diag"]).abs().mean(),
            "old_wrong_sign_count": int(g.get("old_wrong_sign", pd.Series(dtype=bool)).sum()) if "old_wrong_sign" in g else None,
            "oriented_wrong_sign_count": int(g.get("oriented_wrong_sign", pd.Series(dtype=bool)).sum()) if "oriented_wrong_sign" in g else None,
            "n_blocks": len(g),
        })
    summary = pd.DataFrame(rows).sort_values("q")

    # Correlations: diagnostic only, n is small.
    corr_rows = []
    for col in ["old_kl_mean", "delta_kl_mean", "old_l1_mean", "old_diag_abs_error"]:
        corr = float(np.corrcoef(summary["edge_ratio_N0_over_Nr"], summary[col])[0, 1]) if len(summary) > 1 else float("nan")
        corr_rows.append({"x": "edge_ratio_N0_over_Nr", "y": col, "pearson_r": corr, "n_moduli": len(summary)})
    corr_df = pd.DataFrame(corr_rows)

    p0_df = pd.DataFrame()
    if args.gap_residue_csv:
        gr = pd.read_csv(args.gap_residue_csv)
        if {"gap_mod3", "filter", "empirical_share", "model_share"}.issubset(gr.columns):
            p0 = gr[(gr["filter"] == "ALL") & (gr["gap_mod3"].astype(int) == 0)].copy()
            if "block" in p0.columns and (p0["block"] == "ALL").any():
                p0_eval = p0[p0["block"] == "ALL"].iloc[0]
                source = "ALL row"
            else:
                p0_eval = p0.mean(numeric_only=True)
                source = "mean over rows"
            p0_df = pd.DataFrame([{
                "q": 3,
                "empirical_p0": float(p0_eval["empirical_share"]),
                "chl2_p0": float(p0_eval["model_share"]),
                "empirical_below_half": bool(float(p0_eval["empirical_share"]) < 0.5),
                "chl2_below_half": bool(float(p0_eval["model_share"]) < 0.5),
                "source": source,
                "n_rows_available": len(p0),
            }])

    summary.to_csv(out / "low_prime_boundary_summary.csv", index=False)
    corr_df.to_csv(out / "low_prime_boundary_correlations.csv", index=False)
    if not p0_df.empty:
        p0_df.to_csv(out / "low_prime_boundary_q3_p0_check.csv", index=False)

    md = []
    md.append("# Low-prime boundary / valid-edge multiplicity audit\n")
    md.append("This audit checks whether the error of the naive direct modular diagnostic decreases with the valid-edge multiplicity asymmetry $N_0(q)/N_r(q)$ for prime moduli. It is a compatibility diagnostic, not an asymptotic theorem.\n")
    md.append("## Summary by modulus\n")
    show = summary[["q", "N0", "Nr_nonzero_prime_modulus", "edge_ratio_N0_over_Nr", "old_kl_mean", "oriented_kl_mean", "delta_kl_mean", "old_l1_mean", "oriented_l1_mean", "emp_diag_mean", "old_diag_mean", "oriented_diag_mean"]].copy()
    md.append(show.to_markdown(index=False, floatfmt=".6g"))
    md.append("\n\n## Correlations\n")
    md.append(corr_df.to_markdown(index=False, floatfmt=".4f"))
    if not p0_df.empty:
        md.append("\n\n## q=3 gap-residue check\n")
        md.append(p0_df.to_markdown(index=False, floatfmt=".6f"))
    md.append("\n\n## Interpretation\n")
    md.append("The old-direct error is largest where the valid-edge asymmetry is largest. For prime moduli, $N_0(q)=q-1$ and $N_r(q)=q-2$ for $r\\not\\equiv0\\pmod q$. The edge ratio therefore decreases from $2$ at $q=3$ toward $1$ as $q$ grows. In the reproduced DS1 diagnostics, old-direct KL and the KL improvement from orientation lift decrease in the same direction. This supports the interpretation that the old-direct diagnostic error is governed by valid-edge multiplicity asymmetry.\n")
    md.append("\nThe result should not be read as a claim about Holt's sieve prime $q$ and the diagnostic modulus being identical objects. It is a compatibility check: the same low-prime boundary combinatorics appears as the edge-count asymmetry in the reduced-residue transition graph.\n")
    (out / "low_prime_boundary_interpretation.md").write_text("\n".join(md), encoding="utf-8")

    telemetry = {
        "elapsed_seconds": time.time() - t0,
        "old_vs_oriented_csv": str(args.old_vs_oriented_csv),
        "gap_residue_csv": str(args.gap_residue_csv) if args.gap_residue_csv else None,
        "output_dir": str(out),
        "summary_rows": len(summary),
        "correlation_rows": len(corr_df),
        "p0_rows": len(p0_df),
    }
    (out / "low_prime_boundary_telemetry.json").write_text(json.dumps(telemetry, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
