#!/usr/bin/env python3
"""
HOLT_GAP_POPULATION_BASELINE_AUDIT
==================================

Supplemental audit for the CHL-prime-gaps project.

This script is intended for the research/holt-followup-audits branch, not for
reproducing the main CHL2 v1.8 whitepaper.  It explores a bridge between:

  * Holt-style gap-population / survival-interval thinking, and
  * CHL2's local no-interior path survival factor Omega_path.

It does NOT implement Fred Holt's exact cycle-recursion model for G(p#).  That
requires the exact recurrence/population machinery from Holt's work.  Instead,
this script performs a reproducible compatibility audit on an existing prime
stream:

  1. Partition the chronological prime stream by Holt survival intervals
     Delta H(s) = [s^2, nextprime(s)^2], where s is a sieving prime.
  2. Measure empirical gap-residue populations inside those intervals.
  3. Compare those populations with the valid-edge asymptotic reference
     p_r proportional to N_r(q), where N_r(q) is the number of valid reduced
     residue edges b -> b+r mod q.
  4. If a CHL2 path-cache is provided, summarize Omega_path and exp(-Omega_path)
     by survival interval and residue class.

The audit answers a narrow question: how do the empirical DS1 gap populations
and CHL2 local survival intensities look when grouped by Holt's survival
intervals?  It is a bridge audit, not a proof of Holt's model and not a CHL2
paper result.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import platform
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Basic arithmetic utilities
# ---------------------------------------------------------------------------


def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def reduced_residues(q: int) -> list[int]:
    return [a for a in range(q) if gcd(a, q) == 1]


def valid_edge_count(q: int, r: int) -> int:
    residues = set(reduced_residues(q))
    return sum(1 for b in residues if (b + r) % q in residues)


def edge_count_reference_distribution(q: int) -> dict[int, float]:
    counts = {r: valid_edge_count(q, r) for r in range(q)}
    total = float(sum(counts.values()))
    if total <= 0:
        return {r: 0.0 for r in range(q)}
    return {r: counts[r] / total for r in range(q)}


def sieve_primes_upto(n: int) -> list[int]:
    if n < 2:
        return []
    a = bytearray(b"\x01") * (n + 1)
    a[0:2] = b"\x00\x00"
    lim = int(n**0.5)
    for p in range(2, lim + 1):
        if a[p]:
            start = p * p
            step = p
            a[start:n + 1:step] = b"\x00" * (((n - start) // step) + 1)
    return [i for i, is_p in enumerate(a) if is_p]


def primes_around_sqrt(x_min: int, x_max: int, margin: int = 10000) -> list[int]:
    lo = max(2, int(math.isqrt(x_min)) - margin)
    hi = int(math.isqrt(x_max)) + margin
    primes = sieve_primes_upto(hi)
    return [p for p in primes if p >= lo]


def assign_survival_stage(x: np.ndarray, stage_primes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assign each x to [s^2, next_s^2) for consecutive stage primes s,next_s."""
    squares = stage_primes.astype(object) ** 2
    # Convert to int64 if safe. DS1 near 1e11 is safe.
    squares = np.array(squares, dtype=np.int64)
    idx = np.searchsorted(squares, x, side="right") - 1
    idx = np.clip(idx, 0, len(stage_primes) - 2)
    s = stage_primes[idx]
    snext = stage_primes[idx + 1]
    start = s.astype(np.int64) * s.astype(np.int64)
    end = snext.astype(np.int64) * snext.astype(np.int64)
    return s, snext, start, end


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------


def infer_prime_column(columns: Iterable[str]) -> str:
    candidates = [
        "prime", "p", "n", "value", "Prime", "P",
    ]
    cols = list(columns)
    for c in candidates:
        if c in cols:
            return c
    # Fall back to the first column.
    return cols[0]


def iter_prime_chunks(prime_csv: Path, chunksize: int) -> Iterator[np.ndarray]:
    reader = pd.read_csv(prime_csv, chunksize=chunksize)
    col: str | None = None
    for chunk in reader:
        if col is None:
            col = infer_prime_column(chunk.columns)
        arr = pd.to_numeric(chunk[col], errors="coerce").dropna().astype(np.int64).to_numpy()
        if len(arr):
            yield arr


def detect_path_cache_columns(df: pd.DataFrame) -> tuple[str, str, str | None, str | None]:
    """Detect gap and path-survival columns in a CHL2 path cache.

    The stable CHL2 cache written by ``chl2_consecutive_exclusion_audit.py``
    uses the column names ``omega_path_exclusion`` and
    ``logE_path_exclusion``.  Earlier versions of this supplemental Holt audit
    only looked for generic names such as ``omega_path``; as a result it could
    silently report all ``Omega_path`` summaries as missing.  This detector
    accepts both naming conventions.
    """
    g1_candidates = ["g1", "g_prev", "prev_gap"]
    g2_candidates = ["g2", "gap", "g_next", "next_gap"]
    omega_candidates = [
        "omega_path", "Omega_path", "omega", "Omega", "omega_chl2_path",
        "path_omega", "OmegaY_path", "omega_y_path", "omega_path_exclusion",
        "Omega_path_exclusion",
    ]
    loge_candidates = [
        "logE_path_exclusion", "logE_path", "log_survival_path",
        "logE", "log_e_path",
    ]

    def find(cands: list[str]) -> str | None:
        for c in cands:
            if c in df.columns:
                return c
        return None

    g1 = find(g1_candidates)
    g2 = find(g2_candidates)
    omega = find(omega_candidates)
    loge = find(loge_candidates)
    if g1 is None or g2 is None:
        raise ValueError(
            "Could not detect g1/g2 columns in path cache. "
            f"Columns are: {list(df.columns)}"
        )
    if omega is None and loge is None:
        raise ValueError(
            "Could not detect an Omega/log-survival column in path cache. "
            "Expected one of omega_path, omega_path_exclusion, logE_path_exclusion, ... "
            f"Columns are: {list(df.columns)}"
        )
    return g1, g2, omega, loge


def load_path_cache(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    df = pd.read_csv(path)
    g1_col, g2_col, omega_col, loge_col = detect_path_cache_columns(df)
    cols = [g1_col, g2_col]
    if omega_col is not None:
        cols.append(omega_col)
    if loge_col is not None and loge_col not in cols:
        cols.append(loge_col)
    out = df[cols].copy()
    rename = {g1_col: "g1", g2_col: "g2"}
    if omega_col is not None:
        rename[omega_col] = "omega_path"
    if loge_col is not None:
        rename[loge_col] = "logE_path"
    out = out.rename(columns=rename)
    if "omega_path" not in out.columns:
        # CHL2 stores logE=-Omega in some caches; recover Omega.
        out["omega_path"] = -pd.to_numeric(out["logE_path"], errors="coerce")
    else:
        out["omega_path"] = pd.to_numeric(out["omega_path"], errors="coerce")
    out["g1"] = out["g1"].astype(int)
    out["g2"] = out["g2"].astype(int)
    out = out[["g1", "g2", "omega_path"]].drop_duplicates(["g1", "g2"])
    if out["omega_path"].isna().all():
        raise ValueError("Path cache was loaded, but all omega_path values are NaN.")
    return out


# ---------------------------------------------------------------------------
# Aggregation state
# ---------------------------------------------------------------------------


@dataclass
class RunningState:
    interval: dict[tuple[int, int], dict[str, float]]
    residue: dict[tuple[int, int, int], float]
    residue_omega: dict[tuple[int, int, int], dict[str, float]]
    all_residue: dict[tuple[int, int], float]
    all_residue_omega: dict[tuple[int, int], dict[str, float]]
    missing_omega: int = 0
    triples_processed: int = 0


def new_state() -> RunningState:
    return RunningState(
        interval=defaultdict(lambda: defaultdict(float)),
        residue=defaultdict(float),
        residue_omega=defaultdict(lambda: defaultdict(float)),
        all_residue=defaultdict(float),
        all_residue_omega=defaultdict(lambda: defaultdict(float)),
    )


def aggregate_chunk(
    primes: np.ndarray,
    carry: np.ndarray,
    stage_primes: np.ndarray,
    mods: list[int],
    gmax: int,
    path_cache: pd.DataFrame | None,
    state: RunningState,
) -> np.ndarray:
    if len(carry):
        primes = np.concatenate([carry, primes])
    if len(primes) < 3:
        return primes[-2:]

    p0 = primes[:-2]
    p1 = primes[1:-1]
    p2 = primes[2:]
    g1 = p1 - p0
    g2 = p2 - p1
    x = p1

    # Keep only finite support for compatibility with DS1 gap support.
    mask = (g1 > 0) & (g2 > 0) & (g1 <= gmax) & (g2 <= gmax)
    if not np.any(mask):
        return primes[-2:]

    g1 = g1[mask].astype(np.int64)
    g2 = g2[mask].astype(np.int64)
    x = x[mask].astype(np.int64)
    s, snext, interval_start, interval_end = assign_survival_stage(x, stage_primes)

    df = pd.DataFrame({
        "stage_prime": s,
        "next_stage_prime": snext,
        "interval_start": interval_start,
        "interval_end": interval_end,
        "g1": g1,
        "g2": g2,
    })

    if path_cache is not None:
        df = df.merge(path_cache, on=["g1", "g2"], how="left")
        missing = int(df["omega_path"].isna().sum())
        state.missing_omega += missing
        # Keep NaN for aggregation; we count non-missing explicitly.
    else:
        df["omega_path"] = np.nan

    # Aggregate interval summaries.
    grouped = df.groupby(["stage_prime", "next_stage_prime", "interval_start", "interval_end"], sort=False)
    for (sp, ns, istart, iend), sub in grouped:
        key = (int(sp), int(ns))
        rec = state.interval[key]
        rec["stage_prime"] = int(sp)
        rec["next_stage_prime"] = int(ns)
        rec["interval_start"] = int(istart)
        rec["interval_end"] = int(iend)
        rec["interval_width"] = int(iend) - int(istart)
        n = len(sub)
        rec["pair_events"] += n
        rec["sum_g2"] += float(sub["g2"].sum())
        rec["sum_g1"] += float(sub["g1"].sum())
        if sub["omega_path"].notna().any():
            omega = sub["omega_path"].dropna().astype(float)
            rec["omega_count"] += len(omega)
            rec["sum_omega_path"] += float(omega.sum())
            rec["sum_exp_neg_omega_path"] += float(np.exp(-omega).sum())

    # Residue populations by interval and overall.
    for q in mods:
        residues = (df["g2"].to_numpy(dtype=np.int64) % q).astype(int)
        tmp = df[["stage_prime", "next_stage_prime", "g2", "omega_path"]].copy()
        tmp["residue"] = residues
        counts = tmp.groupby(["stage_prime", "next_stage_prime", "residue"]).size()
        for (sp, ns, r), n in counts.items():
            state.residue[(int(sp), int(ns), q, int(r))] += float(n)
            state.all_residue[(q, int(r))] += float(n)

        if tmp["omega_path"].notna().any():
            for (sp, ns, r), sub in tmp.groupby(["stage_prime", "next_stage_prime", "residue"], sort=False):
                omega = sub["omega_path"].dropna().astype(float)
                if len(omega):
                    key = (int(sp), int(ns), q, int(r))
                    rec = state.residue_omega[key]
                    rec["omega_count"] += len(omega)
                    rec["sum_omega_path"] += float(omega.sum())
                    rec["sum_exp_neg_omega_path"] += float(np.exp(-omega).sum())
                    rec_all = state.all_residue_omega[(q, int(r))]
                    rec_all["omega_count"] += len(omega)
                    rec_all["sum_omega_path"] += float(omega.sum())
                    rec_all["sum_exp_neg_omega_path"] += float(np.exp(-omega).sum())

    state.triples_processed += len(df)
    return primes[-2:]


# ---------------------------------------------------------------------------
# Output builders
# ---------------------------------------------------------------------------


def build_interval_summary(state: RunningState) -> pd.DataFrame:
    rows = []
    for rec in state.interval.values():
        n = rec.get("pair_events", 0.0)
        omega_n = rec.get("omega_count", 0.0)
        rows.append({
            "stage_prime": int(rec["stage_prime"]),
            "next_stage_prime": int(rec["next_stage_prime"]),
            "interval_start": int(rec["interval_start"]),
            "interval_end": int(rec["interval_end"]),
            "interval_width": int(rec["interval_width"]),
            "pair_events": int(n),
            "mean_g1": rec.get("sum_g1", np.nan) / n if n else np.nan,
            "mean_g2": rec.get("sum_g2", np.nan) / n if n else np.nan,
            "omega_count": int(omega_n),
            "mean_omega_path": rec.get("sum_omega_path", np.nan) / omega_n if omega_n else np.nan,
            "mean_exp_neg_omega_path": rec.get("sum_exp_neg_omega_path", np.nan) / omega_n if omega_n else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["stage_prime", "next_stage_prime"]).reset_index(drop=True)


def build_residue_by_interval(state: RunningState, mods: list[int]) -> pd.DataFrame:
    rows = []
    # total per interval/q
    totals = defaultdict(float)
    for (sp, ns, q, r), n in state.residue.items():
        totals[(sp, ns, q)] += n

    edge_refs = {q: edge_count_reference_distribution(q) for q in mods}
    for (sp, ns, q, r), n in state.residue.items():
        total = totals[(sp, ns, q)]
        ref = edge_refs[q].get(r, 0.0)
        omega_rec = state.residue_omega.get((sp, ns, q, r), {})
        omega_n = omega_rec.get("omega_count", 0.0)
        rows.append({
            "stage_prime": sp,
            "next_stage_prime": ns,
            "q": q,
            "gap_residue_r": r,
            "count": int(n),
            "share": n / total if total else np.nan,
            "valid_edge_reference_share": ref,
            "share_minus_reference": n / total - ref if total else np.nan,
            "omega_count": int(omega_n),
            "mean_omega_path": omega_rec.get("sum_omega_path", np.nan) / omega_n if omega_n else np.nan,
            "mean_exp_neg_omega_path": omega_rec.get("sum_exp_neg_omega_path", np.nan) / omega_n if omega_n else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["stage_prime", "q", "gap_residue_r"]).reset_index(drop=True)


def build_residue_summary(state: RunningState, mods: list[int]) -> pd.DataFrame:
    rows = []
    totals = defaultdict(float)
    for (q, r), n in state.all_residue.items():
        totals[q] += n
    edge_refs = {q: edge_count_reference_distribution(q) for q in mods}
    for q in mods:
        for r in range(q):
            n = state.all_residue.get((q, r), 0.0)
            total = totals[q]
            ref = edge_refs[q].get(r, 0.0)
            omega_rec = state.all_residue_omega.get((q, r), {})
            omega_n = omega_rec.get("omega_count", 0.0)
            rows.append({
                "q": q,
                "gap_residue_r": r,
                "count": int(n),
                "share": n / total if total else np.nan,
                "valid_edge_reference_share": ref,
                "share_minus_reference": n / total - ref if total else np.nan,
                "omega_count": int(omega_n),
                "mean_omega_path": omega_rec.get("sum_omega_path", np.nan) / omega_n if omega_n else np.nan,
                "mean_exp_neg_omega_path": omega_rec.get("sum_exp_neg_omega_path", np.nan) / omega_n if omega_n else np.nan,
            })
    return pd.DataFrame(rows)


def build_boundary_summary(residue_summary: pd.DataFrame, mods: list[int]) -> pd.DataFrame:
    rows = []
    for q in mods:
        if q < 3:
            continue
        n0 = valid_edge_count(q, 0)
        nonzero_counts = [valid_edge_count(q, r) for r in range(1, q)]
        nr_mean = float(np.mean(nonzero_counts)) if nonzero_counts else np.nan
        ratio = n0 / nr_mean if nr_mean else np.nan
        row0 = residue_summary[(residue_summary["q"] == q) & (residue_summary["gap_residue_r"] == 0)]
        if row0.empty:
            continue
        row0 = row0.iloc[0]
        rows.append({
            "q": q,
            "N0": n0,
            "mean_Nr_nonzero": nr_mean,
            "N0_over_mean_Nr_nonzero": ratio,
            "empirical_p0": row0["share"],
            "edge_reference_p0": row0["valid_edge_reference_share"],
            "empirical_p0_minus_reference": row0["share_minus_reference"],
            "mean_omega_path_r0": row0.get("mean_omega_path", np.nan),
            "mean_exp_neg_omega_path_r0": row0.get("mean_exp_neg_omega_path", np.nan),
        })
    return pd.DataFrame(rows)


def write_interpretation(outdir: Path, residue_summary: pd.DataFrame, boundary: pd.DataFrame, telemetry: dict) -> None:
    lines: list[str] = []
    lines.append("# HOLT_GAP_POPULATION_BASELINE_AUDIT")
    lines.append("")
    lines.append("This supplemental audit partitions a chronological prime stream by Holt survival intervals")
    lines.append("$\\Delta H(s)=[s^2,\\operatorname{nextprime}(s)^2]$ and summarizes empirical gap-residue populations within those intervals.")
    lines.append("")
    lines.append("It does **not** implement Holt's exact cycle-recursion model for $G(p\\#)$. It is a compatibility audit between DS1, CHL2's local path-survival factor, and the survival-interval viewpoint.")
    lines.append("")
    lines.append("## Overall gap-residue populations")
    lines.append("")
    if not residue_summary.empty:
        show = residue_summary[residue_summary["q"].isin([3, 5, 7])].copy()
        show = show[["q", "gap_residue_r", "share", "valid_edge_reference_share", "share_minus_reference", "mean_omega_path", "mean_exp_neg_omega_path"]]
        lines.append(show.to_markdown(index=False, floatfmt=".6g"))
    lines.append("")
    lines.append("## Low-prime boundary view")
    lines.append("")
    if not boundary.empty:
        lines.append(boundary.to_markdown(index=False, floatfmt=".6g"))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("The valid-edge reference column is the asymptotic edge-count distribution $N_r(q)/\\sum_s N_s(q)$. It is not Holt's exact finite-stage gap-population model; it is the edge-count component that also underlies the orientation lift.")
    lines.append("")
    lines.append("When a CHL2 path-cache is provided, the audit also reports event-weighted averages of $\\Omega_Y^{\\rm path}$ and $\\exp(-\\Omega_Y^{\\rm path})$ by survival interval and gap residue. These quantities are a local CHL2 survival diagnostic and should be compared conceptually, not identified, with Holt's population survival across $\\Delta H(s)$.")
    lines.append("")
    lines.append("## Telemetry")
    lines.append("")
    for k in ["triples_processed", "missing_omega", "path_cache_rows", "elapsed_seconds", "prime_csv", "path_cache_file"]:
        lines.append(f"- `{k}`: `{telemetry.get(k)}`")
    lines.append("")
    outdir.joinpath("holt_gap_population_baseline_interpretation.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Supplemental Holt gap-population / CHL2 survival-interval compatibility audit.")
    p.add_argument("--prime-csv", required=True, type=Path, help="Chronological prime stream CSV or CSV.GZ.")
    p.add_argument("--path-cache-file", type=Path, default=None, help="Optional CHL2 path-exclusion cache with g1,g2,omega_path.")
    p.add_argument("--mods", default="3,5,7,11,13", help="Comma-separated diagnostic moduli.")
    p.add_argument("--gmax", type=int, default=2400, help="Finite support cutoff for g1,g2.")
    p.add_argument("--chunksize", type=int, default=1_000_000, help="Prime CSV chunksize.")
    p.add_argument("--sqrt-margin", type=int, default=10000, help="Prime margin around sqrt interval for survival stages.")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    mods = [int(x.strip()) for x in args.mods.split(",") if x.strip()]
    path_cache = load_path_cache(args.path_cache_file)
    path_cache_rows = int(len(path_cache)) if path_cache is not None else 0

    # First pass to estimate x-range cheaply from first/last prime.  Reading last line of gzip is annoying;
    # we do one streaming pass over chunks, but only keep min/max.  The real aggregation pass follows.
    min_p = None
    max_p = None
    total_primes = 0
    for arr in iter_prime_chunks(args.prime_csv, args.chunksize):
        if len(arr):
            total_primes += len(arr)
            mn = int(arr[0])
            mx = int(arr[-1])
            min_p = mn if min_p is None else min(min_p, mn)
            max_p = mx if max_p is None else max(max_p, mx)
    if min_p is None or max_p is None:
        raise RuntimeError("No primes found in prime CSV.")

    stage_primes = np.array(primes_around_sqrt(min_p, max_p, args.sqrt_margin), dtype=np.int64)
    if len(stage_primes) < 2:
        raise RuntimeError("Could not construct survival-stage primes around sqrt range.")

    state = new_state()
    carry = np.array([], dtype=np.int64)
    chunks = 0
    for arr in iter_prime_chunks(args.prime_csv, args.chunksize):
        carry = aggregate_chunk(
            primes=arr,
            carry=carry,
            stage_primes=stage_primes,
            mods=mods,
            gmax=args.gmax,
            path_cache=path_cache,
            state=state,
        )
        chunks += 1

    interval_df = build_interval_summary(state)
    residue_interval_df = build_residue_by_interval(state, mods)
    residue_summary_df = build_residue_summary(state, mods)
    boundary_df = build_boundary_summary(residue_summary_df, mods)

    interval_df.to_csv(outdir / "holt_survival_interval_summary.csv", index=False)
    residue_interval_df.to_csv(outdir / "holt_gap_residue_population_by_interval.csv", index=False)
    residue_summary_df.to_csv(outdir / "holt_gap_residue_population_summary.csv", index=False)
    boundary_df.to_csv(outdir / "holt_boundary_gap_population_summary.csv", index=False)

    telemetry = {
        "script": "holt_gap_population_baseline_audit.py",
        "purpose": "Supplemental Holt gap-population / CHL2 survival-interval compatibility audit",
        "prime_csv": str(args.prime_csv),
        "path_cache_file": str(args.path_cache_file) if args.path_cache_file else None,
        "path_cache_rows": path_cache_rows,
        "mods": mods,
        "gmax": args.gmax,
        "chunksize": args.chunksize,
        "chunks_processed": chunks,
        "total_primes_seen": total_primes,
        "prime_min": min_p,
        "prime_max": max_p,
        "stage_prime_min": int(stage_primes[0]),
        "stage_prime_max": int(stage_primes[-1]),
        "survival_intervals_nonempty": int(len(interval_df)),
        "triples_processed": int(state.triples_processed),
        "missing_omega": int(state.missing_omega),
        "elapsed_seconds": time.time() - t0,
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "outputs": [
            "holt_survival_interval_summary.csv",
            "holt_gap_residue_population_by_interval.csv",
            "holt_gap_residue_population_summary.csv",
            "holt_boundary_gap_population_summary.csv",
            "holt_gap_population_baseline_interpretation.md",
            "holt_gap_population_baseline_telemetry.json",
        ],
    }
    write_interpretation(outdir, residue_summary_df, boundary_df, telemetry)
    (outdir / "holt_gap_population_baseline_telemetry.json").write_text(
        json.dumps(telemetry, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"[OK] wrote Holt gap-population audit outputs to {outdir}")
    print(f"[OK] triples processed: {state.triples_processed:,}")
    if args.path_cache_file:
        print(f"[OK] missing omega_path rows: {state.missing_omega:,}")


if __name__ == "__main__":
    main()
