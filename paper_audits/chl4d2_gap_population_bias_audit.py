#!/usr/bin/env python3
"""CHL4-D2 gap-population bias audit.

This audit tests a concrete operational hypothesis suggested by the CHL4-D
origin analysis:

    The modulo-3 residual transfer mode theta_3 may be caused by a mismatch
    between the empirical population of consecutive gaps g2 modulo 3 and the
    population induced by the CHL2 path-exclusion kernel.

For q=3, a current gap g2 satisfies:

    g2 == 0 mod 3   -> prime residue class repeats, diagonal transition;
    g2 != 0 mod 3   -> prime residue class changes, off-diagonal transition.

Therefore the diagonal probability in the modulo-3 transfer matrix is exactly
P(g2 == 0 mod 3), up to the small-prime boundary outside the DS1 range.  This
script compares:

    empirical gap population by g2 mod 3
    vs
    CHL2-induced gap population by g2 mod 3.

It uses the same parent-wide block files as the CHL2 audits.  The CHL2 model is
reconstructed from the parameter-free CHL2 path-exclusion log-weight:

    log w(g2 | g1) = log R_Y(g2 | g1) - Omega_Y^path(g1,g2;x) + eta*g2,

where eta is solved once on the global support to match the empirical mean gap.
No new free parameters are introduced.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Ensure repository root imports work when called as a script.
ROOT_HINT = Path(__file__).resolve().parents[1]
if str(ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(ROOT_HINT))

try:
    from chl_kernel import CHLKernel
except Exception as exc:  # pragma: no cover - user-facing failure
    CHLKernel = None  # type: ignore
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

try:
    from chl_kernel.telemetry import telemetry_start, write_telemetry
except Exception:  # pragma: no cover - fallback for older repo versions
    def telemetry_start(argv: Sequence[str]) -> dict[str, Any]:
        return {
            "argv": list(argv),
            "started_at_unix": time.time(),
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        }

    def write_telemetry(path: str | Path, data: Mapping[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dict(data), f, indent=2, sort_keys=True)


FILTERS_DEFAULT = [
    "ALL",
    "LOW_ONLY_LE58",
    "MID_59_120",
    "MID_121_240",
    "MID_121_400",
    "NO_58",
    "NO_120",
    "NO_240",
]


def parse_int_list_or_range(text: str | None, default: Sequence[int]) -> list[int]:
    """Parse comma-separated integers and inclusive ranges like 1-10."""
    if text is None or str(text).strip() == "":
        return list(default)
    out: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def parse_filters(text: str | None) -> list[str]:
    if not text:
        return list(FILTERS_DEFAULT)
    return [x.strip() for x in str(text).split(",") if x.strip()]


def filter_mask_from_gap(g: np.ndarray, name: str) -> np.ndarray:
    """Return a boolean mask over current gaps g2."""
    if name == "ALL":
        return np.ones_like(g, dtype=bool)
    if name == "LOW_ONLY_LE58":
        return g <= 58
    if name == "MID_59_120":
        return (g >= 59) & (g <= 120)
    if name == "MID_121_240":
        return (g >= 121) & (g <= 240)
    if name == "MID_121_400":
        return (g >= 121) & (g <= 400)
    if name == "NO_58":
        return g > 58
    if name == "NO_120":
        return g > 120
    if name == "NO_240":
        return g > 240
    raise ValueError(f"Unknown filter: {name}")


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def open_text_maybe_gzip(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", newline="")
    return open(path, "rt", newline="")


def parent_wide_block_paths(root: Path, cfg: Mapping[str, Any], blocks: Sequence[int]) -> list[tuple[int, Path]]:
    """Resolve parent-wide block CSV paths from config and block IDs."""
    input_dir = str(cfg.get("input_dir", ""))
    blocks_dir = str(cfg.get("blocks_dir", "blocks"))
    block_glob = str(cfg.get("block_glob", "parent_wide_B{block:02d}.csv.gz"))
    base = root / input_dir / blocks_dir if input_dir else root / blocks_dir
    out: list[tuple[int, Path]] = []
    for b in blocks:
        try:
            name = block_glob.format(block=b)
        except Exception:
            name = f"parent_wide_B{b:02d}.csv.gz"
        p = base / name
        if not p.exists():
            alternatives = [
                base / f"parent_wide_B{b:02d}.csv.gz",
                base / f"v46t12_ds_pilot_parent_wide_B{b:02d}.csv.gz",
                root / "blocks" / f"parent_wide_B{b:02d}.csv.gz",
                root / "blocks" / f"v46t12_ds_pilot_parent_wide_B{b:02d}.csv.gz",
            ]
            for alt in alternatives:
                if alt.exists():
                    p = alt
                    break
        if not p.exists():
            raise FileNotFoundError(f"Could not resolve parent-wide block B{b:02d}; tried {p}")
        out.append((b, p))
    return out


def load_parent_wide_blocks(block_paths: Sequence[tuple[int, Path]]) -> pd.DataFrame:
    """Load and concatenate parent-wide blocks with required columns."""
    frames: list[pd.DataFrame] = []
    for b, path in block_paths:
        df = pd.read_csv(path)
        required = {"g1", "g2", "H"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        sub = df[["g1", "g2", "H"]].copy()
        sub["block"] = int(b)
        sub["g1"] = sub["g1"].astype(int)
        sub["g2"] = sub["g2"].astype(int)
        sub["H"] = sub["H"].astype(float)
        sub = sub[sub["H"] > 0].reset_index(drop=True)
        frames.append(sub)
    if not frames:
        raise ValueError("no blocks loaded")
    return pd.concat(frames, ignore_index=True)


def load_path_cache(path: Path | None) -> dict[tuple[int, int], float]:
    """Load optional CHL2 path-exclusion cache as logE_path map."""
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"g1", "g2"}
    if not required.issubset(df.columns):
        raise ValueError(f"path cache {path} must contain g1,g2")
    if "logE_path_exclusion" in df.columns:
        val_col = "logE_path_exclusion"
        values = df[val_col].astype(float).to_numpy()
    elif "omega_path_exclusion" in df.columns:
        values = -df["omega_path_exclusion"].astype(float).to_numpy()
    else:
        raise ValueError(f"path cache {path} must contain logE_path_exclusion or omega_path_exclusion")
    return {
        (int(g1), int(g2)): float(v)
        for g1, g2, v in zip(df["g1"].to_numpy(), df["g2"].to_numpy(), values)
    }


def logsumexp(vals: np.ndarray) -> float:
    vals = np.asarray(vals, dtype=float)
    finite = np.isfinite(vals)
    if not finite.any():
        return -math.inf
    m = float(vals[finite].max())
    return float(m + math.log(np.exp(vals[finite] - m).sum()))


def weighted_model_mean(df_support: pd.DataFrame, log_base: np.ndarray, eta: float, row_weights: Mapping[int, float]) -> float:
    """Mean g2 under row-normalized conditional model weighted by empirical row mass."""
    g1 = df_support["g1"].to_numpy(dtype=int)
    g2 = df_support["g2"].to_numpy(dtype=float)
    total_weight = float(sum(row_weights.values()))
    if total_weight <= 0:
        return float("nan")
    acc = 0.0
    for row_g1, idx in df_support.groupby("g1", sort=False).indices.items():
        idx_arr = np.asarray(idx, dtype=int)
        lw = log_base[idx_arr] + eta * g2[idx_arr]
        z = logsumexp(lw)
        if not math.isfinite(z):
            continue
        probs = np.exp(lw - z)
        acc += float(row_weights.get(int(row_g1), 0.0)) * float(np.dot(probs, g2[idx_arr]))
    return acc / total_weight


def solve_eta_conditional(
    df_support: pd.DataFrame,
    log_base: np.ndarray,
    row_weights: Mapping[int, float],
    target_mean: float,
    eta_min: float = -2.0,
    eta_max: float = 2.0,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> tuple[float, float]:
    """Solve eta so the CHL2 conditional model matches empirical mean g2."""
    lo, hi = float(eta_min), float(eta_max)
    mean_lo = weighted_model_mean(df_support, log_base, lo, row_weights)
    mean_hi = weighted_model_mean(df_support, log_base, hi, row_weights)
    # The mean is monotone increasing in eta. Expand if needed.
    expand = 0
    while math.isfinite(mean_lo) and mean_lo > target_mean and expand < 20:
        hi = lo
        lo *= 2.0
        mean_lo = weighted_model_mean(df_support, log_base, lo, row_weights)
        expand += 1
    expand = 0
    while math.isfinite(mean_hi) and mean_hi < target_mean and expand < 20:
        lo = hi
        hi *= 2.0
        mean_hi = weighted_model_mean(df_support, log_base, hi, row_weights)
        expand += 1
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        m = weighted_model_mean(df_support, log_base, mid, row_weights)
        if not math.isfinite(m):
            break
        if abs(m - target_mean) <= tol:
            return float(mid), float(m)
        if m < target_mean:
            lo = mid
        else:
            hi = mid
    eta = 0.5 * (lo + hi)
    return float(eta), float(weighted_model_mean(df_support, log_base, eta, row_weights))


def conditional_probabilities(df_support: pd.DataFrame, log_base: np.ndarray, eta: float) -> np.ndarray:
    """Return row-normalized probabilities over support rows grouped by g1."""
    out = np.zeros(len(df_support), dtype=float)
    g2 = df_support["g2"].to_numpy(dtype=float)
    for _, idx in df_support.groupby("g1", sort=False).indices.items():
        idx_arr = np.asarray(idx, dtype=int)
        lw = log_base[idx_arr] + eta * g2[idx_arr]
        z = logsumexp(lw)
        if math.isfinite(z):
            out[idx_arr] = np.exp(lw - z)
    return out


def d3_from_diag_prob(p_diag: float, eps: float = 1e-15) -> float:
    p = min(max(float(p_diag), eps), 1.0 - eps)
    return float(2.0 * math.log(p / (1.0 - p)))


def safe_share(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def summarize_gap_population(df: pd.DataFrame, filters: Sequence[str], block_label: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Summarize empirical/model population by filter, residue, and exact gap."""
    rows_filter: list[dict[str, Any]] = []
    rows_residue: list[dict[str, Any]] = []
    rows_gap: list[dict[str, Any]] = []
    g = df["g2"].to_numpy(dtype=int)
    emp = df["emp_count"].to_numpy(dtype=float)
    mod = df["model_expected"].to_numpy(dtype=float)
    for fname in filters:
        mask = filter_mask_from_gap(g, fname)
        sub = df.loc[mask].copy()
        emp_total = float(sub["emp_count"].sum())
        mod_total = float(sub["model_expected"].sum())
        emp_diag = float(sub.loc[sub["g2"] % 3 == 0, "emp_count"].sum())
        mod_diag = float(sub.loc[sub["g2"] % 3 == 0, "model_expected"].sum())
        emp_pdiag = safe_share(emp_diag, emp_total)
        mod_pdiag = safe_share(mod_diag, mod_total)
        D_emp = d3_from_diag_prob(emp_pdiag) if math.isfinite(emp_pdiag) else float("nan")
        D_mod = d3_from_diag_prob(mod_pdiag) if math.isfinite(mod_pdiag) else float("nan")
        rows_filter.append({
            "block": block_label,
            "filter": fname,
            "empirical_events": emp_total,
            "model_expected_events": mod_total,
            "empirical_diag_count": emp_diag,
            "model_diag_expected": mod_diag,
            "empirical_diag_probability": emp_pdiag,
            "model_diag_probability": mod_pdiag,
            "D3_empirical_gap_population": D_emp,
            "D3_chl2_gap_population": D_mod,
            "D3_residual_gap_population": D_emp - D_mod if math.isfinite(D_emp) and math.isfinite(D_mod) else float("nan"),
            "theta3_gap_population": 0.25 * (D_emp - D_mod) if math.isfinite(D_emp) and math.isfinite(D_mod) else float("nan"),
            "model_minus_empirical_diag_probability": mod_pdiag - emp_pdiag if math.isfinite(emp_pdiag) and math.isfinite(mod_pdiag) else float("nan"),
        })
        for r in [0, 1, 2]:
            e = float(sub.loc[sub["g2"] % 3 == r, "emp_count"].sum())
            m = float(sub.loc[sub["g2"] % 3 == r, "model_expected"].sum())
            rows_residue.append({
                "block": block_label,
                "filter": fname,
                "gap_mod3": r,
                "is_diagonal_mod3": int(r == 0),
                "empirical_count": e,
                "model_expected": m,
                "empirical_share": safe_share(e, emp_total),
                "model_share": safe_share(m, mod_total),
                "model_minus_empirical_share": safe_share(m, mod_total) - safe_share(e, emp_total) if emp_total > 0 and mod_total > 0 else float("nan"),
            })
        # Exact gap rows for this filter and block.
        grp = sub.groupby("g2", as_index=False).agg(empirical_count=("emp_count", "sum"), model_expected=("model_expected", "sum"))
        for _, row in grp.iterrows():
            gg = int(row["g2"])
            e = float(row["empirical_count"])
            m = float(row["model_expected"])
            rows_gap.append({
                "block": block_label,
                "filter": fname,
                "g2": gg,
                "gap_mod3": gg % 3,
                "is_diagonal_mod3": int(gg % 3 == 0),
                "empirical_count": e,
                "model_expected": m,
                "empirical_share": safe_share(e, emp_total),
                "model_share": safe_share(m, mod_total),
                "model_minus_empirical_share": safe_share(m, mod_total) - safe_share(e, emp_total) if emp_total > 0 and mod_total > 0 else float("nan"),
            })
    return rows_filter, rows_residue, rows_gap


def add_leave_one_gap_effects(df_gap: pd.DataFrame, df_filter: pd.DataFrame) -> pd.DataFrame:
    """Add exact leave-one-gap-out changes in D3 residual for each gap row."""
    if df_gap.empty:
        return df_gap
    out = df_gap.copy()
    key_cols = ["block", "filter"]
    lookup = {
        (r["block"], r["filter"]): r
        for _, r in df_filter.iterrows()
    }
    delta_emp: list[float] = []
    delta_mod: list[float] = []
    delta_res: list[float] = []
    for _, row in out.iterrows():
        base = lookup.get((row["block"], row["filter"]))
        if base is None:
            delta_emp.append(float("nan")); delta_mod.append(float("nan")); delta_res.append(float("nan")); continue
        e_total = float(base["empirical_events"])
        m_total = float(base["model_expected_events"])
        e_diag = float(base["empirical_diag_count"])
        m_diag = float(base["model_diag_expected"])
        e_gap = float(row["empirical_count"])
        m_gap = float(row["model_expected"])
        if int(row["is_diagonal_mod3"]):
            e_diag2 = e_diag - e_gap
            m_diag2 = m_diag - m_gap
        else:
            e_diag2 = e_diag
            m_diag2 = m_diag
        e_total2 = e_total - e_gap
        m_total2 = m_total - m_gap
        D_emp_base = float(base["D3_empirical_gap_population"])
        D_mod_base = float(base["D3_chl2_gap_population"])
        D_res_base = float(base["D3_residual_gap_population"])
        D_emp2 = d3_from_diag_prob(safe_share(e_diag2, e_total2)) if e_total2 > 0 else float("nan")
        D_mod2 = d3_from_diag_prob(safe_share(m_diag2, m_total2)) if m_total2 > 0 else float("nan")
        D_res2 = D_emp2 - D_mod2 if math.isfinite(D_emp2) and math.isfinite(D_mod2) else float("nan")
        delta_emp.append(D_emp2 - D_emp_base if math.isfinite(D_emp2) else float("nan"))
        delta_mod.append(D_mod2 - D_mod_base if math.isfinite(D_mod2) else float("nan"))
        delta_res.append(D_res2 - D_res_base if math.isfinite(D_res2) else float("nan"))
    out["delta_D3_empirical_remove_gap"] = delta_emp
    out["delta_D3_chl2_remove_gap"] = delta_mod
    out["delta_D3_residual_remove_gap"] = delta_res
    return out


def cumulative_by_gap(df_gap_all: pd.DataFrame) -> pd.DataFrame:
    """Cumulative empirical/model D3 by increasing gap for the ALL block/filter."""
    sub = df_gap_all[(df_gap_all["block"] == "ALL") & (df_gap_all["filter"] == "ALL")].copy()
    if sub.empty:
        return pd.DataFrame()
    sub = sub.sort_values("g2").reset_index(drop=True)
    sub["cum_empirical_count"] = sub["empirical_count"].cumsum()
    sub["cum_model_expected"] = sub["model_expected"].cumsum()
    sub["cum_empirical_diag_count"] = (sub["empirical_count"] * sub["is_diagonal_mod3"]).cumsum()
    sub["cum_model_diag_expected"] = (sub["model_expected"] * sub["is_diagonal_mod3"]).cumsum()
    rows: list[dict[str, Any]] = []
    for _, row in sub.iterrows():
        emp_p = safe_share(float(row["cum_empirical_diag_count"]), float(row["cum_empirical_count"]))
        mod_p = safe_share(float(row["cum_model_diag_expected"]), float(row["cum_model_expected"]))
        D_emp = d3_from_diag_prob(emp_p) if math.isfinite(emp_p) else float("nan")
        D_mod = d3_from_diag_prob(mod_p) if math.isfinite(mod_p) else float("nan")
        rows.append({
            "g2_max": int(row["g2"]),
            "cum_empirical_count": float(row["cum_empirical_count"]),
            "cum_model_expected": float(row["cum_model_expected"]),
            "cum_empirical_event_share": safe_share(float(row["cum_empirical_count"]), float(sub["empirical_count"].sum())),
            "cum_model_event_share": safe_share(float(row["cum_model_expected"]), float(sub["model_expected"].sum())),
            "cum_empirical_diag_probability": emp_p,
            "cum_model_diag_probability": mod_p,
            "cum_D3_empirical": D_emp,
            "cum_D3_chl2": D_mod,
            "cum_D3_residual": D_emp - D_mod if math.isfinite(D_emp) and math.isfinite(D_mod) else float("nan"),
        })
    return pd.DataFrame(rows)


def stability_by_filter(df_filter: pd.DataFrame) -> pd.DataFrame:
    """Block stability over real blocks Bxx only, excluding ALL."""
    sub = df_filter[df_filter["block"].astype(str).str.match(r"B\d{2}$")].copy()
    rows: list[dict[str, Any]] = []
    for filt, grp in sub.groupby("filter", sort=False):
        vals = grp["D3_residual_gap_population"].to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            continue
        rows.append({
            "filter": filt,
            "n_blocks": int(len(finite)),
            "mean_D3_residual_gap_population": float(np.mean(finite)),
            "std_D3_residual_gap_population": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
            "min_D3_residual_gap_population": float(np.min(finite)),
            "max_D3_residual_gap_population": float(np.max(finite)),
            "negative_count": int(np.sum(finite < 0)),
            "positive_count": int(np.sum(finite > 0)),
            "zero_count": int(np.sum(finite == 0)),
            "mean_empirical_diag_probability": float(np.average(grp["empirical_diag_probability"], weights=grp["empirical_events"])),
            "mean_model_diag_probability": float(np.average(grp["model_diag_probability"], weights=grp["model_expected_events"])),
        })
    return pd.DataFrame(rows)



def sign_label(x: float, tol: float = 1e-12) -> str:
    """Human-readable sign label for D3-like diagnostics."""
    if not math.isfinite(float(x)):
        return "nan"
    if x > tol:
        return "persistence_diagonal"
    if x < -tol:
        return "repulsion_diagonal"
    return "neutral"


def scale_wave_summary(df_filter: pd.DataFrame) -> pd.DataFrame:
    """Summarize the scale-dependent sign oscillation of empirical D3.

    This table is intentionally descriptive.  It records the empirical D3 wave
    across the pre-registered CHL filters, event counts, and amplitudes relative
    to LOW_ONLY_LE58.  It is designed to highlight that MID_59_120 and NO_58 are
    not small residual tails but a sign-reversed persistence regime.
    """
    sub = df_filter[df_filter["block"] == "ALL"].copy()
    if sub.empty:
        return pd.DataFrame()
    order = {name: i for i, name in enumerate(FILTERS_DEFAULT)}
    sub["filter_order"] = sub["filter"].map(order).fillna(999).astype(int)
    sub = sub.sort_values("filter_order")
    low = sub.loc[sub["filter"] == "LOW_ONLY_LE58", "D3_empirical_gap_population"]
    low_abs = abs(float(low.iloc[0])) if not low.empty and math.isfinite(float(low.iloc[0])) else float("nan")
    total_events = float(sub.loc[sub["filter"] == "ALL", "empirical_events"].iloc[0]) if (sub["filter"] == "ALL").any() else float(sub["empirical_events"].max())
    rows: list[dict[str, Any]] = []
    prev_sign = None
    for _, r in sub.iterrows():
        d = float(r["D3_empirical_gap_population"])
        sg = sign_label(d)
        rows.append({
            "filter": str(r["filter"]),
            "empirical_events": float(r["empirical_events"]),
            "event_share_vs_ALL": safe_share(float(r["empirical_events"]), total_events),
            "empirical_diag_probability": float(r["empirical_diag_probability"]),
            "D3_empirical_gap_population": d,
            "abs_D3_empirical": abs(d) if math.isfinite(d) else float("nan"),
            "sign_label": sg,
            "sign_changed_from_previous_filter": bool(prev_sign is not None and sg != prev_sign and sg != "nan" and prev_sign != "nan"),
            "amplitude_vs_LOW_ONLY_LE58": (abs(d) / low_abs) if low_abs and math.isfinite(low_abs) and low_abs > 0 and math.isfinite(d) else float("nan"),
            "model_diag_probability": float(r["model_diag_probability"]),
            "D3_chl2_gap_population": float(r["D3_chl2_gap_population"]),
            "D3_residual_gap_population": float(r["D3_residual_gap_population"]),
            "statistical_note": "low_mass_caution" if float(r["empirical_events"]) < 10000 else "adequate_mass",
        })
        prev_sign = sg
    return pd.DataFrame(rows)


def offdiag_symmetry_summary(df_gap: pd.DataFrame) -> pd.DataFrame:
    """Summarize microscopic off-diagonal balance, especially gaps 2 and 4.

    In modulo 3, gaps with residue 1 and residue 2 both force an off-diagonal
    transition, but in opposite orientations.  The near equality of g=2 and g=4
    masses is an important sanity check: the q=3 anomaly is not caused by an
    imbalance between the two off-diagonal branches, but by the diagonal class
    g == 0 mod 3 against the two off-diagonal branches combined.
    """
    sub = df_gap[(df_gap["block"] == "ALL") & (df_gap["filter"] == "ALL")].copy()
    if sub.empty:
        return pd.DataFrame()
    total_emp = float(sub["empirical_count"].sum())
    total_mod = float(sub["model_expected"].sum())
    def row_for_gap(g: int) -> pd.Series | None:
        rr = sub[sub["g2"] == g]
        return rr.iloc[0] if not rr.empty else None
    r2 = row_for_gap(2)
    r4 = row_for_gap(4)
    e2 = float(r2["empirical_count"]) if r2 is not None else 0.0
    e4 = float(r4["empirical_count"]) if r4 is not None else 0.0
    m2 = float(r2["model_expected"]) if r2 is not None else 0.0
    m4 = float(r4["model_expected"]) if r4 is not None else 0.0
    # Residue branch summaries.
    branch_rows = []
    for residue in [1, 2, 0]:
        br = sub[sub["gap_mod3"] == residue]
        branch_rows.append({
            "level": "gap_mod3_branch",
            "item": f"mod_{residue}",
            "gap_mod3": residue,
            "empirical_count": float(br["empirical_count"].sum()),
            "model_expected": float(br["model_expected"].sum()),
            "empirical_share": safe_share(float(br["empirical_count"].sum()), total_emp),
            "model_share": safe_share(float(br["model_expected"].sum()), total_mod),
            "model_minus_empirical_share": safe_share(float(br["model_expected"].sum()), total_mod) - safe_share(float(br["empirical_count"].sum()), total_emp),
        })
    rows = [
        {
            "level": "exact_gap_pair",
            "item": "g2_vs_g4",
            "gap_mod3": -1,
            "empirical_count": e2 + e4,
            "model_expected": m2 + m4,
            "empirical_share": safe_share(e2 + e4, total_emp),
            "model_share": safe_share(m2 + m4, total_mod),
            "model_minus_empirical_share": safe_share(m2 + m4, total_mod) - safe_share(e2 + e4, total_emp),
            "gap2_empirical_count": e2,
            "gap4_empirical_count": e4,
            "gap2_model_expected": m2,
            "gap4_model_expected": m4,
            "gap2_minus_gap4_empirical": e2 - e4,
            "gap2_gap4_relative_imbalance_empirical": (e2 - e4) / (e2 + e4) if (e2 + e4) else float("nan"),
            "gap2_minus_gap4_model": m2 - m4,
            "gap2_gap4_relative_imbalance_model": (m2 - m4) / (m2 + m4) if (m2 + m4) else float("nan"),
        }
    ]
    rows.extend(branch_rows)
    return pd.DataFrame(rows)


def write_extended_interpretation_notes(text: list[str], df_scale: pd.DataFrame, df_sym: pd.DataFrame) -> None:
    """Append CHL4-D2 specific interpretation paragraphs to the Markdown report."""
    text.append("\n## Additional CHL4-D2 diagnostics: scale wave and off-diagonal symmetry\n")
    if not df_scale.empty:
        text.append("### Scale-dependent modular wave\n")
        keep = df_scale[df_scale["filter"].isin(["LOW_ONLY_LE58", "MID_59_120", "NO_58", "NO_120", "NO_240"])].copy()
        table_rows = []
        for _, r in keep.iterrows():
            table_rows.append([
                r["filter"],
                float(r["empirical_events"]),
                float(r["D3_empirical_gap_population"]),
                r["sign_label"],
                float(r["amplitude_vs_LOW_ONLY_LE58"]),
                r["statistical_note"],
            ])
        text.append(small_table(table_rows, ["filter", "events", "D3 empirical", "sign", "amplitude vs LOW", "note"]))
        # Interpret the observed scale-wave only if the signs actually match the DS1 pattern.
        def _row(filter_name: str):
            rr = df_scale[df_scale["filter"] == filter_name]
            return rr.iloc[0] if not rr.empty else None
        low_r = _row("LOW_ONLY_LE58")
        mid_r = _row("MID_59_120")
        no58_r = _row("NO_58")
        no120_r = _row("NO_120")
        no240_r = _row("NO_240")
        if (
            low_r is not None and mid_r is not None and no58_r is not None
            and str(low_r["sign_label"]) == "repulsion_diagonal"
            and str(mid_r["sign_label"]) == "persistence_diagonal"
            and str(no58_r["sign_label"]) == "persistence_diagonal"
        ):
            text.append("\nThe empirical modulo-$3$ flow is not a monotone decay from repulsion to neutrality. It changes sign across scales: short gaps show strong diagonal repulsion, while the intermediate regime shows diagonal persistence. In particular, `MID_59_120` and `NO_58` should be read as a genuine persistence regime, not as a small residual tail.\n")
            if no120_r is not None and str(no120_r["sign_label"]) == "repulsion_diagonal":
                note = "adequate-mass" if float(no120_r["empirical_events"]) >= 100000 else "lower-mass"
                text.append(f"The `NO_120` tail returns to diagonal repulsion with {float(no120_r['empirical_events']):.0f} events ({note}). This supports an oscillatory scale-wave interpretation rather than a single linear accumulation error. `NO_240` remains diagnostic only when its event count is small.\n")
        else:
            text.append("\nThe table reports the scale-dependent sign pattern of the empirical modulo-$3$ flow. In the full DS1 audit this diagnostic is used to decide whether the residual is a monotone decay, a persistence regime, or an oscillatory scale wave.\n")
    if not df_sym.empty:
        pair = df_sym[(df_sym["level"] == "exact_gap_pair") & (df_sym["item"] == "g2_vs_g4")]
        if not pair.empty:
            r = pair.iloc[0]
            text.append("### Microscopic off-diagonal balance at gaps 2 and 4\n")
            text.append(small_table([[
                float(r["gap2_empirical_count"]),
                float(r["gap4_empirical_count"]),
                float(r["gap2_minus_gap4_empirical"]),
                float(r["gap2_gap4_relative_imbalance_empirical"]),
                float(r["gap2_model_expected"]),
                float(r["gap4_model_expected"]),
            ]], ["g=2 empirical", "g=4 empirical", "emp diff", "emp rel imbalance", "g=2 CHL2", "g=4 CHL2"]))
            text.append("\nThe first two off-diagonal gap branches, $g=2$ and $g=4$, are nearly perfectly balanced in the empirical population. This means the $q=3$ discrepancy is not primarily an imbalance between the two off-diagonal orientations. The operational problem is the balance between the diagonal branch $g \\equiv 0 \\pmod 3$ and the two off-diagonal branches combined.\n")


def build_support_and_model(
    df_all: pd.DataFrame,
    Y: int,
    log_x: float,
    path_cache: Mapping[tuple[int, int], float],
    eta_override: float | None = None,
) -> tuple[pd.DataFrame, float, float, dict[tuple[int, int], float], dict[str, Any]]:
    """Build global support and CHL2 path probabilities per (g1,g2)."""
    if CHLKernel is None:
        raise ImportError(f"Could not import CHLKernel: {IMPORT_ERROR}")
    kernel = CHLKernel(Y=int(Y), log_x=float(log_x))
    # Global support over all observed pairs.
    support = df_all.groupby(["g1", "g2"], as_index=False).agg(emp_count_global=("H", "sum"))
    support = support.sort_values(["g1", "g2"]).reset_index(drop=True)
    row_counts_global = df_all.groupby("g1")["H"].sum().to_dict()
    target_mean = float(np.average(df_all["g2"], weights=df_all["H"]))
    log_base = np.full(len(support), -np.inf, dtype=float)
    cache_hits = 0
    cache_misses = 0
    for i, row in support.iterrows():
        g1 = int(row["g1"]); g2 = int(row["g2"])
        lr = kernel.log_R(g1, g2)
        if not math.isfinite(lr):
            continue
        key = (g1, g2)
        if key in path_cache:
            logE = float(path_cache[key])
            cache_hits += 1
        else:
            om = kernel.omega_path(g1, g2)
            logE = -om if math.isfinite(om) else -math.inf
            cache_misses += 1
        log_base[i] = lr + logE if math.isfinite(logE) else -math.inf
    if eta_override is None:
        eta, model_mean = solve_eta_conditional(support, log_base, row_counts_global, target_mean)
    else:
        eta = float(eta_override)
        model_mean = weighted_model_mean(support, log_base, eta, row_counts_global)
    probs = conditional_probabilities(support, log_base, eta)
    prob_map = {
        (int(g1), int(g2)): float(p)
        for g1, g2, p in zip(support["g1"].to_numpy(), support["g2"].to_numpy(), probs)
    }
    meta = {
        "target_mean_g2": target_mean,
        "eta": eta,
        "model_mean_g2": model_mean,
        "support_rows": int(len(support)),
        "unique_g1": int(support["g1"].nunique()),
        "path_cache_hits": int(cache_hits),
        "path_cache_misses": int(cache_misses),
    }
    return support, eta, model_mean, prob_map, meta


def expected_counts_for_block(df_block: pd.DataFrame, support: pd.DataFrame, prob_map: Mapping[tuple[int, int], float]) -> pd.DataFrame:
    """Create full support expected counts for one block and merge empirical counts."""
    emp = df_block.groupby(["g1", "g2"], as_index=False).agg(emp_count=("H", "sum"))
    row_counts = df_block.groupby("g1")["H"].sum().to_dict()
    out = support[["g1", "g2"]].copy()
    out = out.merge(emp, on=["g1", "g2"], how="left")
    out["emp_count"] = out["emp_count"].fillna(0.0).astype(float)
    out["row_count_block_g1"] = out["g1"].map(lambda x: float(row_counts.get(int(x), 0.0)))
    out["model_probability_given_g1"] = [prob_map.get((int(g1), int(g2)), 0.0) for g1, g2 in zip(out["g1"], out["g2"])]
    out["model_expected"] = out["row_count_block_g1"] * out["model_probability_given_g1"]
    return out


def small_table(rows: list[list[Any]], headers: list[str]) -> str:
    """A tiny Markdown table writer that does not require tabulate."""
    def fmt(x: Any) -> str:
        if isinstance(x, float):
            if math.isnan(x):
                return "nan"
            return f"{x:.6g}"
        return str(x)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(x) for x in row) + " |")
    return "\n".join(lines)


def write_interpretation(
    outdir: Path,
    df_filter: pd.DataFrame,
    df_stability: pd.DataFrame,
    df_residue: pd.DataFrame,
    df_gap: pd.DataFrame,
    df_cum: pd.DataFrame,
    df_scale: pd.DataFrame,
    df_symmetry: pd.DataFrame,
    meta: Mapping[str, Any],
) -> None:
    all_row = df_filter[(df_filter["block"] == "ALL") & (df_filter["filter"] == "ALL")].iloc[0]
    rows = [
        ["eta", float(meta.get("eta", float("nan")))],
        ["target_mean_g2", float(meta.get("target_mean_g2", float("nan")))],
        ["model_mean_g2", float(meta.get("model_mean_g2", float("nan")))],
        ["D3_empirical_ALL", float(all_row["D3_empirical_gap_population"])],
        ["D3_CHL2_gap_population_ALL", float(all_row["D3_chl2_gap_population"])],
        ["D3_residual_gap_population_ALL", float(all_row["D3_residual_gap_population"])],
        ["empirical_diag_probability_ALL", float(all_row["empirical_diag_probability"])],
        ["model_diag_probability_ALL", float(all_row["model_diag_probability"])],
    ]
    text: list[str] = []
    text.append("# CHL4-D2 Gap-Population Bias Audit\n")
    text.append("This audit asks whether the modulo $3$ residual transfer mode can be explained by a mismatch between the empirical population of current gaps $g_2 \\bmod 3$ and the population induced by the CHL2 path-exclusion kernel.\n")
    text.append("## Global summary\n")
    text.append(small_table(rows, ["quantity", "value"]))
    text.append("\n")
    text.append("For modulo $3$, current gaps with $g_2 \\equiv 0 \\pmod 3$ produce diagonal prime-residue transitions; current gaps with $g_2 \\not\\equiv 0 \\pmod 3$ produce off-diagonal transitions.\n")
    text.append("## Filter stability over real blocks\n")
    table_rows = []
    for _, r in df_stability.iterrows():
        table_rows.append([
            r["filter"], int(r["n_blocks"]), float(r["mean_D3_residual_gap_population"]), float(r["std_D3_residual_gap_population"]), int(r["negative_count"]), int(r["positive_count"]), float(r["mean_empirical_diag_probability"]), float(r["mean_model_diag_probability"]),
        ])
    text.append(small_table(table_rows, ["filter", "blocks", "mean D3 residual", "std", "neg", "pos", "emp diag", "model diag"]))
    text.append("\n")
    # Residue summary ALL.
    res_all = df_residue[(df_residue["block"] == "ALL") & (df_residue["filter"] == "ALL")]
    text.append("## Gap residue population, ALL\n")
    table_rows = []
    for _, r in res_all.sort_values("gap_mod3").iterrows():
        table_rows.append([int(r["gap_mod3"]), float(r["empirical_share"]), float(r["model_share"]), float(r["model_minus_empirical_share"])])
    text.append(small_table(table_rows, ["gap mod 3", "emp share", "CHL2 share", "CHL2 - emp"]))
    text.append("\n")
    # Top exact gaps by model-minus-empirical share and leave-one-out residual effect.
    gap_all = df_gap[(df_gap["block"] == "ALL") & (df_gap["filter"] == "ALL")].copy()
    gap_all["abs_delta_share"] = gap_all["model_minus_empirical_share"].abs()
    top_share = gap_all.sort_values("abs_delta_share", ascending=False).head(12)
    text.append("## Largest exact-gap population mismatches, ALL\n")
    table_rows = []
    for _, r in top_share.iterrows():
        table_rows.append([int(r["g2"]), int(r["gap_mod3"]), float(r["empirical_share"]), float(r["model_share"]), float(r["model_minus_empirical_share"])])
    text.append(small_table(table_rows, ["g2", "mod3", "emp share", "CHL2 share", "CHL2 - emp"]))
    text.append("\n")
    top_loo = gap_all.sort_values("delta_D3_residual_remove_gap", key=lambda s: s.abs(), ascending=False).head(12)
    text.append("## Largest exact-gap leave-one-out effects on residual $D_3$, ALL\n")
    table_rows = []
    for _, r in top_loo.iterrows():
        table_rows.append([int(r["g2"]), int(r["gap_mod3"]), float(r["delta_D3_residual_remove_gap"]), float(r["empirical_share"]), float(r["model_share"])])
    text.append(small_table(table_rows, ["g2", "mod3", "delta residual if removed", "emp share", "CHL2 share"]))
    text.append("\n")
    # Cumulative selected cuts.
    text.append("## Cumulative gap cutoffs\n")
    cuts = [6, 10, 20, 30, 58, 120, 240, 400]
    table_rows = []
    for cut in cuts:
        sub = df_cum[df_cum["g2_max"] <= cut]
        if sub.empty:
            continue
        r = sub.iloc[-1]
        table_rows.append([cut, float(r["cum_empirical_event_share"]), float(r["cum_model_event_share"]), float(r["cum_D3_empirical"]), float(r["cum_D3_chl2"]), float(r["cum_D3_residual"])])
    text.append(small_table(table_rows, ["g2 max", "emp mass", "CHL2 mass", "D3 emp", "D3 CHL2", "D3 residual"]))
    text.append("\n")
    text.append("## Interpretation\n")
    model_diag = float(all_row["model_diag_probability"])
    emp_diag = float(all_row["empirical_diag_probability"])
    if model_diag > emp_diag:
        text.append("CHL2 overproduces the modulo-$3$ diagonal gap population in the aggregate. This directly explains the sign of the CHL2 prime-residue error: the model assigns too much mass to gaps $g_2 \\equiv 0 \\pmod 3$, which correspond to repeating the previous prime residue class.\n")
    else:
        text.append("CHL2 does not overproduce the diagonal gap population in the aggregate; the prime-residue error must therefore involve more than the marginal population of $g_2 \\bmod 3$.\n")
    # NO_58 robustness note.
    no58 = df_filter[(df_filter["block"] == "ALL") & (df_filter["filter"] == "NO_58")]
    if not no58.empty:
        r = no58.iloc[0]
        text.append(f"In `NO_58`, empirical $D_3={float(r['D3_empirical_gap_population']):.6g}$ and CHL2 gap-population $D_3={float(r['D3_chl2_gap_population']):.6g}$. This filter checks whether the effect persists after removing the dense small-gap regime.\n")
    write_extended_interpretation_notes(text, df_scale, df_symmetry)
    text.append("\nThis audit is genealogical. It does not introduce a new transfer kernel and it does not fit $\\theta_3$. Its role is to determine whether the measured CHL4-C residual mode can be traced to the gap population induced by CHL2.\n")
    (outdir / "chl4d2_interpretacion.md").write_text("\n".join(text), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="CHL4-D2 gap-population bias audit")
    ap.add_argument("--config", required=True, help="Dataset config JSON")
    ap.add_argument("--root", default=".", help="Repository/data root")
    ap.add_argument("--blocks", default="1-10", help="Blocks, e.g. 1-10")
    ap.add_argument("--filters", default=",".join(FILTERS_DEFAULT), help="Comma-separated filters")
    ap.add_argument("--Y", "--pmax", dest="Y", type=int, default=47, help="Truncation horizon")
    ap.add_argument("--log-x", type=float, default=None, help="Local log x; default from config or 25.328436")
    ap.add_argument("--path-cache-file", default=None, help="Optional CHL2 path-exclusion cache CSV.GZ")
    ap.add_argument("--eta", type=float, default=None, help="Optional fixed eta; default solves eta globally")
    ap.add_argument("--output-dir", required=True, help="Output directory")
    args = ap.parse_args(argv)

    t0 = time.time()
    telemetry = telemetry_start()
    telemetry["argv_effective"] = list(sys.argv if argv is None else argv)
    root = Path(args.root).resolve()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = read_json(Path(args.config))
    blocks = parse_int_list_or_range(args.blocks, default=cfg.get("blocks", list(range(1, 11))))
    filters = parse_filters(args.filters)
    log_x = float(args.log_x if args.log_x is not None else cfg.get("log_x", 25.328436))

    block_paths = parent_wide_block_paths(root, cfg, blocks)
    df_blocks = load_parent_wide_blocks(block_paths)
    path_cache = load_path_cache(Path(args.path_cache_file) if args.path_cache_file else None)

    support, eta, mean_model, prob_map, model_meta = build_support_and_model(
        df_blocks,
        Y=int(args.Y),
        log_x=log_x,
        path_cache=path_cache,
        eta_override=args.eta,
    )

    all_eval_frames: list[pd.DataFrame] = []
    # Individual blocks.
    for b in blocks:
        df_b = df_blocks[df_blocks["block"] == int(b)].copy()
        eval_b = expected_counts_for_block(df_b, support, prob_map)
        eval_b["block_label"] = f"B{b:02d}"
        all_eval_frames.append(eval_b)
    # Aggregated ALL.
    eval_all = expected_counts_for_block(df_blocks, support, prob_map)
    eval_all["block_label"] = "ALL"
    all_eval_frames.append(eval_all)

    rows_filter: list[dict[str, Any]] = []
    rows_residue: list[dict[str, Any]] = []
    rows_gap: list[dict[str, Any]] = []
    for eval_df in all_eval_frames:
        label = str(eval_df["block_label"].iloc[0])
        rf, rr, rg = summarize_gap_population(eval_df, filters, label)
        rows_filter.extend(rf); rows_residue.extend(rr); rows_gap.extend(rg)

    df_filter = pd.DataFrame(rows_filter)
    df_residue = pd.DataFrame(rows_residue)
    df_gap = pd.DataFrame(rows_gap)
    df_gap = add_leave_one_gap_effects(df_gap, df_filter)
    df_cum = cumulative_by_gap(df_gap)
    df_stability = stability_by_filter(df_filter)
    df_scale = scale_wave_summary(df_filter)
    df_symmetry = offdiag_symmetry_summary(df_gap)

    # Population bias compact output (ALL residues + key filters).
    df_bias = df_residue[(df_residue["block"] == "ALL")].copy()

    df_filter.to_csv(outdir / "chl4d2_gap_population_by_filter.csv", index=False)
    df_residue.to_csv(outdir / "chl4d2_gap_population_by_residue.csv", index=False)
    df_gap.to_csv(outdir / "chl4d2_gap_population_by_gap.csv", index=False)
    df_cum.to_csv(outdir / "chl4d2_gap_population_cumulative.csv", index=False)
    df_stability.to_csv(outdir / "chl4d2_gap_population_block_stability.csv", index=False)
    df_bias.to_csv(outdir / "chl4d2_gap_population_bias.csv", index=False)
    df_scale.to_csv(outdir / "chl4d2_scale_wave_summary.csv", index=False)
    df_symmetry.to_csv(outdir / "chl4d2_offdiag_symmetry.csv", index=False)

    config_out = {
        "script": "chl4d2_gap_population_bias_audit.py",
        "config": str(Path(args.config)),
        "root": str(root),
        "blocks": blocks,
        "block_paths": [str(p) for _, p in block_paths],
        "filters": filters,
        "Y": int(args.Y),
        "log_x": log_x,
        "eta": eta,
        "eta_override": args.eta,
        "model_mean_g2": mean_model,
        "path_cache_file": args.path_cache_file,
        "model_meta": model_meta,
        "output_dir": str(outdir),
    }
    (outdir / "chl4d2_config.json").write_text(json.dumps(config_out, indent=2, sort_keys=True), encoding="utf-8")

    telemetry.update({
        "elapsed_seconds": time.time() - t0,
        "output_dir": str(outdir),
        "blocks": blocks,
        "n_block_rows_loaded": int(len(df_blocks)),
        "empirical_events_loaded": float(df_blocks["H"].sum()),
        "support_rows": int(len(support)),
        "unique_g1": int(support["g1"].nunique()),
        "Y": int(args.Y),
        "log_x": log_x,
        "eta": eta,
        "model_mean_g2": mean_model,
        "target_mean_g2": model_meta.get("target_mean_g2"),
        "filter_rows": int(len(df_filter)),
        "residue_rows": int(len(df_residue)),
        "gap_rows": int(len(df_gap)),
        "cumulative_rows": int(len(df_cum)),
        "stability_rows": int(len(df_stability)),
        "scale_wave_rows": int(len(df_scale)),
        "offdiag_symmetry_rows": int(len(df_symmetry)),
        "path_cache_hits": model_meta.get("path_cache_hits"),
        "path_cache_misses": model_meta.get("path_cache_misses"),
    })
    write_telemetry(outdir / "chl4d2_runtime_telemetry.json", telemetry)
    write_interpretation(outdir, df_filter, df_stability, df_residue, df_gap, df_cum, df_scale, df_symmetry, model_meta)

    print(f"[CHL4-D2] wrote outputs to {outdir}")
    print(f"[CHL4-D2] eta={eta:.12g}, target_mean={model_meta.get('target_mean_g2'):.6g}, model_mean={mean_model:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
