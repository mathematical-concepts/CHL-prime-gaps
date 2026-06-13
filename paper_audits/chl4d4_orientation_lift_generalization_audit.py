#!/usr/bin/env python3
"""
CHL4-D4: Orientation Lift Generalization Audit.

This script generalizes the q=3 orientation-lift lesson to arbitrary
moduli q.  It compares:

1. direct empirical prime-residue transfer matrices;
2. direct CHL2-induced transfer matrices from the OS diagnostic;
3. orientation-lift matrices constructed from empirical gap-residue
   populations;
4. orientation-lift matrices constructed from CHL2 gap-residue populations;
5. a deliberately invalid naive lift, used as a negative control.

The mathematically important point is that a global gap-residue mass p_r
must be distributed over all oriented reduced-residue edges compatible with
that residue.  If

    N_r(q) = #{b in (Z/qZ)^*: b+r in (Z/qZ)^*},

then the oriented edge mass for residue r is p_r / N_r(q), not p_r.
For q=3 this is exactly the rule p_0/2 for the diagonal branch.

The script is diagnostic: it does not modify CHL2 and does not introduce a
new fitted parameter.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from chl_kernel import CHLKernel
except Exception:  # pragma: no cover
    CHLKernel = None  # type: ignore


def parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def parse_blocks(text: str) -> List[int]:
    out: List[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def reduced_residues(q: int) -> List[int]:
    return [r for r in range(q) if gcd(r, q) == 1]


def residue_edge_count(q: int, r: int) -> int:
    """Number of reduced starting residues b for which b+r is reduced."""
    rr = reduced_residues(q)
    return sum(1 for b in rr if gcd((b + r) % q, q) == 1)


def transition_from_gap_residue_population(
    probs_by_residue: Mapping[int, float],
    q: int,
    *,
    lift_kind: str = "orientation",
) -> pd.DataFrame:
    """Build T_q(b,a) from a marginal distribution of gap residues.

    Parameters
    ----------
    probs_by_residue:
        Mapping r -> p_r for r modulo q. Missing residues are assigned 0.
    q:
        Modulus.
    lift_kind:
        ``orientation`` distributes p_r over all valid oriented reduced edges
        using p_r/N_r(q). ``naive_invalid`` adds p_r to each valid edge before
        row normalization and is kept only as a negative control.
    """
    rr = reduced_residues(q)
    if not rr:
        raise ValueError(f"modulus q={q} has no reduced residues")
    edge_mass: Dict[Tuple[int, int], float] = {(b, a): 0.0 for b in rr for a in rr}
    for r in range(q):
        p_r = float(probs_by_residue.get(r, 0.0))
        if p_r == 0.0:
            continue
        valid: List[Tuple[int, int]] = []
        for b in rr:
            a = (b + r) % q
            if gcd(a, q) == 1:
                valid.append((b, a))
        if not valid:
            continue
        if lift_kind == "orientation":
            share = p_r / float(len(valid))
        elif lift_kind == "naive_invalid":
            share = p_r
        else:
            raise ValueError(f"unknown lift_kind={lift_kind!r}")
        for edge in valid:
            edge_mass[edge] += share

    rows = []
    for b in rr:
        z = sum(edge_mass[(b, a)] for a in rr)
        if z <= 0:
            # Should not occur for q with some even gap residue support, but use
            # uniform fallback to keep the diagnostic total.
            prob = 1.0 / len(rr)
            for a in rr:
                rows.append({"q": q, "from_residue": b, "to_residue": a, "probability": prob})
        else:
            for a in rr:
                rows.append({"q": q, "from_residue": b, "to_residue": a, "probability": edge_mass[(b, a)] / z})
    return pd.DataFrame(rows)


def matrix_from_df(df: pd.DataFrame, q: int, prob_col: str = "probability") -> Tuple[List[int], np.ndarray]:
    """Return a row-normalized transition matrix from a transition DataFrame.

    The repository has used two equivalent naming conventions for residue
    columns over time:

    * ``from_residue`` / ``to_residue``;
    * ``from_residue_b`` / ``to_residue_a``.

    This function accepts either convention so that old and new audit outputs
    can be compared without manual CSV editing.
    """
    rr = reduced_residues(q)
    idx = {r: i for i, r in enumerate(rr)}
    from_col = "from_residue" if "from_residue" in df.columns else "from_residue_b"
    to_col = "to_residue" if "to_residue" in df.columns else "to_residue_a"
    if from_col not in df.columns or to_col not in df.columns:
        raise ValueError(
            "transition matrix DataFrame must contain either "
            "from_residue/to_residue or from_residue_b/to_residue_a columns"
        )
    M = np.zeros((len(rr), len(rr)), dtype=float)
    for row in df.itertuples(index=False):
        b = int(getattr(row, from_col))
        a = int(getattr(row, to_col))
        if b in idx and a in idx:
            M[idx[b], idx[a]] += float(getattr(row, prob_col))
    # Row normalize defensively.
    rs = M.sum(axis=1)
    for i, s in enumerate(rs):
        if s > 0:
            M[i, :] /= s
        else:
            M[i, :] = 1.0 / len(rr)
    return rr, M


def diagonal_probability(M: np.ndarray) -> float:
    return float(np.mean(np.diag(M)))


def row_cosine(A: np.ndarray, B: np.ndarray) -> float:
    vals = []
    for i in range(A.shape[0]):
        na = float(np.linalg.norm(A[i]))
        nb = float(np.linalg.norm(B[i]))
        if na == 0 or nb == 0:
            vals.append(0.0)
        else:
            vals.append(float(np.dot(A[i], B[i]) / (na * nb)))
    return float(np.mean(vals))


def kl_div(A: np.ndarray, B: np.ndarray, eps: float = 1e-15) -> float:
    A2 = np.clip(A, eps, 1.0)
    B2 = np.clip(B, eps, 1.0)
    return float(np.mean(np.sum(A2 * (np.log(A2) - np.log(B2)), axis=1)))


def l1_dist(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.mean(np.sum(np.abs(A - B), axis=1)))


def spectral_gap(M: np.ndarray) -> float:
    vals = np.linalg.eigvals(M)
    vals = np.sort(np.abs(vals))[::-1]
    if len(vals) < 2:
        return 1.0
    return float(1.0 - vals[1])


def d3_log_odds(M: np.ndarray) -> float:
    if M.shape != (2, 2):
        return float("nan")
    eps = 1e-15
    return float(math.log(((M[0, 0] + eps) * (M[1, 1] + eps)) / ((M[0, 1] + eps) * (M[1, 0] + eps))))


def matrix_metrics(direct: np.ndarray, lift: np.ndarray, q: int) -> dict:
    return {
        "q": q,
        "row_cosine_direct_lift": row_cosine(direct, lift),
        "kl_direct_to_lift": kl_div(direct, lift),
        "l1_direct_lift": l1_dist(direct, lift),
        "diagonal_probability_direct": diagonal_probability(direct),
        "diagonal_probability_lift": diagonal_probability(lift),
        "spectral_gap_direct": spectral_gap(direct),
        "spectral_gap_lift": spectral_gap(lift),
        "spectral_gap_abs_error": abs(spectral_gap(direct) - spectral_gap(lift)),
        "D3_direct": d3_log_odds(direct),
        "D3_lift": d3_log_odds(lift),
        "D3_direct_minus_lift": d3_log_odds(direct) - d3_log_odds(lift) if q == 3 else float("nan"),
    }


FILTERS = {
    "ALL": lambda m: np.ones_like(m, dtype=bool),
    "LOW_ONLY_LE58": lambda m: m <= 58,
    "MID_59_120": lambda m: (m >= 59) & (m <= 120),
    "MID_121_240": lambda m: (m >= 121) & (m <= 240),
    "MID_121_400": lambda m: (m >= 121) & (m <= 400),
    "NO_58": lambda m: m > 58,
    "NO_120": lambda m: m > 120,
    "NO_240": lambda m: m > 240,
}


def resolve_block_path(root: Path, cfg: dict, block: int) -> Path:
    input_dir = cfg.get("input_dir", "")
    blocks_dir = cfg.get("blocks_dir", "blocks")
    block_glob = cfg.get("block_glob", "")
    candidates = []
    if block_glob:
        try:
            candidates.append(root / input_dir / blocks_dir / block_glob.format(block=block))
        except Exception:
            pass
        try:
            candidates.append(root / input_dir / blocks_dir / block_glob.format(block=f"{block:02d}"))
        except Exception:
            pass
    candidates.extend([
        root / input_dir / blocks_dir / f"parent_wide_B{block:02d}.csv.gz",
        root / input_dir / blocks_dir / f"v46t12_ds_pilot_parent_wide_B{block:02d}.csv.gz",
        root / input_dir / f"blocks/parent_wide_B{block:02d}.csv.gz",
        root / f"data/ds1_1e11_w2e9_g2400/blocks/parent_wide_B{block:02d}.csv.gz",
    ])
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Could not resolve block path for block B%02d. Tried:\n%s" % (block, "\n".join(map(str, candidates))))


def read_block(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"g1", "g2", "H"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns {sorted(missing)}")
    if "max_g" not in df.columns:
        df["max_g"] = np.maximum(df["g1"].to_numpy(), df["g2"].to_numpy())
    return df


def load_path_cache(path: Optional[str | Path]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    if not {"g1", "g2"}.issubset(df.columns):
        raise ValueError("path cache must contain g1,g2")
    if "omega_path_exclusion" in df.columns:
        df = df[["g1", "g2", "omega_path_exclusion"]].rename(columns={"omega_path_exclusion": "omega_path"})
    elif "omega_path" in df.columns:
        df = df[["g1", "g2", "omega_path"]]
    elif "logE_path_exclusion" in df.columns:
        df = df[["g1", "g2", "logE_path_exclusion"]].copy()
        df["omega_path"] = -df["logE_path_exclusion"].astype(float)
        df = df[["g1", "g2", "omega_path"]]
    else:
        raise ValueError("path cache needs omega_path_exclusion, omega_path, or logE_path_exclusion")
    return df.drop_duplicates(["g1", "g2"])


def enrich_model_terms(df: pd.DataFrame, Y: int, log_x: float, cache: Optional[pd.DataFrame]) -> pd.DataFrame:
    out = df.copy()
    if cache is not None:
        out = out.merge(cache, on=["g1", "g2"], how="left")
    else:
        out["omega_path"] = np.nan
    need_pairs = out.loc[out["omega_path"].isna(), ["g1", "g2"]].drop_duplicates()
    if len(need_pairs) > 0:
        if CHLKernel is None:
            raise RuntimeError("CHLKernel is unavailable and path cache is incomplete")
        kernel = CHLKernel(Y=Y, log_x=log_x)
        vals = []
        for r in need_pairs.itertuples(index=False):
            vals.append((int(r.g1), int(r.g2), float(kernel.omega_path(int(r.g1), int(r.g2)))))
        fill = pd.DataFrame(vals, columns=["g1", "g2", "omega_path_fill"])
        out = out.merge(fill, on=["g1", "g2"], how="left")
        out["omega_path"] = out["omega_path"].fillna(out["omega_path_fill"])
        out = out.drop(columns=["omega_path_fill"])
    if "log_R" not in out.columns:
        if CHLKernel is None:
            raise RuntimeError("CHLKernel is unavailable; cannot compute log_R")
        kernel = CHLKernel(Y=Y, log_x=log_x)
        pairs = out[["g1", "g2"]].drop_duplicates()
        lr = []
        for r in pairs.itertuples(index=False):
            lr.append((int(r.g1), int(r.g2), float(kernel.log_R(int(r.g1), int(r.g2)))))
        lrdf = pd.DataFrame(lr, columns=["g1", "g2", "log_R"])
        out = out.merge(lrdf, on=["g1", "g2"], how="left")
    out["base_logw_chl2"] = out["log_R"].astype(float) - out["omega_path"].astype(float)
    return out


def _conditional_masses(df: pd.DataFrame, eta: float) -> np.ndarray:
    g1 = df["g1"].to_numpy(dtype=np.int64)
    g2 = df["g2"].to_numpy(dtype=float)
    base = df["base_logw_chl2"].to_numpy(dtype=float)
    H = df["H"].to_numpy(dtype=float)
    masses = np.zeros(len(df), dtype=float)
    # empirical mass by previous gap state
    state_mass = pd.Series(H).groupby(g1).sum().to_dict()
    # group row indices by g1
    order = np.argsort(g1)
    g1_sorted = g1[order]
    starts = np.r_[0, np.flatnonzero(np.diff(g1_sorted)) + 1]
    ends = np.r_[starts[1:], len(order)]
    for s, e in zip(starts, ends):
        idx = order[s:e]
        key = int(g1[idx[0]])
        lw = base[idx] + eta * g2[idx]
        mx = float(np.max(lw))
        w = np.exp(lw - mx)
        z = float(np.sum(w))
        if z > 0:
            masses[idx] = float(state_mass[key]) * w / z
    return masses


def _model_mean_for_eta(df: pd.DataFrame, eta: float) -> float:
    masses = _conditional_masses(df, eta)
    total = float(np.sum(masses))
    if total <= 0:
        return float("nan")
    return float(np.sum(masses * df["g2"].to_numpy(dtype=float)) / total)


def solve_eta_for_mean(df: pd.DataFrame, target_mean: float) -> float:
    if len(df) == 0:
        return 0.0
    lo, hi = -1e-1, 1e-1
    mlo = _model_mean_for_eta(df, lo)
    mhi = _model_mean_for_eta(df, hi)
    # Expand bracket.
    for _ in range(40):
        if mlo <= target_mean <= mhi:
            break
        if target_mean < mlo:
            hi = lo
            lo *= 2.0
            mlo = _model_mean_for_eta(df, lo)
        else:
            lo = hi
            hi *= 2.0
            mhi = _model_mean_for_eta(df, hi)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        mm = _model_mean_for_eta(df, mid)
        if mm < target_mean:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def empirical_gap_residue_probs(df: pd.DataFrame, q: int) -> Dict[int, float]:
    H = df["H"].to_numpy(dtype=float)
    r = (df["g2"].to_numpy(dtype=np.int64) % q)
    total = float(np.sum(H))
    probs = {i: 0.0 for i in range(q)}
    if total <= 0:
        return probs
    for i in range(q):
        probs[i] = float(np.sum(H[r == i]) / total)
    return probs


def model_gap_residue_probs(df: pd.DataFrame, q: int) -> Tuple[Dict[int, float], float, float]:
    if len(df) == 0:
        return ({i: 0.0 for i in range(q)}, 0.0, 0.0)
    H = df["H"].to_numpy(dtype=float)
    target = float(np.sum(H * df["g2"].to_numpy(dtype=float)) / np.sum(H))
    eta = solve_eta_for_mean(df, target)
    masses = _conditional_masses(df, eta)
    total = float(np.sum(masses))
    r = df["g2"].to_numpy(dtype=np.int64) % q
    probs = {i: 0.0 for i in range(q)}
    if total > 0:
        for i in range(q):
            probs[i] = float(np.sum(masses[r == i]) / total)
    return probs, eta, target


def matrix_rows_from_lift(block: str, filter_name: str, q: int, source: str, probs: Mapping[int, float], lift_kind: str) -> pd.DataFrame:
    mat = transition_from_gap_residue_population(probs, q, lift_kind=lift_kind)
    mat.insert(0, "source", source)
    mat.insert(0, "filter", filter_name)
    mat.insert(0, "block", block)
    mat.insert(3, "lift_kind", lift_kind)
    return mat


def normalize_direct_matrix_df(path: str | Path, source: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Flexible column normalization.
    colmap = {}
    for candidates, std in [
        (["block", "window", "label"], "block"),
        (["q", "mod", "modulus"], "q"),
        (["from_residue", "from_residue_b", "from", "b", "prev_residue"], "from_residue"),
        (["to_residue", "to_residue_a", "to", "a", "next_residue"], "to_residue"),
        (["probability", "empirical_probability", "model_probability", "empirical_prob", "model_prob", "prob"], "probability"),
    ]:
        for c in candidates:
            if c in df.columns:
                colmap[c] = std
                break
    df = df.rename(columns=colmap)
    req = {"block", "q", "from_residue", "to_residue", "probability"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns for direct matrix: {sorted(missing)}")
    out = df[["block", "q", "from_residue", "to_residue", "probability"]].copy()
    out["source"] = source
    out["filter"] = "ALL"
    return out


def compare_direct_to_lift(direct_df: pd.DataFrame, lift_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (source, block, filter_name, q, lift_kind), lsub in lift_df.groupby(["source", "block", "filter", "q", "lift_kind"]):
        dsub = direct_df[(direct_df["source"] == source) & (direct_df["block"].astype(str) == str(block)) & (direct_df["q"].astype(int) == int(q))]
        if dsub.empty:
            continue
        _, Mdir = matrix_from_df(dsub, int(q))
        _, Mlift = matrix_from_df(lsub, int(q))
        met = matrix_metrics(Mdir, Mlift, int(q))
        met.update({"source": source, "block": block, "filter": filter_name, "lift_kind": lift_kind})
        rows.append(met)
    return pd.DataFrame(rows)


def summarize_comparison(cmp_df: pd.DataFrame) -> pd.DataFrame:
    if cmp_df.empty:
        return pd.DataFrame()
    metrics = [
        "row_cosine_direct_lift", "kl_direct_to_lift", "l1_direct_lift",
        "diagonal_probability_direct", "diagonal_probability_lift",
        "spectral_gap_abs_error", "D3_direct", "D3_lift", "D3_direct_minus_lift",
    ]
    rows = []
    for (source, filter_name, q, lift_kind), sub in cmp_df.groupby(["source", "filter", "q", "lift_kind"]):
        row = {"source": source, "filter": filter_name, "q": int(q), "lift_kind": lift_kind, "n_blocks": int(sub["block"].nunique())}
        for m in metrics:
            if m in sub.columns:
                vals = pd.to_numeric(sub[m], errors="coerce")
                row[f"mean_{m}"] = float(vals.mean())
                row[f"std_{m}"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def write_interpretation(outdir: Path, summary: pd.DataFrame, cmp_df: pd.DataFrame, gap_pop: pd.DataFrame) -> None:
    lines = []
    lines.append("# CHL4-D4 Orientation Lift Generalization Audit")
    lines.append("")
    lines.append("This audit generalizes the orientation-lift rule from $q=3$ to arbitrary reduced-residue moduli.")
    lines.append("")
    lines.append("For each gap residue $r$, the correct oriented edge mass is $p_r/N_r(q)$, where $N_r(q)$ is the number of reduced starting residues $b$ such that $b+r$ is also reduced. The naive control uses $p_r$ on every valid edge and is expected to overcount residues with many valid orientations.")
    lines.append("")
    if not summary.empty:
        all_orientation = summary[(summary["filter"] == "ALL") & (summary["lift_kind"] == "orientation")]
        cols = ["source", "q", "n_blocks", "mean_row_cosine_direct_lift", "mean_kl_direct_to_lift", "mean_l1_direct_lift", "mean_diagonal_probability_direct", "mean_diagonal_probability_lift", "mean_D3_direct", "mean_D3_lift"]
        cols = [c for c in cols if c in all_orientation.columns]
        lines.append("## Direct matrices versus orientation lift, ALL")
        lines.append("")
        if not all_orientation.empty:
            lines.append(all_orientation[cols].to_markdown(index=False))
        else:
            lines.append("No ALL/orientation rows were generated.")
        lines.append("")
        naive = summary[(summary["filter"] == "ALL") & (summary["lift_kind"] == "naive_invalid")]
        if not naive.empty:
            lines.append("## Negative control: naive-invalid lift, ALL")
            lines.append("")
            cols2 = ["source", "q", "mean_diagonal_probability_lift", "mean_D3_lift", "mean_kl_direct_to_lift"]
            cols2 = [c for c in cols2 if c in naive.columns]
            lines.append(naive[cols2].to_markdown(index=False))
            lines.append("")
    # Focus q=3 if available.
    q3 = summary[(summary["q"] == 3) & (summary["filter"] == "ALL") & (summary["lift_kind"] == "orientation")]
    if not q3.empty:
        lines.append("## $q=3$ checkpoint")
        lines.append("")
        for row in q3.itertuples(index=False):
            lines.append(f"- `{row.source}`: mean $D_3$ direct = {getattr(row, 'mean_D3_direct'):.6g}, mean $D_3$ lift = {getattr(row, 'mean_D3_lift'):.6g}, mean direct-minus-lift = {getattr(row, 'mean_D3_direct_minus_lift'):.6g}.")
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("If the orientation lift agrees with the empirical matrix but the old CHL2 direct OS matrix does not, then the discrepancy is not a gap-population error. It is a row-wise lifting error. If the orientation lift also improves $q=5,7,11,13$, the same correction should replace the earlier direct OS induction rule in future diagnostics.")
    lines.append("")
    (outdir / "chl4d4_interpretacion.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="CHL4-D4 orientation lift generalization audit")
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--blocks", default="1-10")
    ap.add_argument("--mods", default="3,5,7,11,13")
    ap.add_argument("--Y", type=int, default=47)
    ap.add_argument("--log-x", type=float, default=25.328436)
    ap.add_argument("--path-cache-file", default=None)
    ap.add_argument("--empirical-matrix-csv", required=True)
    ap.add_argument("--model-matrix-csv", required=True)
    ap.add_argument("--filters", default="ALL,LOW_ONLY_LE58,MID_59_120,MID_121_240,MID_121_400,NO_58,NO_120,NO_240")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    root = Path(args.root)
    cfg = load_json(args.config)
    blocks = parse_blocks(args.blocks)
    mods = parse_int_list(args.mods)
    filters = [f.strip() for f in args.filters.split(",") if f.strip()]
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    path_cache = load_path_cache(args.path_cache_file)

    all_gap_pop_rows: List[dict] = []
    all_lift_rows: List[pd.DataFrame] = []
    block_diag_rows: List[dict] = []

    for block in blocks:
        b_label = f"B{block:02d}"
        bpath = resolve_block_path(root, cfg, block)
        df0 = read_block(bpath)
        df0 = enrich_model_terms(df0, args.Y, args.log_x, path_cache)
        for filter_name in filters:
            if filter_name not in FILTERS:
                raise ValueError(f"unknown filter {filter_name}")
            mask = FILTERS[filter_name](df0["max_g"].to_numpy())
            df = df0.loc[mask].copy()
            if len(df) == 0:
                continue
            for q in mods:
                emp_probs = empirical_gap_residue_probs(df, q)
                model_probs, eta, target_mean = model_gap_residue_probs(df, q)
                for source, probs in [("empirical_gap_population", emp_probs), ("chl2_gap_population", model_probs)]:
                    for r in range(q):
                        all_gap_pop_rows.append({
                            "block": b_label, "filter": filter_name, "q": q, "source": source,
                            "gap_residue": r, "probability": float(probs.get(r, 0.0)),
                        })
                    for lift_kind in ["orientation", "naive_invalid"]:
                        all_lift_rows.append(matrix_rows_from_lift(b_label, filter_name, q, source, probs, lift_kind))
                block_diag_rows.append({
                    "block": b_label, "filter": filter_name, "q": q,
                    "n_rows_support": int(len(df)), "empirical_events": float(df["H"].sum()),
                    "eta_model": eta, "target_mean_g2": target_mean,
                })

    # Aggregate ALL across blocks by summing probabilities weighted by events for gap pop.
    gap_pop = pd.DataFrame(all_gap_pop_rows)
    lift_df = pd.concat(all_lift_rows, ignore_index=True) if all_lift_rows else pd.DataFrame()

    # Construct ALL blocks if not already absent? Here B labels are blocks; direct matrices may include ALL.
    # We also add an ALL aggregate source over blocks for each filter/q/source.
    diag_df = pd.DataFrame(block_diag_rows)
    if not diag_df.empty:
        event_weights = diag_df[["block", "filter", "q", "empirical_events"]]
        gp = gap_pop.merge(event_weights, on=["block", "filter", "q"], how="left")
        gp["weighted_prob"] = gp["probability"] * gp["empirical_events"]
        agg = gp.groupby(["filter", "q", "source", "gap_residue"], as_index=False).agg(weighted_prob=("weighted_prob", "sum"), empirical_events=("empirical_events", "sum"))
        agg["probability"] = agg["weighted_prob"] / agg["empirical_events"]
        agg["block"] = "ALL"
        gap_pop = pd.concat([gap_pop, agg[["block", "filter", "q", "source", "gap_residue", "probability"]]], ignore_index=True)
        for rowkey, sub in agg.groupby(["filter", "q", "source"]):
            filter_name, q, source = rowkey
            probs = {int(r.gap_residue): float(r.probability) for r in sub.itertuples(index=False)}
            for lift_kind in ["orientation", "naive_invalid"]:
                all_lift_rows.append(matrix_rows_from_lift("ALL", filter_name, int(q), str(source), probs, lift_kind))
        lift_df = pd.concat(all_lift_rows, ignore_index=True) if all_lift_rows else pd.DataFrame()

    empirical_direct = normalize_direct_matrix_df(args.empirical_matrix_csv, "empirical_gap_population")
    model_direct = normalize_direct_matrix_df(args.model_matrix_csv, "chl2_gap_population")
    direct_df = pd.concat([empirical_direct, model_direct], ignore_index=True)
    cmp_df = compare_direct_to_lift(direct_df, lift_df)
    summary = summarize_comparison(cmp_df)

    gap_pop.to_csv(outdir / "chl4d4_gap_residue_populations.csv", index=False)
    lift_df.to_csv(outdir / "chl4d4_orientation_lift_matrices.csv", index=False)
    direct_df.to_csv(outdir / "chl4d4_direct_matrices_used.csv", index=False)
    cmp_df.to_csv(outdir / "chl4d4_direct_vs_orientation_lift.csv", index=False)
    summary.to_csv(outdir / "chl4d4_orientation_lift_summary.csv", index=False)
    pd.DataFrame(block_diag_rows).to_csv(outdir / "chl4d4_block_diagnostics.csv", index=False)
    write_interpretation(outdir, summary, cmp_df, gap_pop)

    telemetry = {
        "script": "chl4d4_orientation_lift_generalization_audit.py",
        "elapsed_seconds": time.perf_counter() - t0,
        "argv": sys.argv,
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "blocks": blocks,
        "mods": mods,
        "filters": filters,
        "Y": args.Y,
        "log_x": args.log_x,
        "n_gap_pop_rows": int(len(gap_pop)),
        "n_lift_rows": int(len(lift_df)),
        "n_comparison_rows": int(len(cmp_df)),
        "n_summary_rows": int(len(summary)),
    }
    (outdir / "chl4d4_runtime_telemetry.json").write_text(json.dumps(telemetry, indent=2), encoding="utf-8")
    (outdir / "chl4d4_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
