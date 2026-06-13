#!/usr/bin/env python3
"""
CHL4-D5: Orientation-Lift Oliver--Soundararajan replacement diagnostic.

This script replaces the previous direct CHL2-induced prime-residue transition
matrix diagnostic with an orientation-lift construction.

Given a marginal distribution of CHL2 next-gap residues

    p_r = P_CHL2(g mod q = r),

we construct the induced transition matrix on reduced residues by distributing
p_r over the valid oriented edges b -> a with a = b+r mod q.  If

    N_r(q) = #{b in (Z/qZ)^*: b+r in (Z/qZ)^*},

then each valid edge receives mass p_r / N_r(q), followed by row normalization.
For q=3 this gives the critical p_0/2 diagonal branch.

The script compares this orientation-lifted CHL2 matrix with the empirical
absolute prime-residue transition matrix.  It can either:

  1. read an existing D4 orientation-lift CSV; or
  2. recompute the CHL2 gap-residue population from parent-wide blocks.

No parameters are fitted.
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

try:
    from chl_kernel import CHLKernel
except Exception:  # pragma: no cover
    CHLKernel = None  # type: ignore


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


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


def canonical_column(df: pd.DataFrame, candidates: Sequence[str], target: str) -> Dict[str, str]:
    for c in candidates:
        if c in df.columns:
            return {c: target}
    return {}


def safe_log_ratio(num: float, den: float, eps: float = 1e-15) -> float:
    return float(math.log((num + eps) / (den + eps)))


# ---------------------------------------------------------------------------
# Orientation lift
# ---------------------------------------------------------------------------


def transition_from_gap_residue_population(
    probs_by_residue: Mapping[int, float],
    q: int,
    *,
    lift_kind: str = "orientation",
) -> pd.DataFrame:
    """Build T_q(b,a) from a marginal gap-residue population.

    ``orientation`` distributes each mass p_r over all valid oriented edges.
    ``naive_invalid`` deliberately assigns p_r to every valid edge and is kept
    only as a negative control.
    """
    rr = reduced_residues(q)
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
            for a in rr:
                rows.append({"q": q, "from_residue": b, "to_residue": a, "probability": 1.0 / len(rr)})
        else:
            for a in rr:
                rows.append({"q": q, "from_residue": b, "to_residue": a, "probability": edge_mass[(b, a)] / z})
    return pd.DataFrame(rows)


def matrix_from_rows(df: pd.DataFrame, q: int, prob_col: str) -> Tuple[List[int], np.ndarray]:
    rr = reduced_residues(q)
    idx = {r: i for i, r in enumerate(rr)}
    M = np.zeros((len(rr), len(rr)), dtype=float)
    for row in df.itertuples(index=False):
        b = int(getattr(row, "from_residue"))
        a = int(getattr(row, "to_residue"))
        if b in idx and a in idx:
            M[idx[b], idx[a]] += float(getattr(row, prob_col))
    for i in range(M.shape[0]):
        s = float(M[i].sum())
        if s > 0:
            M[i] /= s
        else:
            M[i] = 1.0 / len(rr)
    return rr, M


def diagonal_probability(M: np.ndarray) -> float:
    return float(np.mean(np.diag(M)))


def row_cosine_weighted(A: np.ndarray, B: np.ndarray, weights: np.ndarray | None = None) -> float:
    vals = []
    ws = []
    for i in range(A.shape[0]):
        na = float(np.linalg.norm(A[i]))
        nb = float(np.linalg.norm(B[i]))
        val = 0.0 if na == 0 or nb == 0 else float(np.dot(A[i], B[i]) / (na * nb))
        vals.append(val)
        ws.append(1.0 if weights is None else float(weights[i]))
    sw = sum(ws)
    return float(sum(v * w for v, w in zip(vals, ws)) / sw) if sw > 0 else float("nan")


def kl_weighted(A: np.ndarray, B: np.ndarray, weights: np.ndarray | None = None, eps: float = 1e-15) -> float:
    A2 = np.clip(A, eps, 1.0)
    B2 = np.clip(B, eps, 1.0)
    per = np.sum(A2 * (np.log(A2) - np.log(B2)), axis=1)
    if weights is None:
        return float(np.mean(per))
    sw = float(np.sum(weights))
    return float(np.sum(per * weights) / sw) if sw > 0 else float("nan")


def l1_weighted(A: np.ndarray, B: np.ndarray, weights: np.ndarray | None = None) -> float:
    per = np.sum(np.abs(A - B), axis=1)
    if weights is None:
        return float(np.mean(per))
    sw = float(np.sum(weights))
    return float(np.sum(per * weights) / sw) if sw > 0 else float("nan")


def spectral_gap(M: np.ndarray) -> float:
    vals = np.linalg.eigvals(M)
    vals = np.sort(np.abs(vals))[::-1]
    return 1.0 if len(vals) < 2 else float(1.0 - vals[1])


def d3_log_odds(M: np.ndarray) -> float:
    if M.shape != (2, 2):
        return float("nan")
    eps = 1e-15
    return float(math.log(((M[0, 0] + eps) * (M[1, 1] + eps)) / ((M[0, 1] + eps) * (M[1, 0] + eps))))


def pearson_chi2_from_counts(emp_counts: np.ndarray, row_counts: np.ndarray, model_probs: np.ndarray) -> Tuple[float, int, float]:
    E = row_counts[:, None] * model_probs
    mask = E > 0
    chi2 = float(np.sum(((emp_counts[mask] - E[mask]) ** 2) / E[mask]))
    df = int(np.sum(mask) - emp_counts.shape[0])  # row-normalized degrees, rough diagnostic
    n = float(np.sum(emp_counts))
    return chi2, df, chi2 / n if n > 0 else float("nan")


# ---------------------------------------------------------------------------
# Flexible input readers
# ---------------------------------------------------------------------------


def normalize_empirical_direct(path: str | Path) -> pd.DataFrame:
    """Read direct empirical prime-residue matrix with counts and probabilities."""
    df = pd.read_csv(path)
    ren: Dict[str, str] = {}
    ren.update(canonical_column(df, ["block", "window", "label"], "block"))
    ren.update(canonical_column(df, ["q", "mod", "modulus"], "q"))
    ren.update(canonical_column(df, ["from_residue", "from_residue_b", "from", "b", "prev_residue"], "from_residue"))
    ren.update(canonical_column(df, ["to_residue", "to_residue_a", "to", "a", "next_residue"], "to_residue"))
    ren.update(canonical_column(df, ["empirical_probability", "probability", "empirical_prob", "prob"], "empirical_probability"))
    ren.update(canonical_column(df, ["empirical_count", "count", "observed_count"], "empirical_count"))
    ren.update(canonical_column(df, ["row_count", "from_count"], "row_count"))
    out = df.rename(columns=ren)
    req = {"block", "q", "from_residue", "to_residue", "empirical_probability"}
    missing = req - set(out.columns)
    if missing:
        raise ValueError(f"{path} missing required empirical matrix columns: {sorted(missing)}")
    if "row_count" not in out.columns:
        # Reconstruct a pseudo-row count from empirical_count if available.
        if "empirical_count" in out.columns:
            out["row_count"] = out.groupby(["block", "q", "from_residue"])["empirical_count"].transform("sum")
        else:
            out["row_count"] = 1.0
            out["empirical_count"] = out["empirical_probability"]
    if "empirical_count" not in out.columns:
        out["empirical_count"] = out["row_count"].astype(float) * out["empirical_probability"].astype(float)
    return out[["block", "q", "from_residue", "to_residue", "empirical_count", "row_count", "empirical_probability"]].copy()


def normalize_old_model_direct(path: str | Path) -> pd.DataFrame:
    """Read an optional old CHL2 direct OS model matrix."""
    df = pd.read_csv(path)
    ren: Dict[str, str] = {}
    ren.update(canonical_column(df, ["block", "window", "label"], "block"))
    ren.update(canonical_column(df, ["q", "mod", "modulus"], "q"))
    ren.update(canonical_column(df, ["from_residue", "from_residue_b", "from", "b", "prev_residue"], "from_residue"))
    ren.update(canonical_column(df, ["to_residue", "to_residue_a", "to", "a", "next_residue"], "to_residue"))
    ren.update(canonical_column(df, ["model_probability", "probability", "model_prob", "prob"], "model_probability"))
    out = df.rename(columns=ren)
    req = {"block", "q", "from_residue", "to_residue", "model_probability"}
    missing = req - set(out.columns)
    if missing:
        raise ValueError(f"{path} missing required old model matrix columns: {sorted(missing)}")
    return out[["block", "q", "from_residue", "to_residue", "model_probability"]].copy()


def read_d4_lift_matrix(path: str | Path, *, source: str = "chl2_gap_population", lift_kind: str = "orientation") -> pd.DataFrame:
    df = pd.read_csv(path)
    ren: Dict[str, str] = {}
    ren.update(canonical_column(df, ["block"], "block"))
    ren.update(canonical_column(df, ["filter"], "filter"))
    ren.update(canonical_column(df, ["source"], "source"))
    ren.update(canonical_column(df, ["lift_kind"], "lift_kind"))
    ren.update(canonical_column(df, ["q"], "q"))
    ren.update(canonical_column(df, ["from_residue", "from_residue_b"], "from_residue"))
    ren.update(canonical_column(df, ["to_residue", "to_residue_a"], "to_residue"))
    ren.update(canonical_column(df, ["probability", "model_probability"], "model_probability"))
    out = df.rename(columns=ren)
    req = {"block", "q", "from_residue", "to_residue", "model_probability"}
    missing = req - set(out.columns)
    if missing:
        raise ValueError(f"{path} missing required lift matrix columns: {sorted(missing)}")
    if "source" in out.columns:
        out = out[out["source"].astype(str) == source]
    if "lift_kind" in out.columns:
        out = out[out["lift_kind"].astype(str) == lift_kind]
    if "filter" in out.columns:
        out = out[out["filter"].astype(str) == "ALL"]
    if out.empty:
        raise ValueError(f"No rows left after filtering {path} for source={source}, lift_kind={lift_kind}, filter=ALL")
    return out[["block", "q", "from_residue", "to_residue", "model_probability"]].copy()


# ---------------------------------------------------------------------------
# Recompute CHL2 gap-residue populations from blocks if no D4 lift is supplied
# ---------------------------------------------------------------------------


FILTERS = {"ALL": lambda m: np.ones_like(m, dtype=bool)}


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_block_path(root: Path, cfg: dict, block: int) -> Path:
    input_dir = cfg.get("input_dir", "")
    blocks_dir = cfg.get("blocks_dir", "blocks")
    block_glob = cfg.get("block_glob", "")
    candidates: List[Path] = []
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
        root / f"data/ds1_1e11_w2e9_g2400/blocks/parent_wide_B{block:02d}.csv.gz",
    ])
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Could not resolve block path for B%02d. Tried:\n%s" % (block, "\n".join(map(str, candidates))))


def read_block(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"g1", "g2", "H"}.issubset(df.columns):
        raise ValueError(f"{path} needs g1,g2,H columns")
    if "max_g" not in df.columns:
        df["max_g"] = np.maximum(df["g1"].to_numpy(), df["g2"].to_numpy())
    return df


def load_path_cache(path: Optional[str | Path]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    if "omega_path_exclusion" in df.columns:
        return df[["g1", "g2", "omega_path_exclusion"]].rename(columns={"omega_path_exclusion": "omega_path"}).drop_duplicates(["g1", "g2"])
    if "omega_path" in df.columns:
        return df[["g1", "g2", "omega_path"]].drop_duplicates(["g1", "g2"])
    if "logE_path_exclusion" in df.columns:
        out = df[["g1", "g2", "logE_path_exclusion"]].copy()
        out["omega_path"] = -out["logE_path_exclusion"].astype(float)
        return out[["g1", "g2", "omega_path"]].drop_duplicates(["g1", "g2"])
    raise ValueError("path cache needs omega_path_exclusion, omega_path, or logE_path_exclusion")


def enrich_chl2_terms(df: pd.DataFrame, Y: int, log_x: float, cache: Optional[pd.DataFrame]) -> pd.DataFrame:
    out = df.copy()
    if cache is not None:
        out = out.merge(cache, on=["g1", "g2"], how="left")
    else:
        out["omega_path"] = np.nan
    need = out.loc[out["omega_path"].isna(), ["g1", "g2"]].drop_duplicates()
    if len(need) > 0:
        if CHLKernel is None:
            raise RuntimeError("CHLKernel unavailable and path cache incomplete")
        kernel = CHLKernel(Y=Y, log_x=log_x)
        vals = [(int(r.g1), int(r.g2), float(kernel.omega_path(int(r.g1), int(r.g2)))) for r in need.itertuples(index=False)]
        fill = pd.DataFrame(vals, columns=["g1", "g2", "omega_path_fill"])
        out = out.merge(fill, on=["g1", "g2"], how="left")
        out["omega_path"] = out["omega_path"].fillna(out["omega_path_fill"])
        out = out.drop(columns=["omega_path_fill"])
    if "log_R" not in out.columns:
        if CHLKernel is None:
            raise RuntimeError("CHLKernel unavailable; cannot compute log_R")
        kernel = CHLKernel(Y=Y, log_x=log_x)
        pairs = out[["g1", "g2"]].drop_duplicates()
        vals = [(int(r.g1), int(r.g2), float(kernel.log_R(int(r.g1), int(r.g2)))) for r in pairs.itertuples(index=False)]
        lr = pd.DataFrame(vals, columns=["g1", "g2", "log_R"])
        out = out.merge(lr, on=["g1", "g2"], how="left")
    out["base_logw_chl2"] = out["log_R"].astype(float) - out["omega_path"].astype(float)
    return out


def conditional_masses(df: pd.DataFrame, eta: float) -> np.ndarray:
    g1 = df["g1"].to_numpy(dtype=np.int64)
    g2 = df["g2"].to_numpy(dtype=float)
    base = df["base_logw_chl2"].to_numpy(dtype=float)
    H = df["H"].to_numpy(dtype=float)
    masses = np.zeros(len(df), dtype=float)
    state_mass = pd.Series(H).groupby(g1).sum().to_dict()
    order = np.argsort(g1)
    g1s = g1[order]
    starts = np.r_[0, np.flatnonzero(np.diff(g1s)) + 1]
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


def model_mean_for_eta(df: pd.DataFrame, eta: float) -> float:
    m = conditional_masses(df, eta)
    tot = float(np.sum(m))
    return float(np.sum(m * df["g2"].to_numpy(dtype=float)) / tot) if tot > 0 else float("nan")


def solve_eta(df: pd.DataFrame, target: float) -> float:
    lo, hi = -0.1, 0.1
    for _ in range(40):
        mlo = model_mean_for_eta(df, lo)
        mhi = model_mean_for_eta(df, hi)
        if mlo <= target <= mhi:
            break
        if target < mlo:
            hi = lo; lo *= 2
        else:
            lo = hi; hi *= 2
    for _ in range(80):
        mid = (lo + hi) / 2
        if model_mean_for_eta(df, mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def model_gap_residue_population(df: pd.DataFrame, q: int) -> Tuple[Dict[int, float], float]:
    H = df["H"].to_numpy(dtype=float)
    target = float(np.sum(H * df["g2"].to_numpy(dtype=float)) / np.sum(H))
    eta = solve_eta(df, target)
    masses = conditional_masses(df, eta)
    total = float(np.sum(masses))
    residues = df["g2"].to_numpy(dtype=np.int64) % q
    probs = {r: 0.0 for r in range(q)}
    if total > 0:
        for r in range(q):
            probs[r] = float(np.sum(masses[residues == r]) / total)
    return probs, eta


def compute_model_lift_from_blocks(config_path: str, root: str, blocks: Sequence[int], mods: Sequence[int], Y: int, log_x: float, path_cache_file: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rootp = Path(root)
    cfg = load_json(config_path)
    cache = load_path_cache(path_cache_file)
    lift_rows: List[pd.DataFrame] = []
    pop_rows: List[dict] = []
    event_rows: List[dict] = []
    for b in blocks:
        label = f"B{b:02d}"
        df = enrich_chl2_terms(read_block(resolve_block_path(rootp, cfg, b)), Y, log_x, cache)
        for q in mods:
            probs, eta = model_gap_residue_population(df, q)
            events = float(df["H"].sum())
            event_rows.append({"block": label, "q": q, "events": events})
            for r in range(q):
                pop_rows.append({"block": label, "q": q, "source": "chl2_gap_population", "gap_residue": r, "probability": probs.get(r, 0.0)})
            mat = transition_from_gap_residue_population(probs, q, lift_kind="orientation")
            mat.insert(0, "block", label)
            mat.insert(1, "source", "chl2_gap_population")
            mat.insert(2, "lift_kind", "orientation")
            lift_rows.append(mat)
    pop = pd.DataFrame(pop_rows)
    lift = pd.concat(lift_rows, ignore_index=True)
    # ALL aggregate, weighted by block events.
    if pop_rows:
        ev = pd.DataFrame(event_rows)
        pp = pop.merge(ev, on=["block", "q"], how="left")
        pp["weighted"] = pp["probability"] * pp["events"]
        agg = pp.groupby(["q", "source", "gap_residue"], as_index=False).agg(weighted=("weighted", "sum"), events=("events", "sum"))
        agg["probability"] = agg["weighted"] / agg["events"]
        agg["block"] = "ALL"
        pop = pd.concat([pop, agg[["block", "q", "source", "gap_residue", "probability"]]], ignore_index=True)
        for (q, source), sub in agg.groupby(["q", "source"]):
            probs = {int(r.gap_residue): float(r.probability) for r in sub.itertuples(index=False)}
            mat = transition_from_gap_residue_population(probs, int(q), lift_kind="orientation")
            mat.insert(0, "block", "ALL")
            mat.insert(1, "source", source)
            mat.insert(2, "lift_kind", "orientation")
            lift = pd.concat([lift, mat], ignore_index=True)
    lift = lift.rename(columns={"probability": "model_probability"})
    return lift, pop


# ---------------------------------------------------------------------------
# Diagnostics and outputs
# ---------------------------------------------------------------------------


def combine_empirical_and_model(emp: pd.DataFrame, model: pd.DataFrame, mods: Sequence[int], blocks: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if blocks is not None:
        emp = emp[emp["block"].astype(str).isin([str(b) for b in blocks])]
        model = model[model["block"].astype(str).isin([str(b) for b in blocks])]
    rows = []
    for (block, q), esub in emp.groupby(["block", "q"]):
        q = int(q)
        if q not in mods:
            continue
        msub = model[(model["block"].astype(str) == str(block)) & (model["q"].astype(int) == q)]
        if msub.empty:
            continue
        # Merge by cells.
        merged = esub.merge(msub[["q", "from_residue", "to_residue", "model_probability"]], on=["q", "from_residue", "to_residue"], how="left")
        if merged["model_probability"].isna().any():
            raise ValueError(f"missing model probability for block={block}, q={q}")
        for r in merged.itertuples(index=False):
            row_count = float(r.row_count)
            mp = float(r.model_probability)
            rows.append({
                "block": block,
                "q": q,
                "from_residue": int(r.from_residue),
                "to_residue": int(r.to_residue),
                "row_count": row_count,
                "empirical_count": float(r.empirical_count),
                "empirical_probability": float(r.empirical_probability),
                "model_probability": mp,
                "model_expected_count": row_count * mp,
                "is_diagonal": int(r.from_residue == r.to_residue),
            })
    return pd.DataFrame(rows)


def summarize_transition(combined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (block, q), sub in combined.groupby(["block", "q"]):
        q = int(q)
        rr, Emp = matrix_from_rows(sub.rename(columns={"empirical_probability": "prob"}), q, "prob")
        _, Mod = matrix_from_rows(sub.rename(columns={"model_probability": "prob"}), q, "prob")
        # Row weights by empirical row count.
        row_weights = []
        for b in rr:
            row_weights.append(float(sub[sub["from_residue"] == b]["row_count"].iloc[0]))
        w = np.array(row_weights, dtype=float)
        emp_counts = np.zeros_like(Emp)
        for r in sub.itertuples(index=False):
            i = rr.index(int(r.from_residue)); j = rr.index(int(r.to_residue))
            emp_counts[i, j] = float(r.empirical_count)
        chi2, df, chi2n = pearson_chi2_from_counts(emp_counts, w, Mod)
        diag_emp = diagonal_probability(Emp)
        diag_model = diagonal_probability(Mod)
        uniform_diag = 1.0 / len(rr)
        wrong_sign = bool((diag_emp - uniform_diag) * (diag_model - uniform_diag) < 0)
        rows.append({
            "block": block,
            "q": q,
            "n_reduced_residues": len(rr),
            "row_cosine_weighted": row_cosine_weighted(Emp, Mod, w),
            "kl_empirical_to_model_weighted": kl_weighted(Emp, Mod, w),
            "l1_weighted": l1_weighted(Emp, Mod, w),
            "diagonal_probability_empirical": diag_emp,
            "diagonal_probability_model": diag_model,
            "uniform_diagonal_probability": uniform_diag,
            "diagonal_wrong_sign_vs_uniform": wrong_sign,
            "spectral_gap_empirical": spectral_gap(Emp),
            "spectral_gap_model": spectral_gap(Mod),
            "spectral_gap_abs_error": abs(spectral_gap(Emp) - spectral_gap(Mod)),
            "pearson_chi2": chi2,
            "pearson_chi2_df": df,
            "pearson_chi2_per_transition": chi2n,
            "D3_empirical": d3_log_odds(Emp),
            "D3_model": d3_log_odds(Mod),
            "D3_model_minus_empirical": d3_log_odds(Mod) - d3_log_odds(Emp) if q == 3 else float("nan"),
            "total_transitions": float(sub["empirical_count"].sum()),
        })
    return pd.DataFrame(rows)


def compare_with_old(old_model_path: Optional[str], emp: pd.DataFrame, new_model: pd.DataFrame, mods: Sequence[int]) -> pd.DataFrame:
    if not old_model_path:
        return pd.DataFrame()
    old = normalize_old_model_direct(old_model_path)
    old_combined = combine_empirical_and_model(emp, old, mods)
    new_combined = combine_empirical_and_model(emp, new_model, mods)
    old_sum = summarize_transition(old_combined)
    new_sum = summarize_transition(new_combined)
    rows = []
    for (block, q), nrow in new_sum.groupby(["block", "q"]):
        osub = old_sum[(old_sum["block"].astype(str) == str(block)) & (old_sum["q"].astype(int) == int(q))]
        if osub.empty:
            continue
        o = osub.iloc[0]
        n = nrow.iloc[0]
        rows.append({
            "block": block,
            "q": int(q),
            "old_kl": float(o["kl_empirical_to_model_weighted"]),
            "oriented_kl": float(n["kl_empirical_to_model_weighted"]),
            "delta_kl_old_minus_oriented": float(o["kl_empirical_to_model_weighted"] - n["kl_empirical_to_model_weighted"]),
            "old_l1": float(o["l1_weighted"]),
            "oriented_l1": float(n["l1_weighted"]),
            "old_diag_model": float(o["diagonal_probability_model"]),
            "oriented_diag_model": float(n["diagonal_probability_model"]),
            "emp_diag": float(n["diagonal_probability_empirical"]),
            "old_wrong_sign": bool(o["diagonal_wrong_sign_vs_uniform"]),
            "oriented_wrong_sign": bool(n["diagonal_wrong_sign_vs_uniform"]),
        })
    return pd.DataFrame(rows)


def write_interpretation(outdir: Path, summary: pd.DataFrame, old_vs_new: pd.DataFrame) -> None:
    lines: List[str] = []
    lines.append("# CHL4-D5 Orientation-Lift OS Replacement")
    lines.append("")
    lines.append("This audit replaces the previous direct CHL2-induced OS diagnostic with an orientation-lift construction from CHL2 gap-residue populations.")
    lines.append("")
    all_rows = summary[summary["block"].astype(str) == "ALL"] if "block" in summary.columns else summary
    if not all_rows.empty:
        cols = ["q", "row_cosine_weighted", "kl_empirical_to_model_weighted", "l1_weighted", "diagonal_probability_empirical", "diagonal_probability_model", "uniform_diagonal_probability", "diagonal_wrong_sign_vs_uniform", "pearson_chi2_per_transition", "spectral_gap_abs_error", "D3_empirical", "D3_model"]
        cols = [c for c in cols if c in all_rows.columns]
        lines.append("## Oriented OS summary, ALL")
        lines.append("")
        lines.append(all_rows[cols].to_markdown(index=False))
        lines.append("")
    q3 = all_rows[all_rows["q"].astype(int) == 3] if not all_rows.empty else pd.DataFrame()
    if not q3.empty:
        r = q3.iloc[0]
        lines.append("## $q=3$ checkpoint")
        lines.append("")
        lines.append(f"- Empirical diagonal probability: {float(r['diagonal_probability_empirical']):.6g}.")
        lines.append(f"- Oriented CHL2 diagonal probability: {float(r['diagonal_probability_model']):.6g}.")
        lines.append(f"- Uniform diagonal probability: {float(r['uniform_diagonal_probability']):.6g}.")
        lines.append(f"- Wrong-sign flag: `{bool(r['diagonal_wrong_sign_vs_uniform'])}`.")
        lines.append(f"- $D_3^{{emp}}={float(r['D3_empirical']):.6g}$ and $D_3^{{oriented}}={float(r['D3_model']):.6g}$.")
        lines.append("")
    if not old_vs_new.empty:
        all_old = old_vs_new[old_vs_new["block"].astype(str) == "ALL"]
        if not all_old.empty:
            lines.append("## Old direct OS versus orientation-lift OS, ALL")
            lines.append("")
            cols = ["q", "old_kl", "oriented_kl", "delta_kl_old_minus_oriented", "emp_diag", "old_diag_model", "oriented_diag_model", "old_wrong_sign", "oriented_wrong_sign"]
            lines.append(all_old[cols].to_markdown(index=False))
            lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("If the oriented matrix removes the $q=3$ wrong-sign flag while preserving or improving the diagnostics for $q=5,7,11,13$, the old OS discrepancy should be treated as an orientation-lift defect rather than a failure of the CHL2 gap kernel.")
    (outdir / "chl4d5_interpretacion.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="CHL4-D5 orientation-lift OS replacement diagnostic")
    ap.add_argument("--empirical-matrix-csv", required=True, help="Direct empirical prime-residue matrix CSV, preferably with counts.")
    ap.add_argument("--mods", default="3,5,7,11,13")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--mode", default="from-blocks", choices=["from-blocks", "from-d4-lift"], help="How to obtain the oriented CHL2 matrix.")
    ap.add_argument("--d4-lift-csv", default=None, help="Existing D4 orientation lift matrix CSV.")
    ap.add_argument("--old-model-matrix-csv", default=None, help="Optional old direct CHL2 OS model matrix for comparison.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--root", default=".")
    ap.add_argument("--blocks", default="1-10")
    ap.add_argument("--Y", type=int, default=47)
    ap.add_argument("--log-x", type=float, default=25.328436)
    ap.add_argument("--path-cache-file", default=None)
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    mods = parse_int_list(args.mods)

    empirical = normalize_empirical_direct(args.empirical_matrix_csv)
    empirical = empirical[empirical["q"].astype(int).isin(mods)].copy()

    if args.mode == "from-d4-lift":
        if not args.d4_lift_csv:
            raise ValueError("--d4-lift-csv is required for --mode from-d4-lift")
        model = read_d4_lift_matrix(args.d4_lift_csv, source="chl2_gap_population", lift_kind="orientation")
        gap_pop = pd.DataFrame()
    else:
        if not args.config:
            raise ValueError("--config is required for --mode from-blocks")
        model, gap_pop = compute_model_lift_from_blocks(
            config_path=args.config,
            root=args.root,
            blocks=parse_blocks(args.blocks),
            mods=mods,
            Y=args.Y,
            log_x=args.log_x,
            path_cache_file=args.path_cache_file,
        )
    model = model[model["q"].astype(int).isin(mods)].copy()

    combined = combine_empirical_and_model(empirical, model, mods)
    summary = summarize_transition(combined)
    old_vs_new = compare_with_old(args.old_model_matrix_csv, empirical, model, mods) if args.old_model_matrix_csv else pd.DataFrame()

    combined.to_csv(outdir / "chl2_os_oriented_prime_residue_transition_by_mod.csv", index=False)
    summary.to_csv(outdir / "chl2_os_oriented_prime_residue_summary.csv", index=False)
    if not old_vs_new.empty:
        old_vs_new.to_csv(outdir / "chl2_os_old_direct_vs_oriented.csv", index=False)
    if not gap_pop.empty:
        gap_pop.to_csv(outdir / "chl2_os_oriented_gap_residue_populations.csv", index=False)
    model.to_csv(outdir / "chl2_os_oriented_model_matrices.csv", index=False)
    write_interpretation(outdir, summary, old_vs_new)

    telemetry = {
        "script": "chl4d5_orientation_lift_os_replacement.py",
        "elapsed_seconds": time.perf_counter() - t0,
        "argv": sys.argv,
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "mode": args.mode,
        "mods": mods,
        "n_transition_rows": int(len(combined)),
        "n_summary_rows": int(len(summary)),
        "n_old_vs_new_rows": int(len(old_vs_new)),
        "output_files": [
            "chl2_os_oriented_prime_residue_transition_by_mod.csv",
            "chl2_os_oriented_prime_residue_summary.csv",
            "chl2_os_oriented_model_matrices.csv",
            "chl4d5_interpretacion.md",
        ],
    }
    (outdir / "chl4d5_runtime_telemetry.json").write_text(json.dumps(telemetry, indent=2), encoding="utf-8")
    (outdir / "chl4d5_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
