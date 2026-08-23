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
import hashlib
import json
import math
import os
import platform
import subprocess
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


def safe_to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    """Render Markdown when pandas' optional tabulate dependency is present."""
    try:
        return df.to_markdown(index=index)
    except Exception:
        return "```\n" + df.to_string(index=index) + "\n```"



def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(int(chunk_size)), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_provenance(path: str | Path, *, hash_content: bool) -> dict:
    """Return stable metadata for one input or script file."""
    p = Path(path)
    record: dict = {"path": str(p), "exists": bool(p.is_file())}
    if not p.is_file():
        return record
    stat = p.stat()
    record.update({
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(p) if hash_content else None,
    })
    return record


def git_provenance(repo_root: str | Path) -> dict:
    """Read commit and tracked-worktree status without requiring Git."""
    root = Path(repo_root)

    def run_git(*args: str) -> str | None:
        try:
            proc = subprocess.run(
                ["git", "-C", str(root), *args],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            return None
        if proc.returncode != 0:
            return None
        return proc.stdout.strip()

    commit = run_git("rev-parse", "HEAD")
    branch = run_git("rev-parse", "--abbrev-ref", "HEAD")
    describe = run_git("describe", "--tags", "--always", "--dirty")
    status = run_git("status", "--porcelain", "--untracked-files=no")
    tracked_changes = None if status is None else ([line for line in status.splitlines() if line] if status else [])
    return {
        "repo_root": str(root),
        "available": commit is not None,
        "commit": commit,
        "branch": branch,
        "describe": describe,
        "tracked_worktree_clean": None if tracked_changes is None else not bool(tracked_changes),
        "tracked_changes": tracked_changes,
    }


def build_provenance(
    *,
    repo_root: str | Path,
    input_paths: Mapping[str, str | Path | None],
    hash_inputs: bool,
) -> dict:
    """Build the release provenance payload embedded in JSON outputs."""
    inputs = {
        str(label): file_provenance(path, hash_content=bool(hash_inputs))
        for label, path in sorted(input_paths.items())
        if path not in (None, "")
    }
    return {
        "schema": "chl-release-provenance@1",
        "script": file_provenance(Path(__file__).resolve(), hash_content=True),
        "git": git_provenance(repo_root),
        "input_hashes_enabled": bool(hash_inputs),
        "inputs": inputs,
    }


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


def matrix_from_rows(
    df: pd.DataFrame,
    q: int,
    prob_col: str,
    *,
    zero_row_policy: str = "error",
) -> Tuple[List[int], np.ndarray]:
    """Build a reduced-residue matrix without silently repairing zero rows.

    ``zero_row_policy='uniform'`` is retained only for explicitly named
    synthetic controls. Scientific matrices use the default ``'error'``.
    """
    if zero_row_policy not in {"error", "uniform"}:
        raise ValueError(f"unknown zero_row_policy={zero_row_policy!r}")
    rr = reduced_residues(q)
    idx = {r: i for i, r in enumerate(rr)}
    M = np.zeros((len(rr), len(rr)), dtype=float)
    for row in df.itertuples(index=False):
        b = int(getattr(row, "from_residue"))
        a = int(getattr(row, "to_residue"))
        if b in idx and a in idx:
            value = float(getattr(row, prob_col))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"invalid matrix probability for q={q}, from_residue={b}, "
                    f"to_residue={a}: {value!r}"
                )
            M[idx[b], idx[a]] += value
    for i, b in enumerate(rr):
        s = float(M[i].sum())
        if s > 0:
            M[i] /= s
        elif zero_row_policy == "uniform":
            M[i] = 1.0 / len(rr)
        else:
            raise ValueError(f"zero-sum model row for q={q}, from_residue={b}")
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
    result = out[["block", "q", "from_residue", "to_residue", "model_probability"]].copy()
    result.attrs["matrix_support_audit"] = validate_probability_matrix_groups(
        result,
        prob_col="model_probability",
        label=f"old model input {path}",
        require_normalized=True,
    )
    return result


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


def model_gap_residue_populations(
    df: pd.DataFrame,
    mods: Sequence[int],
) -> Tuple[Dict[int, Dict[int, float]], float, float, float]:
    """Compute one anchored CHL2 mass distribution and project it modulo q.

    The terminal anchor ``eta`` is a property of the block/stratum, not of the
    diagnostic modulus.  It is therefore solved once and reused for every q.
    """
    H = df["H"].to_numpy(dtype=float)
    total_events = float(np.sum(H))
    if total_events <= 0:
        raise ValueError("block has non-positive total event mass")
    target = float(np.sum(H * df["g2"].to_numpy(dtype=float)) / total_events)
    eta = solve_eta(df, target)
    masses = conditional_masses(df, eta)
    total_mass = float(np.sum(masses))
    if total_mass <= 0:
        raise ValueError("CHL2 conditional masses sum to zero")
    g2 = df["g2"].to_numpy(dtype=np.int64)
    by_q: Dict[int, Dict[int, float]] = {}
    for q0 in mods:
        q = int(q0)
        residues = g2 % q
        probs = {r: float(np.sum(masses[residues == r]) / total_mass) for r in range(q)}
        z = float(sum(probs.values()))
        if not np.isfinite(z) or abs(z - 1.0) > 1e-10:
            raise ValueError(f"gap-residue population does not normalize for q={q}: {z}")
        by_q[q] = probs
    return by_q, float(eta), float(target), total_events


def compute_model_lift_from_blocks(
    config_path: str,
    root: str,
    blocks: Sequence[int],
    mods: Sequence[int],
    Y: int,
    log_x: float,
    path_cache_file: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rootp = Path(root)
    cfg = load_json(config_path)
    cache = load_path_cache(path_cache_file)
    lift_rows: List[pd.DataFrame] = []
    pop_rows: List[dict] = []
    event_rows: List[dict] = []
    for b in blocks:
        label = f"B{b:02d}"
        block_path = resolve_block_path(rootp, cfg, b)
        df = enrich_chl2_terms(read_block(block_path), Y, log_x, cache)
        probs_by_q, eta, target_mean, events = model_gap_residue_populations(df, mods)
        for q in mods:
            q = int(q)
            probs = probs_by_q[q]
            event_rows.append({
                "block": label,
                "q": q,
                "events": events,
                "eta": eta,
                "target_mean_g2": target_mean,
                "source_file": str(block_path),
            })
            for r in range(q):
                pop_rows.append({
                    "block": label,
                    "q": q,
                    "source": "chl2_gap_population",
                    "gap_residue": r,
                    "probability": probs.get(r, 0.0),
                    "events": events,
                    "eta": eta,
                    "target_mean_g2": target_mean,
                    "source_file": str(block_path),
                })
            mat = transition_from_gap_residue_population(probs, q, lift_kind="orientation")
            mat.insert(0, "block", label)
            mat.insert(1, "source", "chl2_gap_population")
            mat.insert(2, "lift_kind", "orientation")
            lift_rows.append(mat)

    if not lift_rows:
        raise ValueError("no orientation-lift matrices were produced")
    pop = pd.DataFrame(pop_rows)
    lift = pd.concat(lift_rows, ignore_index=True)

    # ALL aggregate: first aggregate the gap population by true block event mass,
    # then apply the orientation lift once to that global population.
    ev = pd.DataFrame(event_rows)
    pp = pop[["block", "q", "source", "gap_residue", "probability"]].merge(
        ev[["block", "q", "events"]], on=["block", "q"], how="left", validate="many_to_one"
    )
    pp["weighted"] = pp["probability"] * pp["events"]
    agg = pp.groupby(["q", "source", "gap_residue"], as_index=False).agg(
        weighted=("weighted", "sum"), events=("events", "sum")
    )
    agg["probability"] = agg["weighted"] / agg["events"]
    agg["block"] = "ALL"
    agg["eta"] = np.nan
    agg["target_mean_g2"] = np.nan
    agg["source_file"] = "event_weighted_B01-B10"
    pop = pd.concat([
        pop,
        agg[[
            "block", "q", "source", "gap_residue", "probability", "events",
            "eta", "target_mean_g2", "source_file",
        ]],
    ], ignore_index=True)

    all_lifts: List[pd.DataFrame] = []
    for (q, source), sub in agg.groupby(["q", "source"]):
        probs = {int(r.gap_residue): float(r.probability) for r in sub.itertuples(index=False)}
        mat = transition_from_gap_residue_population(probs, int(q), lift_kind="orientation")
        mat.insert(0, "block", "ALL")
        mat.insert(1, "source", source)
        mat.insert(2, "lift_kind", "orientation")
        all_lifts.append(mat)
    if all_lifts:
        lift = pd.concat([lift, *all_lifts], ignore_index=True)
    lift = lift.rename(columns={"probability": "model_probability"})
    return lift, pop

# ---------------------------------------------------------------------------
# Diagnostics and outputs
# ---------------------------------------------------------------------------


def _validate_complete_matrix_cells(df: pd.DataFrame, q: int, *, label: str) -> None:
    rr = reduced_residues(int(q))
    expected = {(b, a) for b in rr for a in rr}
    keys = list(zip(df["from_residue"].astype(int), df["to_residue"].astype(int)))
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate matrix cells for {label}, q={q}")
    actual = set(keys)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"incomplete matrix for {label}, q={q}; missing={missing[:8]}, extra={extra[:8]}")


def validate_probability_matrix_groups(
    df: pd.DataFrame,
    *,
    prob_col: str,
    label: str,
    require_normalized: bool = True,
    tolerance: float = 1e-10,
) -> dict:
    """Validate complete, finite, positive-row transition matrices by group."""
    required = {"block", "q", "from_residue", "to_residue", prob_col}
    missing_columns = sorted(required - set(df.columns))
    if missing_columns:
        raise ValueError(f"{label} missing matrix columns: {missing_columns}")

    rows_checked = 0
    groups_checked = 0
    row_sums_seen: List[float] = []
    zero_sum_rows: List[dict] = []
    for (block, q0), sub in df.groupby(["block", "q"], dropna=False):
        q = int(q0)
        group_label = f"{label}, block={block}"
        _validate_complete_matrix_cells(sub, q, label=group_label)
        work = sub.copy()
        probs = pd.to_numeric(work[prob_col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(probs).all():
            raise ValueError(f"non-finite model probabilities for {group_label}, q={q}")
        if (probs < 0).any():
            raise ValueError(f"negative model probabilities for {group_label}, q={q}")
        work["__prob"] = probs
        row_sums = work.groupby("from_residue", sort=True)["__prob"].sum()
        for residue, total in row_sums.items():
            total_f = float(total)
            row_sums_seen.append(total_f)
            if total_f <= 0.0:
                zero_sum_rows.append({
                    "block": str(block),
                    "q": q,
                    "from_residue": int(residue),
                })
        if zero_sum_rows:
            first = zero_sum_rows[0]
            raise ValueError(
                "zero-sum model row for "
                f"block={first['block']}, q={first['q']}, "
                f"from_residue={first['from_residue']}"
            )
        if require_normalized and not np.allclose(
            row_sums.to_numpy(dtype=float),
            1.0,
            rtol=0.0,
            atol=float(tolerance),
        ):
            bad = {
                int(residue): float(total)
                for residue, total in row_sums.items()
                if not math.isclose(float(total), 1.0, rel_tol=0.0, abs_tol=float(tolerance))
            }
            raise ValueError(
                f"model probability rows do not sum to one for {group_label}, q={q}: {bad}"
            )
        rows_checked += int(len(work))
        groups_checked += 1

    if groups_checked == 0:
        raise ValueError(f"{label} contains no matrix groups")
    return {
        "matrix_support_gate_pass": True,
        "matrix_groups_checked": int(groups_checked),
        "matrix_cells_checked": int(rows_checked),
        "zero_sum_model_rows": [],
        "model_row_sum_min": float(min(row_sums_seen)),
        "model_row_sum_max": float(max(row_sums_seen)),
    }


def legacy_table4_provenance_summary() -> dict:
    """Return the release status of the reconstructed v1.8 Table 4 artifact."""
    return {
        "status": "closed_legacy_implementation_artifact",
        "scientific_baseline": False,
        "historical_label": "naive_table4_legacy",
        "v2_control": "absolute_prime_os_previous_gap_conditioned",
        "q_3_5_7_source": "previous_gap_conditioned_direct_matrix",
        "q_11_13_source": "missing_modulus_zero_row_uniform_fallback",
        "documentation": "docs/NAIVE_TABLE4_LEGACY_PROVENANCE.md",
    }


def ensure_all_empirical(emp: pd.DataFrame) -> pd.DataFrame:
    """Append a count-exact ALL empirical matrix when only blocks are present."""
    out = emp.copy()
    if (out["block"].astype(str) == "ALL").any():
        return out
    base = out[out["block"].astype(str) != "ALL"].copy()
    agg = base.groupby(["q", "from_residue", "to_residue"], as_index=False).agg(
        empirical_count=("empirical_count", "sum")
    )
    row = agg.groupby(["q", "from_residue"])["empirical_count"].transform("sum")
    agg["row_count"] = row
    agg["empirical_probability"] = np.divide(
        agg["empirical_count"], row, out=np.zeros(len(agg), dtype=float), where=row.to_numpy() > 0
    )
    agg["block"] = "ALL"
    return pd.concat([
        out,
        agg[["block", "q", "from_residue", "to_residue", "empirical_count", "row_count", "empirical_probability"]],
    ], ignore_index=True)


def ensure_all_model(model: pd.DataFrame, empirical: pd.DataFrame) -> pd.DataFrame:
    """Append ALL model matrices using empirical row-mass weighting if absent."""
    out = model.copy()
    validate_probability_matrix_groups(
        out,
        prob_col="model_probability",
        label="model matrix",
        require_normalized=True,
    )
    if (out["block"].astype(str) == "ALL").any():
        return out
    emp_blocks = empirical[empirical["block"].astype(str) != "ALL"]
    row_counts = emp_blocks[["block", "q", "from_residue", "row_count"]].drop_duplicates()
    work = out.merge(row_counts, on=["block", "q", "from_residue"], how="left", validate="many_to_one")
    if work["row_count"].isna().any():
        bad = work.loc[work["row_count"].isna(), ["block", "q", "from_residue"]].drop_duplicates()
        raise ValueError(f"cannot aggregate model ALL; missing empirical row counts: {bad.head().to_dict('records')}")
    work["expected"] = work["model_probability"].astype(float) * work["row_count"].astype(float)
    agg = work.groupby(["q", "from_residue", "to_residue"], as_index=False).agg(expected=("expected", "sum"))
    z = agg.groupby(["q", "from_residue"])["expected"].transform("sum")
    if (z.to_numpy(dtype=float) <= 0.0).any():
        bad = agg.loc[z.to_numpy(dtype=float) <= 0.0, ["q", "from_residue"]].drop_duplicates()
        raise ValueError(
            f"cannot aggregate model ALL; zero-sum model rows: {bad.to_dict('records')}"
        )
    agg["model_probability"] = agg["expected"] / z
    agg["block"] = "ALL"
    all_rows = agg[["block", "q", "from_residue", "to_residue", "model_probability"]]
    validate_probability_matrix_groups(
        all_rows,
        prob_col="model_probability",
        label="aggregated ALL model matrix",
        require_normalized=True,
    )
    return pd.concat([out, all_rows], ignore_index=True)


def combine_empirical_and_model(
    emp: pd.DataFrame,
    model: pd.DataFrame,
    mods: Sequence[int],
    blocks: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if blocks is not None:
        wanted = [str(b) for b in blocks]
        emp = emp[emp["block"].astype(str).isin(wanted)]
        model = model[model["block"].astype(str).isin(wanted)]
    rows = []
    missing_groups: List[Tuple[str, int]] = []
    for (block, q0), esub in emp.groupby(["block", "q"]):
        q = int(q0)
        if q not in mods:
            continue
        msub = model[(model["block"].astype(str) == str(block)) & (model["q"].astype(int) == q)]
        if msub.empty:
            missing_groups.append((str(block), q))
            continue
        _validate_complete_matrix_cells(esub, q, label=f"empirical block={block}")
        validate_probability_matrix_groups(
            msub,
            prob_col="model_probability",
            label=f"model block={block}",
            require_normalized=True,
        )
        merged = esub.merge(
            msub[["q", "from_residue", "to_residue", "model_probability"]],
            on=["q", "from_residue", "to_residue"],
            how="left",
            validate="one_to_one",
        )
        if merged["model_probability"].isna().any():
            raise ValueError(f"missing model probability for block={block}, q={q}")
        for r in merged.itertuples(index=False):
            row_count = float(r.row_count)
            mp = float(r.model_probability)
            rows.append({
                "block": str(block),
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
    if missing_groups:
        raise ValueError(f"model matrices missing for empirical groups: {missing_groups}")
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


def compare_with_old(
    old_model_path: Optional[str],
    emp: pd.DataFrame,
    new_model: pd.DataFrame,
    mods: Sequence[int],
    *,
    old_model_label: str,
    oriented_model_label: str,
) -> pd.DataFrame:
    if not old_model_path:
        return pd.DataFrame()
    old = normalize_old_model_direct(old_model_path)
    old = ensure_all_model(old, emp)
    new_model = ensure_all_model(new_model, emp)
    old_combined = combine_empirical_and_model(emp, old, mods)
    new_combined = combine_empirical_and_model(emp, new_model, mods)
    old_sum = summarize_transition(old_combined)
    new_sum = summarize_transition(new_combined)
    rows = []
    for (block, q), nrow in new_sum.groupby(["block", "q"]):
        osub = old_sum[(old_sum["block"].astype(str) == str(block)) & (old_sum["q"].astype(int) == int(q))]
        if osub.empty:
            raise ValueError(f"old summary missing block={block}, q={q}")
        o = osub.iloc[0]
        n = nrow.iloc[0]
        rows.append({
            "block": str(block),
            "q": int(q),
            "old_model_label": old_model_label,
            "oriented_model_label": oriented_model_label,
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

def _coerce_bool_series(series: pd.Series) -> pd.Series:
    """Coerce booleans stored as bool, 0/1, or common text forms."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    numeric = pd.to_numeric(series, errors="coerce")
    out = numeric.eq(1)
    unresolved = numeric.isna()
    if unresolved.any():
        text = series.astype(str).str.strip().str.lower()
        out.loc[unresolved] = text.loc[unresolved].isin({"true", "t", "yes", "y"})
    return out.fillna(False).astype(bool)


def summarize_replacement_outcome(old_vs_new: pd.DataFrame) -> dict:
    """Return the exact counts used by the generated interpretation."""
    result: dict = {
        "comparison_available": False,
        "n_comparisons": 0,
        "kl_improved_count": 0,
        "l1_improved_count": 0,
        "higher_modulus_comparisons": 0,
        "higher_modulus_kl_improved_count": 0,
        "higher_modulus_l1_improved_count": 0,
        "q3_comparisons": 0,
        "q3_old_wrong_sign_count": 0,
        "q3_oriented_wrong_sign_count": 0,
        "replacement_gate_pass": False,
    }
    if old_vs_new.empty:
        return result

    work = old_vs_new.copy()
    for col in ["q", "old_kl", "oriented_kl", "old_l1", "oriented_l1"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["q", "old_kl", "oriented_kl", "old_l1", "oriented_l1"])
    if work.empty:
        return result

    kl_improved = work["oriented_kl"] < work["old_kl"]
    l1_improved = work["oriented_l1"] < work["old_l1"]
    q3 = work[work["q"].astype(int) == 3].copy()
    higher = work[work["q"].astype(int) != 3].copy()
    old_wrong = _coerce_bool_series(q3["old_wrong_sign"]) if "old_wrong_sign" in q3.columns else pd.Series(False, index=q3.index)
    oriented_wrong = _coerce_bool_series(q3["oriented_wrong_sign"]) if "oriented_wrong_sign" in q3.columns else pd.Series(False, index=q3.index)

    result.update({
        "comparison_available": True,
        "n_comparisons": int(len(work)),
        "kl_improved_count": int(kl_improved.sum()),
        "l1_improved_count": int(l1_improved.sum()),
        "higher_modulus_comparisons": int(len(higher)),
        "higher_modulus_kl_improved_count": int((higher["oriented_kl"] < higher["old_kl"]).sum()),
        "higher_modulus_l1_improved_count": int((higher["oriented_l1"] < higher["old_l1"]).sum()),
        "q3_comparisons": int(len(q3)),
        "q3_old_wrong_sign_count": int(old_wrong.sum()),
        "q3_oriented_wrong_sign_count": int(oriented_wrong.sum()),
    })
    result["replacement_gate_pass"] = bool(
        len(work) > 0
        and int(kl_improved.sum()) == len(work)
        and int(l1_improved.sum()) == len(work)
        and len(q3) > 0
        and int(old_wrong.sum()) == len(q3)
        and int(oriented_wrong.sum()) == 0
    )
    return result


def write_interpretation(outdir: Path, summary: pd.DataFrame, old_vs_new: pd.DataFrame) -> dict:
    """Write a result-driven report and return the replacement gate summary."""
    lines: List[str] = []
    outcome = summarize_replacement_outcome(old_vs_new)
    lines.append("# CHL4-D5 Orientation-Lift OS Replacement")
    lines.append("")
    lines.append("This audit compares the reproducible absolute-prime OS previous-gap-conditioned control with an orientation-lift construction from CHL2 gap-residue populations. It does not identify that control with the historical `naive_table4_legacy` implementation artifact.")
    lines.append("")
    all_rows = summary[summary["block"].astype(str) == "ALL"] if "block" in summary.columns else summary
    if not all_rows.empty:
        cols = ["q", "row_cosine_weighted", "kl_empirical_to_model_weighted", "l1_weighted", "diagonal_probability_empirical", "diagonal_probability_model", "uniform_diagonal_probability", "diagonal_wrong_sign_vs_uniform", "pearson_chi2_per_transition", "spectral_gap_abs_error", "D3_empirical", "D3_model"]
        cols = [c for c in cols if c in all_rows.columns]
        lines.append("## Oriented OS summary, ALL")
        lines.append("")
        lines.append(safe_to_markdown(all_rows[cols], index=False))
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
            lines.append("## Previous-gap-conditioned OS versus orientation-lift OS, ALL")
            lines.append("")
            cols = ["q", "old_kl", "oriented_kl", "delta_kl_old_minus_oriented", "emp_diag", "old_diag_model", "oriented_diag_model", "old_wrong_sign", "oriented_wrong_sign"]
            lines.append(safe_to_markdown(all_old[cols], index=False))
            lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if outcome["comparison_available"]:
        lines.append(
            f"The orientation lift improves weighted KL in `{outcome['kl_improved_count']}/{outcome['n_comparisons']}` "
            f"and weighted L1 in `{outcome['l1_improved_count']}/{outcome['n_comparisons']}` audited block-modulus comparisons."
        )
        if outcome["higher_modulus_comparisons"]:
            lines.append(
                f"For q=5,7,11,13, KL improves in `{outcome['higher_modulus_kl_improved_count']}/{outcome['higher_modulus_comparisons']}` "
                f"and L1 in `{outcome['higher_modulus_l1_improved_count']}/{outcome['higher_modulus_comparisons']}` comparisons."
            )
        if outcome["q3_comparisons"]:
            lines.append(
                f"For q=3, the previous-gap-conditioned control has the wrong diagonal sign in "
                f"`{outcome['q3_old_wrong_sign_count']}/{outcome['q3_comparisons']}` comparisons, whereas the orientation lift has it in "
                f"`{outcome['q3_oriented_wrong_sign_count']}/{outcome['q3_comparisons']}`."
            )
        if outcome["replacement_gate_pass"]:
            lines.append(
                "The replacement gate passes in this run. These results support treating the discrepancy of the reproducible previous-gap-conditioned OS control as an orientation-projection defect rather than as a failure of the CHL2 gap kernel."
            )
        else:
            lines.append(
                "The full replacement gate does not pass in this run; the direct-control discrepancy should therefore not be attributed solely to orientation projection."
            )
    else:
        lines.append("No previous-gap-conditioned model matrix was supplied, so this run does not support an old-versus-oriented replacement claim.")
    lines.append(
        "The v1.8 `naive_table4_legacy` provenance has been reconstructed as a mixed implementation artifact: "
        "q=3,5,7 used the available previous-gap-conditioned direct matrices, while absent q=11,13 inputs "
        "became zero rows and were silently normalized to uniform reduced-residue rows. The artifact is retained "
        "for historical explanation only; it is not the reproducible control supplied to this run. See "
        "`docs/NAIVE_TABLE4_LEGACY_PROVENANCE.md`."
    )
    (outdir / "chl4d5_interpretacion.md").write_text("\n".join(lines), encoding="utf-8")
    return outcome


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="CHL4-D5 orientation-lift OS replacement diagnostic")
    ap.add_argument("--empirical-matrix-csv", required=True, help="Direct empirical prime-residue matrix CSV, preferably with counts.")
    ap.add_argument("--mods", default="3,5,7,11,13")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--mode", default="from-blocks", choices=["from-blocks", "from-d4-lift"], help="How to obtain the oriented CHL2 matrix.")
    ap.add_argument("--d4-lift-csv", default=None, help="Existing D4 orientation lift matrix CSV.")
    ap.add_argument("--old-model-matrix-csv", default=None, help="Optional reproducible direct OS model matrix for comparison.")
    ap.add_argument("--old-model-label", default="absolute_prime_os_previous_gap_conditioned")
    ap.add_argument("--oriented-model-label", default="orientation_lift_chl2_gap_population")
    ap.add_argument("--config", default=None)
    ap.add_argument("--root", default=".")
    ap.add_argument("--blocks", default="1-10")
    ap.add_argument("--Y", type=int, default=47)
    ap.add_argument("--log-x", type=float, default=25.328436022934504)
    ap.add_argument("--path-cache-file", default=None)
    ap.add_argument("--hash-inputs", action=argparse.BooleanOptionalAction, default=False, help="Compute SHA-256 for every declared input and embed it in config/telemetry.")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    mods = parse_int_list(args.mods)

    empirical = normalize_empirical_direct(args.empirical_matrix_csv)
    empirical = empirical[empirical["q"].astype(int).isin(mods)].copy()
    empirical = ensure_all_empirical(empirical)

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
    model = ensure_all_model(model, empirical)

    combined = combine_empirical_and_model(empirical, model, mods)
    summary = summarize_transition(combined)
    old_model_support_audit = None
    if args.old_model_matrix_csv:
        old_model_support_audit = normalize_old_model_direct(
            args.old_model_matrix_csv
        ).attrs.get("matrix_support_audit")
    old_vs_new = (
        compare_with_old(
            args.old_model_matrix_csv,
            empirical,
            model,
            mods,
            old_model_label=args.old_model_label,
            oriented_model_label=args.oriented_model_label,
        )
        if args.old_model_matrix_csv
        else pd.DataFrame()
    )

    input_paths: dict[str, str | Path | None] = {
        "empirical_matrix_csv": args.empirical_matrix_csv,
        "old_model_matrix_csv": args.old_model_matrix_csv,
        "d4_lift_csv": args.d4_lift_csv,
        "config": args.config,
        "path_cache_file": args.path_cache_file,
    }
    if not gap_pop.empty and {"block", "source_file"}.issubset(gap_pop.columns):
        source_rows = gap_pop[gap_pop["block"].astype(str) != "ALL"][["block", "source_file"]].drop_duplicates()
        for row in source_rows.itertuples(index=False):
            source_path = str(row.source_file)
            if source_path and source_path != "nan":
                input_paths[f"parent_wide_{row.block}"] = source_path
    provenance = build_provenance(
        repo_root=args.root,
        input_paths=input_paths,
        hash_inputs=args.hash_inputs,
    )

    combined.insert(2, "model_label", args.oriented_model_label)
    output_files: List[str] = []
    combined.to_csv(outdir / "chl2_os_oriented_prime_residue_transition_by_mod.csv", index=False)
    output_files.append("chl2_os_oriented_prime_residue_transition_by_mod.csv")
    summary.to_csv(outdir / "chl2_os_oriented_prime_residue_summary.csv", index=False)
    output_files.append("chl2_os_oriented_prime_residue_summary.csv")
    if not old_vs_new.empty:
        old_vs_new.to_csv(outdir / "chl2_os_old_direct_vs_oriented.csv", index=False)
        output_files.append("chl2_os_old_direct_vs_oriented.csv")
    if not gap_pop.empty:
        gap_pop.to_csv(outdir / "chl2_os_oriented_gap_residue_populations.csv", index=False)
        output_files.append("chl2_os_oriented_gap_residue_populations.csv")
    model.to_csv(outdir / "chl2_os_oriented_model_matrices.csv", index=False)
    output_files.append("chl2_os_oriented_model_matrices.csv")
    interpretation_summary = write_interpretation(outdir, summary, old_vs_new)
    output_files.append("chl4d5_interpretacion.md")

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
        "n_gap_population_rows": int(len(gap_pop)),
        "n_model_matrix_rows": int(len(model)),
        "contains_all_aggregate": bool((summary["block"].astype(str) == "ALL").any()),
        "old_model_label": args.old_model_label,
        "oriented_model_label": args.oriented_model_label,
        "old_model_support_gate_pass": (
            None
            if old_model_support_audit is None
            else bool(old_model_support_audit.get("matrix_support_gate_pass"))
        ),
        "old_model_support_audit": old_model_support_audit,
        "legacy_table4_provenance": legacy_table4_provenance_summary(),
        "interpretation_summary": interpretation_summary,
        "provenance": provenance,
        "output_files": output_files,
    }
    (outdir / "chl4d5_runtime_telemetry.json").write_text(json.dumps(telemetry, indent=2), encoding="utf-8")
    config_payload = dict(vars(args))
    config_payload["old_model_support_gate_pass"] = (
        None
        if old_model_support_audit is None
        else bool(old_model_support_audit.get("matrix_support_gate_pass"))
    )
    config_payload["old_model_support_audit"] = old_model_support_audit
    config_payload["legacy_table4_provenance"] = legacy_table4_provenance_summary()
    config_payload["provenance"] = provenance
    (outdir / "chl4d5_config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
