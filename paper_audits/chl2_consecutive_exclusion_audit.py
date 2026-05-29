#!/usr/bin/env python3
"""
CHL2_CONSECUTIVE_EXCLUSION_AUDIT
=================================

Conditional Hardy--Littlewood Markov kernel with an explicit no-interior-prime
exclusion factor for consecutive prime gaps.

Background
----------
CHL1 used the conditional singular-series ratio

    R_Y(g2|g1) = S_Y({0,g1,g1+g2}) / S_Y({0,g1})

as a Markov kernel for g1 -> g2.  This is a triple-constellation weight: it
models the arithmetic weight of the three endpoints being prime.  Consecutive
gaps require an additional event: no primes occur inside the future interval
(p_{n+1}, p_{n+1}+g2).

CHL2 introduces a parameter-free exclusion factor

    E_no_interior(g2;x) = exp(-Omega_Y(g2;x)),

where Omega_Y is the expected number of interior primes under a conditional
Hardy--Littlewood intensity.  The default, efficient implementation conditions
on the two endpoints of the candidate future gap:

    Omega_Y(g;x) = sum_{2 <= u <= g-2, u even} (1/log x)
                  * S_Y({0,u,g}) / S_Y({0,g}).

This is the first-hit / no-interior correction analogous to the exponential
survival factor in a non-homogeneous Poisson process, but with local HL
singular-series weights rather than independent Cramer intensities.

An optional path-sensitive variant is included:

    Omega_Y^path(g1,g2;x) = sum_u (1/log x)
        * S_Y({0,g1,g1+u,g1+g2}) / S_Y({0,g1,g1+g2}),

which conditions on the full triple endpoint constellation.  It is more
expensive; use --path-exclusion for diagnostic runs.

No Shape/Balance/Tail geometry is used in the core CHL2 models.  Geometry is
reserved for residual analysis after CHL2.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from collections import Counter

import numpy as np
import pandas as pd

# Allow execution as a file from a fresh clone without requiring pip install -e . first.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from chl_kernel.telemetry import telemetry_start, write_telemetry


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



# ---------------------------------------------------------------------------
# Multiprocessing support for definitive path-sensitive exclusion.
# ---------------------------------------------------------------------------
# The H4/H3 no-interior factor is pure arithmetic.  It can therefore be
# evaluated in parallel without changing the model.  Each worker owns a local
# cache of singular-series products/ratios; this avoids locks and ensures that
# repeated topologies are not recomputed inside that worker.

_WORKER_PRIMES: Tuple[int, ...] = tuple()
_WORKER_LOG_X: float = 1.0
_WORKER_CACHE_MAXSIZE: int = 2_000_000
_WORKER_SINGULAR_CACHE: Dict[Tuple[int, ...], float] = {}


def _path_worker_init(primes: Sequence[int], log_x: float, cache_maxsize: int) -> None:
    global _WORKER_PRIMES, _WORKER_LOG_X, _WORKER_CACHE_MAXSIZE, _WORKER_SINGULAR_CACHE
    _WORKER_PRIMES = tuple(int(p) for p in primes)
    _WORKER_LOG_X = float(log_x)
    _WORKER_CACHE_MAXSIZE = int(cache_maxsize)
    _WORKER_SINGULAR_CACHE = {}


def _worker_singular_log(offsets: Tuple[int, ...]) -> float:
    key = tuple(int(x) for x in offsets)
    val = _WORKER_SINGULAR_CACHE.get(key)
    if val is not None:
        return val
    k = len(key)
    acc = 0.0
    for p in _WORKER_PRIMES:
        pf = float(p)
        nu = len({int(o % p) for o in key})
        f = (1.0 - nu / pf) / ((1.0 - 1.0 / pf) ** k)
        if f <= 0.0:
            acc = -np.inf
            break
        acc += math.log(f)
    val = float(acc)
    if _WORKER_CACHE_MAXSIZE <= 0 or len(_WORKER_SINGULAR_CACHE) < _WORKER_CACHE_MAXSIZE:
        _WORKER_SINGULAR_CACHE[key] = val
    else:
        # Rigorous memory valve: clearing can only reduce cache hit rate, never
        # change the computed value.
        _WORKER_SINGULAR_CACHE.clear()
        _WORKER_SINGULAR_CACHE[key] = val
    return val


def _path_exclusion_chunk(task: Tuple[int, List[Tuple[int, int]]]) -> Tuple[int, List[Tuple[int, int, int, float, float]]]:
    chunk_id, pairs = task
    L = max(float(_WORKER_LOG_X), 1.0)
    rows: List[Tuple[int, int, int, float, float]] = []
    for g1, g2 in pairs:
        g1 = int(g1); g2 = int(g2)
        log_s3 = _worker_singular_log((0, g1, g1 + g2))
        omega = 0.0
        n_int = 0
        if math.isfinite(log_s3):
            for u in range(2, g2, 2):
                n_int += 1
                log_s4 = _worker_singular_log((0, g1, g1 + u, g1 + g2))
                if math.isfinite(log_s4):
                    omega += math.exp(log_s4 - log_s3) / L
        else:
            n_int = max(0, (g2 // 2) - 1)
        rows.append((g1, g2, n_int, float(omega), -float(omega)))
    return chunk_id, rows


def _iter_chunks(seq: Sequence[Tuple[int, int]], chunk_size: int) -> Iterable[List[Tuple[int, int]]]:
    chunk_size = max(1, int(chunk_size))
    for i in range(0, len(seq), chunk_size):
        yield list(seq[i:i + chunk_size])


def auto_path_chunk_size(
    n_pairs: int,
    workers: int,
    requested_chunk_size: int,
    target_tasks_per_worker: int = 6,
    min_chunk_size: int = 32,
    max_chunk_size: int = 512,
) -> int:
    """Choose a path-cache chunk size that keeps worker processes busy.

    The original CHL2 default used large chunks (for example 5000 unique
    pairs per task).  On single-block diagnostics this can create only a few
    tasks, leaving most CPU cores idle.  A positive requested chunk size is
    respected exactly; requested_chunk_size <= 0 enables automatic sizing.
    """
    n_pairs = int(max(0, n_pairs))
    workers = int(max(1, workers))
    if requested_chunk_size and int(requested_chunk_size) > 0:
        return int(requested_chunk_size)
    if n_pairs <= 0:
        return 1
    target_tasks = max(workers, workers * int(max(1, target_tasks_per_worker)))
    chunk = int(math.ceil(n_pairs / target_tasks))
    return int(max(min_chunk_size, min(max_chunk_size, chunk)))


def compute_path_exclusion_cache_parallel(
    pairs_df: pd.DataFrame,
    primes: Sequence[int],
    log_x: float,
    workers: int,
    chunk_size: int,
    cache_maxsize: int,
    target_tasks_per_worker: int = 6,
) -> pd.DataFrame:
    pairs_df = pairs_df[["g1", "g2"]].drop_duplicates().sort_values(["g1", "g2"]).reset_index(drop=True)
    pairs = [(int(a), int(b)) for a, b in pairs_df[["g1", "g2"]].itertuples(index=False, name=None)]
    workers = max(1, int(workers))
    effective_chunk_size = auto_path_chunk_size(
        n_pairs=len(pairs),
        workers=workers,
        requested_chunk_size=int(chunk_size),
        target_tasks_per_worker=int(target_tasks_per_worker),
    )
    tasks = [(i, chunk) for i, chunk in enumerate(_iter_chunks(pairs, effective_chunk_size))]
    print(
        f"[CHL2-path] unique pairs={len(pairs):,}; chunks={len(tasks):,}; "
        f"workers={workers}; path_chunk_size={effective_chunk_size}",
        flush=True,
    )
    t0 = time.time()
    results: List[Tuple[int, List[Tuple[int, int, int, float, float]]]] = []
    if workers == 1:
        _path_worker_init(tuple(primes), float(log_x), int(cache_maxsize))
        for task in tasks:
            results.append(_path_exclusion_chunk(task))
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_path_worker_init, initargs=(tuple(primes), float(log_x), int(cache_maxsize))) as ex:
            futs = [ex.submit(_path_exclusion_chunk, task) for task in tasks]
            done = 0
            for fut in as_completed(futs):
                results.append(fut.result())
                done += 1
                if done == len(tasks) or done % max(1, len(tasks)//20) == 0:
                    print(f"[CHL2-path] completed {done:,}/{len(tasks):,} chunks ({100.0*done/len(tasks):.1f}%)", flush=True)
    results.sort(key=lambda x: x[0])
    flat: List[Tuple[int, int, int, float, float]] = []
    for _, rows in results:
        flat.extend(rows)
    print(f"[CHL2-path] path cache computed in {time.time()-t0:.1f}s", flush=True)
    return pd.DataFrame(flat, columns=["g1", "g2", "n_even_interior", "omega_path_exclusion", "logE_path_exclusion"])


def collect_unique_pairs(block_files: Sequence[Tuple[int, Path]], source: str) -> pd.DataFrame:
    frames = []
    chosen = block_files if source == "all" else block_files[:1]
    for b, path in chosen:
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"[CHL2-path] scanning unique pairs from block {b}: {path}", flush=True)
        frames.append(pd.read_csv(path, usecols=["g1", "g2"]).drop_duplicates())
    if not frames:
        return pd.DataFrame(columns=["g1", "g2"])
    return pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)


@dataclass
class ModelEval:
    model: str
    filter_name: str
    block: int
    n_events: float
    support_rows: int
    observed_rows: int
    eta: float
    loglik_sum: float
    loglik_per_event: float
    cross_entropy: float
    empirical_cond_entropy: float
    conditional_kl: float
    mean_g2_model: float
    mean_g2_empirical: float


def parse_blocks_arg(s: Optional[str], default: Sequence[int]) -> List[int]:
    if not s:
        return list(default)
    out: List[int] = []
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def primes_upto(n: int) -> List[int]:
    n = int(n)
    if n < 2:
        return []
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for p in range(2, int(n**0.5) + 1):
        if sieve[p]:
            start = p * p
            sieve[start:n+1:p] = b"\x00" * (((n - start) // p) + 1)
    return [i for i in range(2, n + 1) if sieve[i]]


def logsumexp(a: np.ndarray) -> float:
    if a.size == 0:
        return -np.inf
    m = np.max(a)
    if not np.isfinite(m):
        return -np.inf
    return float(m + np.log(np.sum(np.exp(a - m))))


def weighted_mean_from_logw(values: np.ndarray, logw: np.ndarray) -> float:
    lz = logsumexp(logw)
    if not np.isfinite(lz):
        return float('nan')
    w = np.exp(logw - lz)
    return float(np.sum(values * w))


def singular_log_tuple(offsets: Tuple[int, ...], primes: Sequence[int]) -> float:
    """Truncated singular-series log for a finite tuple H.

    S_Y(H) = prod_{p<=Y} (1 - nu_p(H)/p) / (1 - 1/p)^|H|.
    Returns -inf for inadmissible tuples.
    """
    k = len(offsets)
    acc = 0.0
    for p in primes:
        residues = {int(o % p) for o in offsets}
        nu = len(residues)
        f = (1.0 - nu / float(p)) / ((1.0 - 1.0 / float(p)) ** k)
        if f <= 0.0:
            return -np.inf
        acc += math.log(f)
    return float(acc)


def singular_logs_for_pairs(g1: np.ndarray, g2: np.ndarray, primes: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized logS2(g1), logS3(g1,g2), log ratio S3/S2, admissible mask."""
    g1 = g1.astype(np.int64, copy=False)
    g2 = g2.astype(np.int64, copy=False)
    s = g1 + g2
    n = len(g1)
    log_s2 = np.zeros(n, dtype=np.float64)
    log_s3 = np.zeros(n, dtype=np.float64)
    admissible = np.ones(n, dtype=bool)

    for p in primes:
        pf = float(p)
        g1m = g1 % p
        sm = s % p
        nu2 = np.where(g1m == 0, 1.0, 2.0)
        nu3 = np.ones(n, dtype=np.float64)
        nu3 += (g1m != 0)
        nu3 += (sm != 0) & (sm != g1m)
        f2 = (1.0 - nu2 / pf) / ((1.0 - 1.0 / pf) ** 2)
        f3 = (1.0 - nu3 / pf) / ((1.0 - 1.0 / pf) ** 3)
        ok = (f2 > 0) & (f3 > 0)
        admissible &= ok
        log_s2 += np.log(np.where(f2 > 0, f2, 1.0))
        log_s3 += np.log(np.where(f3 > 0, f3, 1.0))
    log_ratio = log_s3 - log_s2
    log_s2 = np.where(admissible, log_s2, -np.inf)
    log_s3 = np.where(admissible, log_s3, -np.inf)
    log_ratio = np.where(admissible, log_ratio, -np.inf)
    return log_s2, log_s3, log_ratio, admissible


def singular_log_s2_gap(g: np.ndarray, primes: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Truncated singular-series log for H2={0,g}; return logS2 and admissible mask."""
    g = g.astype(np.int64, copy=False)
    n = len(g)
    log_s2 = np.zeros(n, dtype=np.float64)
    admissible = np.ones(n, dtype=bool)
    for p in primes:
        pf = float(p)
        gm = g % p
        nu2 = np.where(gm == 0, 1.0, 2.0)
        f2 = (1.0 - nu2 / pf) / ((1.0 - 1.0 / pf) ** 2)
        ok = f2 > 0
        admissible &= ok
        log_s2 += np.log(np.where(ok, f2, 1.0))
    return np.where(admissible, log_s2, -np.inf), admissible


def compute_gap_exclusion_logs(g_values: Sequence[int], primes: Sequence[int], log_x: float) -> Tuple[Dict[int, float], pd.DataFrame]:
    """Compute parameter-free future-gap no-interior log factors.

    For each candidate gap g, compute

        Omega_Y(g;x) = sum_{u even, 2<=u<=g-2} S_Y({0,u,g})/S_Y({0,g}) / log(x)
        logE = -Omega_Y.

    This conditions only on the endpoints of the future gap and is therefore
    safe for order-zero models as well as conditional models.  It does not use
    the previous gap g1.
    """
    out: Dict[int, float] = {}
    rows = []
    cache_s2: Dict[int, float] = {}
    cache_s3: Dict[Tuple[int, int], float] = {}
    L = max(float(log_x), 1.0)
    for g in sorted({int(x) for x in g_values}):
        if g <= 2:
            out[g] = 0.0
            rows.append({"g": g, "n_even_interior": 0, "omega_gap_exclusion": 0.0, "logE_gap_exclusion": 0.0})
            continue
        log_s2 = cache_s2.get(g)
        if log_s2 is None:
            log_s2 = singular_log_tuple((0, g), primes)
            cache_s2[g] = log_s2
        omega = 0.0
        n_int = 0
        if np.isfinite(log_s2):
            for u in range(2, g, 2):
                n_int += 1
                key = (u, g)
                log_s3 = cache_s3.get(key)
                if log_s3 is None:
                    log_s3 = singular_log_tuple((0, u, g), primes)
                    cache_s3[key] = log_s3
                if np.isfinite(log_s3):
                    omega += math.exp(log_s3 - log_s2) / L
        else:
            # If endpoints are inadmissible, model support will remove this gap.
            for _u in range(2, g, 2):
                n_int += 1
        out[g] = -float(omega)
        rows.append({"g": g, "n_even_interior": n_int, "omega_gap_exclusion": float(omega), "logE_gap_exclusion": -float(omega)})
    return out, pd.DataFrame(rows)


def compute_cramer_exclusion_logs(g_values: Sequence[int], log_x: float) -> Dict[int, float]:
    """Simple no-interior factor exp(-#odd candidates/log x)."""
    L = max(float(log_x), 1.0)
    out = {}
    for g in sorted({int(x) for x in g_values}):
        n_int = max(0, (int(g) // 2) - 1)
        out[int(g)] = -float(n_int) / L
    return out


def compute_path_exclusion_logs_for_df(df: pd.DataFrame, primes: Sequence[int], log_x: float) -> Tuple[np.ndarray, pd.DataFrame]:
    """Compute the full path-sensitive H4/H3 no-interior log factor per row.

    Omega_path(g1,g2;x) = sum_{u even,2<=u<=g2-2} S_Y({0,g1,g1+u,g1+g2}) / S_Y({0,g1,g1+g2}) / log x.

    This is more expensive than the default gap exclusion.  It is intended for
    diagnostics or reduced supports.
    """
    L = max(float(log_x), 1.0)
    cache_s3: Dict[Tuple[int, int], float] = {}
    cache_s4: Dict[Tuple[int, int, int], float] = {}
    vals = np.zeros(len(df), dtype=np.float64)
    diag_rows = []
    g1_arr = df["g1"].to_numpy(np.int64)
    g2_arr = df["g2"].to_numpy(np.int64)
    for i, (g1, g2) in enumerate(zip(g1_arr, g2_arr)):
        g1 = int(g1); g2 = int(g2)
        key3 = (g1, g2)
        log_s3 = cache_s3.get(key3)
        if log_s3 is None:
            log_s3 = singular_log_tuple((0, g1, g1 + g2), primes)
            cache_s3[key3] = log_s3
        omega = 0.0
        n_int = 0
        if np.isfinite(log_s3):
            for u in range(2, g2, 2):
                n_int += 1
                key4 = (g1, u, g2)
                log_s4 = cache_s4.get(key4)
                if log_s4 is None:
                    log_s4 = singular_log_tuple((0, g1, g1 + u, g1 + g2), primes)
                    cache_s4[key4] = log_s4
                if np.isfinite(log_s4):
                    omega += math.exp(log_s4 - log_s3) / L
        vals[i] = -float(omega)
        if i < 10000:  # keep diagnostics bounded
            diag_rows.append({"g1": g1, "g2": g2, "n_even_interior": n_int, "omega_path_exclusion": float(omega), "logE_path_exclusion": -float(omega)})
    return vals, pd.DataFrame(diag_rows)


def solve_eta_conditional(
    df: pd.DataFrame,
    log_base: np.ndarray,
    target_mean: float,
    eta_min: float = -2.0,
    eta_max: float = 2.0,
    max_iter: int = 80,
) -> Tuple[float, float]:
    """Solve eta for P(g2|g1) using the empirical g1 distribution."""
    g1 = df["g1"].to_numpy(np.int64)
    g2 = df["g2"].to_numpy(np.float64)
    counts = df["H"].to_numpy(np.float64)
    finite = np.isfinite(log_base)
    obs_counts_by_g1 = pd.Series(counts).groupby(g1).sum()
    obs_counts_by_g1 = obs_counts_by_g1[obs_counts_by_g1 > 0]
    total_obs = float(obs_counts_by_g1.sum())
    if total_obs <= 0:
        return 0.0, float('nan')
    groups: List[Tuple[float, np.ndarray]] = []
    for state, c in obs_counts_by_g1.items():
        idx = np.flatnonzero((g1 == state) & finite)
        if idx.size > 0:
            groups.append((float(c), idx))

    def mean_at(eta: float) -> float:
        acc = 0.0
        mass = 0.0
        for c, idx in groups:
            lw = log_base[idx] + eta * g2[idx]
            mu = weighted_mean_from_logw(g2[idx], lw)
            if np.isfinite(mu):
                acc += c * mu
                mass += c
        return acc / mass if mass > 0 else float('nan')

    lo, hi = eta_min, eta_max
    mlo, mhi = mean_at(lo), mean_at(hi)
    expand = 0
    while np.isfinite(mlo) and mlo > target_mean and expand < 10:
        hi = lo
        lo *= 2
        mlo = mean_at(lo)
        expand += 1
    expand = 0
    while np.isfinite(mhi) and mhi < target_mean and expand < 10:
        lo = hi
        hi *= 2
        mhi = mean_at(hi)
        expand += 1
    if not (np.isfinite(mlo) and np.isfinite(mhi)) or mlo > target_mean or mhi < target_mean:
        return 0.0, mean_at(0.0)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        mm = mean_at(mid)
        if not np.isfinite(mm):
            break
        if mm < target_mean:
            lo = mid
        else:
            hi = mid
    eta = 0.5 * (lo + hi)
    return eta, mean_at(eta)


def solve_eta_order0(g_support: np.ndarray, log_base: np.ndarray, target_mean: float) -> Tuple[float, float]:
    finite = np.isfinite(log_base)
    g = g_support[finite].astype(np.float64)
    lb = log_base[finite]
    if g.size == 0:
        return 0.0, float('nan')
    def mean_at(eta: float) -> float:
        return weighted_mean_from_logw(g, lb + eta * g)
    lo, hi = -2.0, 2.0
    mlo, mhi = mean_at(lo), mean_at(hi)
    if not np.isfinite(mlo) or not np.isfinite(mhi) or mlo > target_mean or mhi < target_mean:
        return 0.0, mean_at(0.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if mean_at(mid) < target_mean:
            lo = mid
        else:
            hi = mid
    eta = 0.5 * (lo + hi)
    return eta, mean_at(eta)


def conditional_log_probs(df: pd.DataFrame, log_base: np.ndarray, eta: float) -> np.ndarray:
    g1 = df["g1"].to_numpy(np.int64)
    g2 = df["g2"].to_numpy(np.float64)
    logw = log_base + eta * g2
    out = np.full(len(df), -np.inf, dtype=np.float64)
    for state in np.unique(g1):
        idx = np.flatnonzero(g1 == state)
        lz = logsumexp(logw[idx])
        if np.isfinite(lz):
            out[idx] = logw[idx] - lz
    return out


def order0_log_probs(g_values: np.ndarray, log_base: np.ndarray, eta: float) -> Dict[int, float]:
    finite = np.isfinite(log_base)
    logw = log_base[finite] + eta * g_values[finite].astype(np.float64)
    lz = logsumexp(logw)
    out: Dict[int, float] = {}
    if not np.isfinite(lz):
        return out
    for g, lw in zip(g_values[finite], logw):
        out[int(g)] = float(lw - lz)
    return out


def empirical_cond_entropy(df: pd.DataFrame) -> float:
    obs = df[df["H"] > 0]
    total = float(obs["H"].sum())
    if total <= 0:
        return float('nan')
    hsum = 0.0
    for _, grp in obs.groupby("g1"):
        c = grp["H"].to_numpy(np.float64)
        cg = float(c.sum())
        if cg <= 0:
            continue
        p = c / cg
        hsum += (cg / total) * float(-np.sum(p * np.log(p)))
    return hsum


def eval_conditional_model(df: pd.DataFrame, log_base: np.ndarray, model_name: str, filter_name: str, block: int) -> ModelEval:
    obs_mask = df["H"].to_numpy(np.float64) > 0
    n_events = float(df.loc[obs_mask, "H"].sum())
    target_mean = float((df.loc[obs_mask, "H"] * df.loc[obs_mask, "g2"]).sum() / n_events) if n_events else float('nan')
    eta, mean_model = solve_eta_conditional(df, log_base, target_mean)
    logp = conditional_log_probs(df, log_base, eta)
    counts = df["H"].to_numpy(np.float64)
    obs_logp = logp[obs_mask]
    obs_counts = counts[obs_mask]
    bad = ~np.isfinite(obs_logp)
    if bad.any():
        obs_logp = obs_logp.copy()
        obs_logp[bad] = -1e9
    ll_sum = float(np.sum(obs_counts * obs_logp))
    ll_event = ll_sum / n_events if n_events else float('nan')
    h_emp = empirical_cond_entropy(df)
    ce = -ll_event
    kl = ce - h_emp if np.isfinite(h_emp) else float('nan')
    return ModelEval(model_name, filter_name, block, n_events, len(df), int(obs_mask.sum()), eta, ll_sum, ll_event, ce, h_emp, kl, mean_model, target_mean)


def eval_order0_model(df: pd.DataFrame, g_support: np.ndarray, log_base_support: np.ndarray, model_name: str, filter_name: str, block: int, fit_eta: bool = True) -> ModelEval:
    obs = df[df["H"] > 0]
    n_events = float(obs["H"].sum())
    target_mean = float((obs["H"] * obs["g2"]).sum() / n_events) if n_events else float('nan')
    if fit_eta:
        eta, mean_model = solve_eta_order0(g_support, log_base_support, target_mean)
    else:
        eta = 0.0
        mean_model = weighted_mean_from_logw(g_support.astype(float), log_base_support)
    lp_map = order0_log_probs(g_support, log_base_support, eta)
    logp_obs = np.array([lp_map.get(int(g), -1e9) for g in obs["g2"].to_numpy(np.int64)], dtype=np.float64)
    ll_sum = float(np.sum(obs["H"].to_numpy(np.float64) * logp_obs))
    ll_event = ll_sum / n_events if n_events else float('nan')
    h_emp = empirical_cond_entropy(df)
    ce = -ll_event
    kl = ce - h_emp if np.isfinite(h_emp) else float('nan')
    return ModelEval(model_name, filter_name, block, n_events, len(df), len(obs), eta, ll_sum, ll_event, ce, h_emp, kl, mean_model, target_mean)


def load_config(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def resolve_block_files(config: dict, root: Path, blocks: Sequence[int]) -> List[Tuple[int, Path]]:
    input_dir = root / config.get("input_dir", "")
    blocks_dir = input_dir / config.get("blocks_dir", "blocks")
    glob_pat = config.get("block_glob", "v46t12_ds_pilot_parent_wide_B{block:02d}.csv.gz")
    return [(b, blocks_dir / glob_pat.format(block=b)) for b in blocks]


def filter_mask(df: pd.DataFrame, name: str) -> np.ndarray:
    if "max_g" in df.columns:
        max_g = df["max_g"].to_numpy(np.int64)
    else:
        max_g = np.maximum(df["g1"].to_numpy(np.int64), df["g2"].to_numpy(np.int64))
    return FILTERS[name](max_g)


def analyze_block(
    df_all: pd.DataFrame,
    block: int,
    primes: Sequence[int],
    log_x: float,
    filters: Sequence[str],
    path_exclusion: bool = False,
    path_cache_file: Optional[Path] = None,
) -> Tuple[List[ModelEval], pd.DataFrame]:
    required = {"g1", "g2", "H"}
    missing = required - set(df_all.columns)
    if missing:
        raise ValueError(f"Block {block}: missing required columns {sorted(missing)}")
    df_all = df_all.copy()
    df_all["g1"] = df_all["g1"].astype(np.int64)
    df_all["g2"] = df_all["g2"].astype(np.int64)
    df_all["H"] = df_all["H"].astype(np.float64)
    if "max_g" not in df_all.columns:
        df_all["max_g"] = np.maximum(df_all["g1"], df_all["g2"])

    # CHL1 singular ratio for triple endpoints.
    _, _, log_ratio_all, adm3_all = singular_logs_for_pairs(df_all["g1"].to_numpy(), df_all["g2"].to_numpy(), primes)
    df_all["_log_ratio"] = log_ratio_all
    df_all["_adm3"] = adm3_all

    # Efficient no-interior factor by candidate future gap g2.
    all_g2 = sorted(df_all["g2"].astype(int).unique())
    gap_excl_map, excl_diag = compute_gap_exclusion_logs(all_g2, primes, log_x)
    cramer_excl_map = compute_cramer_exclusion_logs(all_g2, log_x)
    df_all["_logE_gap"] = df_all["g2"].astype(int).map(gap_excl_map).astype(float)
    df_all["_logE_cramer"] = df_all["g2"].astype(int).map(cramer_excl_map).astype(float)
    if path_exclusion:
        if path_cache_file is not None:
            cache_df = pd.read_csv(path_cache_file)
            needed = {"g1", "g2", "logE_path_exclusion", "omega_path_exclusion", "n_even_interior"}
            missing_cache = needed - set(cache_df.columns)
            if missing_cache:
                raise ValueError(f"Path cache {path_cache_file} missing columns {sorted(missing_cache)}")
            cache_df = cache_df[["g1", "g2", "logE_path_exclusion", "omega_path_exclusion", "n_even_interior"]]
            df_all = df_all.merge(cache_df, on=["g1", "g2"], how="left")
            miss = int(df_all["logE_path_exclusion"].isna().sum())
            if miss:
                raise ValueError(f"Block {block}: {miss} support rows missing from path-exclusion cache")
            df_all["_logE_path"] = df_all["logE_path_exclusion"].to_numpy(np.float64)
            path_diag = cache_df.head(10000).copy().assign(block=block, kind="path_h4_cache_sample")
        else:
            path_vals, path_diag = compute_path_exclusion_logs_for_df(df_all[["g1", "g2"]], primes, log_x)
            df_all["_logE_path"] = path_vals
            path_diag = path_diag.assign(block=block, kind="path_h4")
        excl_diag = excl_diag.assign(block=block)
        excl_diag = pd.concat([excl_diag.assign(kind="gap_endpoint"), path_diag], ignore_index=True)
    else:
        df_all["_logE_path"] = np.nan
        excl_diag = excl_diag.assign(block=block, kind="gap_endpoint")

    evals: List[ModelEval] = []
    for fname in filters:
        mask = filter_mask(df_all, fname)
        df = df_all.loc[mask].reset_index(drop=True)
        if df.empty or df["H"].sum() <= 0:
            continue
        log_ratio = df["_log_ratio"].to_numpy(np.float64)
        adm3 = np.isfinite(log_ratio)
        zero_cond = np.where(adm3, 0.0, -np.inf)
        logE_gap = df["_logE_gap"].to_numpy(np.float64)
        logE_cramer = df["_logE_cramer"].to_numpy(np.float64)

        # Order-1 conditional models.
        evals.append(eval_conditional_model(df, log_ratio, "CHL1_ratio_only_cond_eta", fname, block))
        evals.append(eval_conditional_model(df, np.where(adm3, log_ratio + logE_gap, -np.inf), "CHL2_gap_excl_cond_eta", fname, block))
        evals.append(eval_conditional_model(df, np.where(adm3, log_ratio + logE_cramer, -np.inf), "CHL2_cramer_excl_cond_eta", fname, block))
        evals.append(eval_conditional_model(df, np.where(adm3, zero_cond + logE_gap, -np.inf), "noPhi_gap_excl_cond_eta", fname, block))
        evals.append(eval_conditional_model(df, zero_cond, "noPhi_cond_eta", fname, block))
        if path_exclusion:
            logE_path = df["_logE_path"].to_numpy(np.float64)
            evals.append(eval_conditional_model(df, np.where(adm3, log_ratio + logE_path, -np.inf), "CHL2_path_excl_cond_eta", fname, block))

        # Order-zero baselines: use only the candidate gap g2.  No g1/triple mask.
        g_unique = np.array(sorted(df["g2"].astype(int).unique()), dtype=np.int64)
        log_s2_g, adm2_g = singular_log_s2_gap(g_unique, primes)
        logE_g = np.array([gap_excl_map[int(g)] for g in g_unique], dtype=np.float64)
        zero_g = np.where(adm2_g, 0.0, -np.inf)
        cramer_g = np.where(adm2_g, -g_unique.astype(np.float64) / log_x, -np.inf)
        # Cramer--Granville baseline: order-zero Cramer exponential spacing,
        # sieve-corrected by the pair singular series S_Y({0,g}).  The
        # gap-exclusion variant additionally applies the same parameter-free
        # no-interior survival factor used by the order-zero HL2 baseline.
        # Both are order-zero: they depend only on the candidate g2 and never
        # on g1, hence no path-memory leakage is introduced.
        cg_g = np.where(adm2_g, log_s2_g - g_unique.astype(np.float64) / log_x, -np.inf)
        cg_gap_g = np.where(adm2_g, log_s2_g + logE_g - g_unique.astype(np.float64) / log_x, -np.inf)
        evals.append(eval_order0_model(df, g_unique, log_s2_g, "HL2_order0_eta", fname, block, fit_eta=True))
        evals.append(eval_order0_model(df, g_unique, np.where(adm2_g, log_s2_g + logE_g, -np.inf), "HL2_gap_excl_order0_eta", fname, block, fit_eta=True))
        evals.append(eval_order0_model(df, g_unique, zero_g, "noPhi_order0_eta", fname, block, fit_eta=True))
        evals.append(eval_order0_model(df, g_unique, cramer_g, "Cramer_order0_exp", fname, block, fit_eta=False))
        evals.append(eval_order0_model(df, g_unique, cg_g, "Cramer_Granville_order0_exp", fname, block, fit_eta=False))
        evals.append(eval_order0_model(df, g_unique, cg_gap_g, "Cramer_Granville_gap_excl_order0_exp", fname, block, fit_eta=False))

    return evals, excl_diag


def summarize_metrics(block_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (f, model), grp in block_df.groupby(["filter_name", "model"]):
        total_n = float(grp["n_events"].sum())
        if total_n <= 0:
            continue
        ll_event = float(grp["loglik_sum"].sum() / total_n)
        ce = -ll_event
        emp_h = float(np.average(grp["empirical_cond_entropy"], weights=grp["n_events"]))
        kl = ce - emp_h
        rows.append({
            "filter": f,
            "model": model,
            "n_events": total_n,
            "loglik_per_event": ll_event,
            "cross_entropy": ce,
            "empirical_cond_entropy_weighted": emp_h,
            "conditional_kl": kl,
            "eta_weighted": float(np.average(grp["eta"], weights=grp["n_events"])),
            "mean_g2_empirical": float(np.average(grp["mean_g2_empirical"], weights=grp["n_events"])),
            "mean_g2_model": float(np.average(grp["mean_g2_model"], weights=grp["n_events"])),
            "n_blocks": int(grp["block"].nunique()),
            "support_rows_sum": int(grp["support_rows"].sum()),
            "observed_rows_sum": int(grp["observed_rows"].sum()),
        })
    return pd.DataFrame(rows).sort_values(["filter", "loglik_per_event"], ascending=[True, False])


def pairwise_gains(summary: pd.DataFrame) -> pd.DataFrame:
    baselines = [
        "CHL1_ratio_only_cond_eta",
        "noPhi_cond_eta",
        "noPhi_gap_excl_cond_eta",
        "HL2_order0_eta",
        "HL2_gap_excl_order0_eta",
        "Cramer_order0_exp",
        "Cramer_Granville_order0_exp",
        "Cramer_Granville_gap_excl_order0_exp",
    ]
    rows = []
    for f, grp in summary.groupby("filter"):
        d = {r["model"]: r for _, r in grp.iterrows()}
        for model, r in d.items():
            for b in baselines:
                if b in d and model != b:
                    rows.append({
                        "filter": f,
                        "model": model,
                        "baseline": b,
                        "delta_loglik_model_minus_baseline": float(r["loglik_per_event"] - d[b]["loglik_per_event"]),
                        "delta_KL_baseline_minus_model": float(d[b]["conditional_kl"] - r["conditional_kl"]),
                    })
    return pd.DataFrame(rows).sort_values(["filter", "baseline", "delta_loglik_model_minus_baseline"], ascending=[True, True, False])


def memory_irreducibility(block_df: pd.DataFrame, main_model: str) -> pd.DataFrame:
    rows = []
    baselines = ["CHL1_ratio_only_cond_eta", "HL2_order0_eta", "HL2_gap_excl_order0_eta", "noPhi_cond_eta", "noPhi_gap_excl_cond_eta", "noPhi_order0_eta", "Cramer_order0_exp", "Cramer_Granville_order0_exp", "Cramer_Granville_gap_excl_order0_exp"]
    for f, grp in block_df.groupby("filter_name"):
        wide = grp.pivot_table(index="block", columns="model", values=["loglik_sum", "n_events"], aggfunc="sum")
        for baseline in baselines:
            if ("loglik_sum", main_model) not in wide.columns or ("loglik_sum", baseline) not in wide.columns:
                continue
            ll1 = wide[("loglik_sum", main_model)]
            ll0 = wide[("loglik_sum", baseline)]
            n = wide[("n_events", main_model)]
            diff_per_event_by_block = (ll1 - ll0) / n
            total_diff = float((ll1 - ll0).sum())
            total_n = float(n.sum())
            se = float(diff_per_event_by_block.std(ddof=1) / math.sqrt(len(diff_per_event_by_block))) if len(diff_per_event_by_block) > 1 else np.nan
            mean_block = float(diff_per_event_by_block.mean())
            z_block = mean_block / se if se and np.isfinite(se) and se > 0 else np.nan
            rows.append({
                "filter": f,
                "order1_model": main_model,
                "baseline": baseline,
                "n_events": total_n,
                "delta_loglik_per_event_total": total_diff / total_n if total_n else np.nan,
                "LR_stat_2_delta_LL": 2.0 * total_diff,
                "block_mean_delta_loglik": mean_block,
                "block_se_delta_loglik": se,
                "block_z": z_block,
                "n_blocks": len(diff_per_event_by_block),
                "interpretation_note": "LR_stat is descriptive unless models are regular nested likelihoods; use block_z/holdout evidence for robustness.",
            })
    return pd.DataFrame(rows).sort_values(["filter", "delta_loglik_per_event_total"], ascending=[True, False])



def analyze_block_file_task(task: Tuple[int, str, Tuple[int, ...], float, Tuple[str, ...], bool, Optional[str]]) -> Tuple[List[dict], pd.DataFrame]:
    b, path_str, primes, log_x, filters, path_exclusion, path_cache_file_str = task
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(path)
    print(f"[CHL2] worker reading block {b}: {path}", flush=True)
    df = pd.read_csv(path)
    evals, excl_diag = analyze_block(
        df,
        b,
        primes,
        log_x,
        filters,
        path_exclusion=path_exclusion,
        path_cache_file=Path(path_cache_file_str) if path_cache_file_str else None,
    )
    return [e.__dict__ for e in evals], excl_diag



# ---------------------------------------------------------------------------
# Absolute-prime-residue Oliver--Soundararajan diagnostic for CHL2.
# ---------------------------------------------------------------------------
# This diagnostic is deliberately separated from the parent-wide gap-residue
# proxy.  It reads the chronological prime sequence and compares the empirical
# transition matrix
#
#     T_q(a,b) = P(p_i == a mod q | p_{i-1} == b mod q)
#
# against the transition matrix predicted by the CHL2 kernel.  For the model
# prediction of p_i from p_{i-1}, the state uses the actual previous gap
# g_{i-2}=p_{i-1}-p_{i-2}, but never the target gap g_{i-1}; hence no target
# leakage is introduced.


def _dedupe_paths(paths: Iterable[Path]) -> List[Path]:
    """Deduplicate paths while preserving order."""
    out: List[Path] = []
    seen = set()
    for path in paths:
        if path is None:
            continue
        pp = Path(path)
        key = str(pp)
        if key not in seen:
            out.append(pp)
            seen.add(key)
    return out


def prime_csv_candidates(
    prime_arg: Optional[str],
    config: dict,
    root: Path,
    *,
    config_path: Optional[Path] = None,
) -> List[Path]:
    """Return plausible chronological-prime CSV paths for the OS diagnostic.

    ``--prime-csv AUTO`` searches common config keys and generated-data
    filenames.  Explicit paths are resolved relative to the current working
    directory, the provided root, and the config directory.  This makes the
    CHL2 script robust across datasets generated by older and newer pipelines.
    """
    if not prime_arg:
        return []

    root = Path(root)
    config_path = Path(config_path) if config_path is not None else None
    config_dir = config_path.parent if config_path is not None else Path.cwd()
    input_dir = config.get("input_dir", "") or ""
    candidates: List[Path] = []

    def add_rel(value: str) -> None:
        if value is None:
            return
        value = str(value).strip()
        if not value:
            return
        path = Path(value)
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.extend([
                path,
                Path.cwd() / path,
                root / path,
                config_dir / path,
                config_dir.parent / path,
            ])
            if input_dir:
                candidates.extend([
                    root / input_dir / path,
                    config_dir / input_dir / path,
                    config_dir.parent / input_dir / path,
                ])

    if str(prime_arg).upper() == "AUTO":
        for key in (
            "real_prime_sequence",
            "prime_csv",
            "prime_sequence",
            "primes_csv",
            "real_primes_csv",
            "real_prime_csv",
            "prime_file",
        ):
            val = config.get(key)
            if val:
                add_rel(str(val))
        for name in (
            "real_primes.csv.gz",
            "real_primes.csv",
            "v46t12_ds1_real_primes.csv.gz",
            "v46t12_ds1_real_primes.csv",
        ):
            add_rel(name)
    else:
        add_rel(str(prime_arg))

    return _dedupe_paths(candidates)


def resolve_prime_csv(
    prime_arg: Optional[str],
    config: dict,
    root: Path,
    *,
    config_path: Optional[Path] = None,
) -> Optional[Path]:
    """Resolve a chronological-prime CSV path.

    Returns the first existing candidate.  If no candidate exists, returns the
    first candidate so the config/status output can show what path was tried.
    """
    candidates = prime_csv_candidates(prime_arg, config, root, config_path=config_path)
    for path in candidates:
        if Path(path).exists():
            return Path(path)
    return candidates[0] if candidates else None

def infer_prime_column(path: Path) -> Tuple[Optional[str], bool]:
    """Return (column_name, header_mode).  header_mode=False means use header=None."""
    try:
        sample = pd.read_csv(path, nrows=10)
        numeric_cols = []
        for c in sample.columns:
            vals = pd.to_numeric(sample[c], errors="coerce")
            if vals.notna().sum() >= max(1, len(vals)//2):
                numeric_cols.append(c)
        preferred = ["p", "prime", "primes", "current_prime", "p_n", "n"]
        lower_map = {str(c).lower(): c for c in sample.columns}
        for name in preferred:
            if name in lower_map:
                return lower_map[name], True
        if numeric_cols:
            return numeric_cols[0], True
    except Exception:
        pass
    return None, False


def stream_prime_values(path: Path, chunksize: int = 1_000_000) -> Iterable[np.ndarray]:
    col, has_header = infer_prime_column(path)
    if has_header and col is not None:
        for chunk in pd.read_csv(path, usecols=[col], chunksize=chunksize):
            vals = pd.to_numeric(chunk[col], errors="coerce").dropna().astype(np.int64).to_numpy()
            if vals.size:
                yield vals
    else:
        for chunk in pd.read_csv(path, header=None, usecols=[0], chunksize=chunksize):
            vals = pd.to_numeric(chunk.iloc[:, 0], errors="coerce").dropna().astype(np.int64).to_numpy()
            if vals.size:
                yield vals


def reduced_residues_for_mod(q: int, mode: str = "reduced") -> List[int]:
    q = int(q)
    if mode == "all":
        return list(range(q))
    return [a for a in range(q) if math.gcd(a, q) == 1]


def spectral_gap_row_stochastic(T: np.ndarray) -> float:
    if T.size == 0 or T.shape[0] < 2:
        return float("nan")
    try:
        vals = np.linalg.eigvals(T)
        mags = np.sort(np.abs(vals))[::-1]
        if mags.size < 2:
            return float("nan")
        return float(1.0 - mags[1])
    except Exception:
        return float("nan")


def row_cosine_weighted(emp: np.ndarray, pred: np.ndarray, row_counts: np.ndarray) -> float:
    total = float(np.sum(row_counts))
    if total <= 0:
        return float("nan")
    acc = 0.0
    for i in range(emp.shape[0]):
        if row_counts[i] <= 0:
            continue
        a = emp[i]
        b = pred[i]
        den = float(np.linalg.norm(a) * np.linalg.norm(b))
        cos = float(np.dot(a, b) / den) if den > 0 else 0.0
        acc += (row_counts[i] / total) * cos
    return float(acc)


def matrix_kl_weighted(emp: np.ndarray, pred: np.ndarray, row_counts: np.ndarray, eps: float = 1e-15) -> float:
    total = float(np.sum(row_counts))
    if total <= 0:
        return float("nan")
    out = 0.0
    for i in range(emp.shape[0]):
        if row_counts[i] <= 0:
            continue
        p = np.clip(emp[i].astype(float), eps, 1.0)
        p = p / p.sum()
        q = np.clip(pred[i].astype(float), eps, 1.0)
        q = q / q.sum()
        out += (row_counts[i] / total) * float(np.sum(p * np.log(p / q)))
    return float(out)



def pearson_chi_square_fast(obs_counts: np.ndarray, exp_counts: np.ndarray, row_counts: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Pearson chi-square table diagnostic without SciPy dependency.

    The expected table may be unnormalised.  For each empirical row, expected
    counts are scaled to the empirical row total before computing the statistic.
    This is the same convention used in the paper's prime-residue diagnostic.
    """
    O = np.asarray(obs_counts, dtype=float)
    E = np.asarray(exp_counts, dtype=float)
    if O.shape != E.shape:
        raise ValueError(f"obs/expected shape mismatch: {O.shape} vs {E.shape}")
    if row_counts is None:
        row_counts = O.sum(axis=1)
    row_counts = np.asarray(row_counts, dtype=float)
    chi2 = 0.0
    inf_flag = False
    cells_used = 0
    rows_used = 0
    df = 0
    eps = 1e-15
    for i in range(O.shape[0]):
        if row_counts[i] <= 0:
            continue
        e_row = E[i].copy()
        e_sum = float(e_row.sum())
        if e_sum > 0:
            e_row *= float(row_counts[i]) / e_sum
        positive_expected = e_row > eps
        positive_observed = O[i] > 0
        if np.any(positive_observed & ~positive_expected):
            inf_flag = True
        mask = positive_expected
        if mask.any():
            chi2 += float(np.sum((O[i, mask] - e_row[mask]) ** 2 / e_row[mask]))
            cells_used += int(mask.sum())
            rows_used += 1
            df += max(int(mask.sum()) - 1, 0)
    if inf_flag:
        chi2 = float("inf")
    total = float(np.sum(row_counts))
    return {
        "pearson_chi2": float(chi2),
        "pearson_chi2_df": int(df),
        "pearson_chi2_pvalue": float("nan"),
        "pearson_chi2_per_transition": float(chi2 / total) if total > 0 and np.isfinite(chi2) else float("inf"),
        "pearson_chi2_cells_used": int(cells_used),
        "pearson_chi2_rows_used": int(rows_used),
        "pearson_chi2_infinite_flag": bool(inf_flag),
    }

def normalize_rows_from_counts(counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    row_counts = counts.sum(axis=1).astype(float)
    P = np.zeros_like(counts, dtype=float)
    for i, c in enumerate(row_counts):
        if c > 0:
            P[i, :] = counts[i, :] / c
    return P, row_counts


def load_parentwide_aggregate(block_files: Sequence[Tuple[int, Path]]) -> pd.DataFrame:
    frames = []
    for b, path in block_files:
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"[CHL2-OS] reading support for aggregate kernel from block {b}: {path}", flush=True)
        usecols = ["g1", "g2", "H"]
        df = pd.read_csv(path, usecols=usecols)
        df["g1"] = df["g1"].astype(np.int64)
        df["g2"] = df["g2"].astype(np.int64)
        df["H"] = df["H"].astype(np.float64)
        frames.append(df)
    agg = pd.concat(frames, ignore_index=True).groupby(["g1", "g2"], as_index=False)["H"].sum()
    print(f"[CHL2-OS] aggregate support rows={len(agg):,}; empirical events={agg['H'].sum():,.0f}", flush=True)
    return agg



def build_order0_kernel_mod_maps(
    df: pd.DataFrame,
    primes: Sequence[int],
    log_x: float,
    model_name: str,
    mods: Sequence[int],
) -> Tuple[Dict[int, Dict[int, np.ndarray]], Dict[str, float]]:
    """Build P(g2 mod q) maps for order-zero baselines.

    These maps intentionally do not condition on g1.  They are expanded to all
    observed g1 support states solely so that the absolute-prime-residue OS
    diagnostic can compare an order-zero model against the same empirical
    transition rows as CHL2.
    """
    g_unique = np.array(sorted(df["g2"].astype(int).unique()), dtype=np.int64)
    log_s2_g, adm2_g = singular_log_s2_gap(g_unique, primes)
    gap_excl_map, _ = compute_gap_exclusion_logs(g_unique.tolist(), primes, log_x)
    logE_g = np.array([gap_excl_map[int(g)] for g in g_unique], dtype=np.float64)
    zero_g = np.where(adm2_g, 0.0, -np.inf)
    cramer_g = np.where(adm2_g, -g_unique.astype(np.float64) / log_x, -np.inf)
    if model_name == "Cramer_order0_exp":
        log_base = cramer_g
        eta = 0.0
    elif model_name == "Cramer_Granville_order0_exp":
        log_base = np.where(adm2_g, log_s2_g - g_unique.astype(np.float64) / log_x, -np.inf)
        eta = 0.0
    elif model_name == "Cramer_Granville_gap_excl_order0_exp":
        log_base = np.where(adm2_g, log_s2_g + logE_g - g_unique.astype(np.float64) / log_x, -np.inf)
        eta = 0.0
    elif model_name == "HL2_order0_eta":
        obs = df[df["H"] > 0]
        target_mean = float((obs["H"] * obs["g2"]).sum() / obs["H"].sum()) if len(obs) else float("nan")
        eta, _ = solve_eta_order0(g_unique, log_s2_g, target_mean)
        log_base = log_s2_g
    elif model_name == "HL2_gap_excl_order0_eta":
        obs = df[df["H"] > 0]
        target_mean = float((obs["H"] * obs["g2"]).sum() / obs["H"].sum()) if len(obs) else float("nan")
        log_base = np.where(adm2_g, log_s2_g + logE_g, -np.inf)
        eta, _ = solve_eta_order0(g_unique, log_base, target_mean)
    elif model_name == "noPhi_order0_eta":
        obs = df[df["H"] > 0]
        target_mean = float((obs["H"] * obs["g2"]).sum() / obs["H"].sum()) if len(obs) else float("nan")
        eta, _ = solve_eta_order0(g_unique, zero_g, target_mean)
        log_base = zero_g
    else:
        raise ValueError(f"Unsupported order-zero OS model {model_name}")
    lp_map = order0_log_probs(g_unique, log_base, eta)
    weights = np.array([math.exp(lp_map.get(int(g), -np.inf)) for g in g_unique], dtype=float)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    maps: Dict[int, Dict[int, np.ndarray]] = {int(q): {} for q in mods}
    unique_g1 = sorted(df["g1"].astype(int).unique())
    for q in mods:
        q = int(q)
        arr = np.bincount((g_unique % q).astype(np.int64), weights=weights, minlength=q).astype(float)
        if arr.sum() > 0:
            arr = arr / arr.sum()
        for g1 in unique_g1:
            maps[q][int(g1)] = arr.copy()
    obs = df[df["H"] > 0]
    target_mean = float((obs["H"] * obs["g2"]).sum() / obs["H"].sum()) if len(obs) else float("nan")
    mean_model = float(np.sum(g_unique.astype(float) * weights)) if weights.sum() > 0 else float("nan")
    meta = {"eta": float(eta), "target_mean_g2": target_mean, "mean_g2_model": mean_model, "support_rows": float(len(df)), "empirical_events": float(df["H"].sum())}
    return maps, meta

def build_chl2_kernel_mod_maps(
    block_files: Sequence[Tuple[int, Path]],
    primes: Sequence[int],
    log_x: float,
    model_name: str,
    mods: Sequence[int],
    path_cache_file: Optional[Path] = None,
) -> Tuple[Dict[int, Dict[int, np.ndarray]], Dict[str, float]]:
    """Build P(g2 mod q | g1) maps from the full parent-wide support.

    The model is parameter-free except eta, which is solved once on the full
    aggregate support to match the empirical mean of g2.  This is the same PNT
    scale anchor used in CHL2 and does not inspect prime-residue targets.
    """
    df = load_parentwide_aggregate(block_files)
    order0_names = {
        "Cramer_order0_exp",
        "Cramer_Granville_order0_exp",
        "Cramer_Granville_gap_excl_order0_exp",
        "HL2_order0_eta",
        "HL2_gap_excl_order0_eta",
        "noPhi_order0_eta",
    }
    if str(model_name) in order0_names:
        return build_order0_kernel_mod_maps(df, primes, log_x, str(model_name), mods)
    _, _, log_ratio, adm3 = singular_logs_for_pairs(df["g1"].to_numpy(), df["g2"].to_numpy(), primes)
    all_g2 = sorted(df["g2"].astype(int).unique())
    gap_excl_map, _diag = compute_gap_exclusion_logs(all_g2, primes, log_x)
    cramer_excl_map = compute_cramer_exclusion_logs(all_g2, log_x)
    logE_gap = df["g2"].astype(int).map(gap_excl_map).astype(float).to_numpy()
    logE_cramer = df["g2"].astype(int).map(cramer_excl_map).astype(float).to_numpy()
    zero = np.where(np.isfinite(log_ratio), 0.0, -np.inf)
    model_name = str(model_name)
    if model_name == "CHL1_ratio_only_cond_eta":
        log_base = log_ratio
    elif model_name == "CHL2_gap_excl_cond_eta":
        log_base = np.where(adm3, log_ratio + logE_gap, -np.inf)
    elif model_name == "CHL2_cramer_excl_cond_eta":
        log_base = np.where(adm3, log_ratio + logE_cramer, -np.inf)
    elif model_name == "noPhi_gap_excl_cond_eta":
        log_base = np.where(adm3, zero + logE_gap, -np.inf)
    elif model_name == "noPhi_cond_eta":
        log_base = zero
    elif model_name == "CHL2_path_excl_cond_eta":
        if path_cache_file is None or not path_cache_file.exists():
            raise FileNotFoundError(f"Path-sensitive OS prediction requested but path cache is missing: {path_cache_file}")
        cache_df = pd.read_csv(path_cache_file, usecols=["g1", "g2", "logE_path_exclusion"])
        df = df.merge(cache_df, on=["g1", "g2"], how="left")
        if df["logE_path_exclusion"].isna().any():
            miss = int(df["logE_path_exclusion"].isna().sum())
            raise ValueError(f"Aggregate OS support has {miss} pairs missing from path cache")
        logE_path = df["logE_path_exclusion"].to_numpy(np.float64)
        log_base = np.where(adm3, log_ratio + logE_path, -np.inf)
    else:
        raise ValueError(f"Unsupported OS model {model_name}")

    obs = df[df["H"] > 0]
    target_mean = float((obs["H"] * obs["g2"]).sum() / obs["H"].sum()) if len(obs) else float("nan")
    eta, mean_model = solve_eta_conditional(df, log_base, target_mean)
    logp = conditional_log_probs(df, log_base, eta)

    maps: Dict[int, Dict[int, np.ndarray]] = {int(q): {} for q in mods}
    df_tmp = df.assign(_logp=logp)
    for g1, grp in df_tmp.groupby("g1"):
        lp = grp["_logp"].to_numpy(np.float64)
        finite = np.isfinite(lp)
        if not finite.any():
            continue
        probs = np.exp(lp[finite])
        g2vals = grp["g2"].to_numpy(np.int64)[finite]
        # Numerical guard; conditional_log_probs should already be normalized.
        psum = float(probs.sum())
        if psum <= 0:
            continue
        probs = probs / psum
        for q in mods:
            arr = np.bincount((g2vals % int(q)).astype(np.int64), weights=probs, minlength=int(q)).astype(float)
            s = arr.sum()
            maps[int(q)][int(g1)] = arr / s if s > 0 else arr
    meta = {"eta": float(eta), "target_mean_g2": float(target_mean), "mean_g2_model": float(mean_model), "support_rows": float(len(df)), "empirical_events": float(df["H"].sum())}
    return maps, meta


def run_os_prime_residue_test(
    prime_csv: Path,
    block_files: Sequence[Tuple[int, Path]],
    primes: Sequence[int],
    log_x: float,
    model_name: str,
    mods: Sequence[int],
    residue_mode: str,
    path_cache_file: Optional[Path],
    max_transitions: int,
    chunksize: int,
    outdir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Vectorized absolute prime-residue Oliver--Soundararajan diagnostic.

    The older implementation streamed the chronological prime sequence one
    transition at a time and updated the prediction inside nested Python loops.
    This version keeps the same mathematics but groups each prime chunk by
    ``(g_prev, from_residue, to_residue)`` using NumPy.  It is much faster on
    DS1 while preserving the no-target-leakage rule: the model sees the actual
    previous gap ``g_{i-2}``, never the target gap ``g_{i-1}``.

    The matrix CSV writes ``model_expected_count = row_count * model_prob`` so
    the Pearson chi-square statistic can be recomputed directly from the CSV.
    """
    if not prime_csv.exists():
        raise FileNotFoundError(prime_csv)
    mods = [int(q) for q in mods]

    print(f"[CHL2-OS] building kernel maps for {model_name}", flush=True)
    kernel_maps, meta = build_chl2_kernel_mod_maps(block_files, primes, log_x, model_name, mods, path_cache_file=path_cache_file)

    residues = {q: reduced_residues_for_mod(q, residue_mode) for q in mods}
    row_index = {q: {r: i for i, r in enumerate(residues[q])} for q in mods}
    dense_index: Dict[int, np.ndarray] = {}
    for q in mods:
        idx = np.full(q, -1, dtype=np.int16)
        for i, r in enumerate(residues[q]):
            idx[int(r) % q] = i
        dense_index[q] = idx

    # Precompute P(a | b, g_prev, q) matrices for every supported previous gap.
    transition_rows: Dict[int, Dict[int, np.ndarray]] = {}
    for q in mods:
        n = len(residues[q])
        transition_rows[q] = {}
        for gp, km in kernel_maps.get(q, {}).items():
            mat = np.zeros((n, n), dtype=np.float64)
            for bi, b in enumerate(residues[q]):
                for d, prob in enumerate(km):
                    if prob <= 0.0:
                        continue
                    j = row_index[q].get((int(b) + int(d)) % q)
                    if j is not None:
                        mat[bi, j] += float(prob)
            transition_rows[q][int(gp)] = mat

    counts = {q: np.zeros((len(residues[q]), len(residues[q])), dtype=np.float64) for q in mods}
    preds = {q: np.zeros_like(counts[q], dtype=np.float64) for q in mods}
    skipped_no_kernel = Counter()
    skipped_residue = Counter()
    used = Counter()

    history = np.array([], dtype=np.int64)
    total_transitions = 0
    chunk_no = 0
    t0 = time.time()
    print(f"[CHL2-OS] vectorized streaming prime sequence: {prime_csv}", flush=True)
    for vals in stream_prime_values(prime_csv, chunksize=chunksize):
        vals_arr = np.asarray(vals, dtype=np.int64)
        if vals_arr.size == 0:
            continue
        arr = np.concatenate([history, vals_arr]) if history.size else vals_arr
        if arr.size < 3:
            history = arr[-2:].copy()
            continue

        prevprev_arr = arr[:-2]
        prev_arr = arr[1:-1]
        cur_arr = arr[2:]
        g_prev_arr = prev_arr - prevprev_arr

        if max_transitions:
            remaining = int(max_transitions) - int(total_transitions)
            if remaining <= 0:
                break
            if g_prev_arr.size > remaining:
                prev_arr = prev_arr[:remaining]
                cur_arr = cur_arr[:remaining]
                g_prev_arr = g_prev_arr[:remaining]

        n_events = int(g_prev_arr.size)
        if n_events <= 0:
            history = arr[-2:].copy()
            continue

        for q in mods:
            n = len(residues[q])
            if n == 0:
                continue
            bi_all = dense_index[q][np.mod(prev_arr, q)]
            ai_all = dense_index[q][np.mod(cur_arr, q)]
            valid = (bi_all >= 0) & (ai_all >= 0)
            invalid = int(n_events - int(valid.sum()))
            if invalid:
                skipped_residue[q] += invalid
            if not np.any(valid):
                continue

            g_valid = g_prev_arr[valid].astype(np.int64, copy=False)
            bi_valid = bi_all[valid].astype(np.int64, copy=False)
            ai_valid = ai_all[valid].astype(np.int64, copy=False)

            key = g_valid * (n * n) + bi_valid * n + ai_valid
            uniq, freq = np.unique(key, return_counts=True)
            decoded_gp = uniq // (n * n)
            rem = uniq - decoded_gp * (n * n)
            decoded_bi = rem // n
            decoded_ai = rem - decoded_bi * n

            row_cache = transition_rows[q]
            for gp, bi, ai, cnt in zip(decoded_gp, decoded_bi, decoded_ai, freq):
                mat = row_cache.get(int(gp))
                if mat is None:
                    skipped_no_kernel[q] += int(cnt)
                    continue
                c = float(cnt)
                counts[q][int(bi), int(ai)] += c
                preds[q][int(bi), :] += c * mat[int(bi), :]
                used[q] += int(cnt)

        total_transitions += n_events
        chunk_no += 1
        if chunk_no % 5 == 0:
            elapsed = time.time() - t0
            rate = total_transitions / elapsed if elapsed > 0 else 0.0
            print(f"[CHL2-OS] streamed {total_transitions:,} transitions in {elapsed:.1f}s ({rate:,.0f}/s)", flush=True)

        history = arr[-2:].copy()
        if max_transitions and total_transitions >= int(max_transitions):
            break

    summary_rows = []
    matrix_rows = []
    for q in mods:
        emp, row_counts = normalize_rows_from_counts(counts[q])
        pred, _pred_row_counts = normalize_rows_from_counts(preds[q])
        chi = pearson_chi_square_fast(counts[q], preds[q], row_counts=row_counts)
        rc = row_cosine_weighted(emp, pred, row_counts)
        kl = matrix_kl_weighted(emp, pred, row_counts)
        fro = float(np.linalg.norm(emp - pred))
        l1_weighted = float(np.sum((row_counts / row_counts.sum()) * np.sum(np.abs(emp - pred), axis=1))) if row_counts.sum() > 0 else float("nan")
        sg_emp = spectral_gap_row_stochastic(emp)
        sg_pred = spectral_gap_row_stochastic(pred)
        diag_emp = float(np.average(np.diag(emp), weights=row_counts)) if row_counts.sum() > 0 else float("nan")
        diag_pred = float(np.average(np.diag(pred), weights=row_counts)) if row_counts.sum() > 0 else float("nan")
        uniform_diag = 1.0 / len(residues[q]) if residues[q] else float("nan")
        wrong_sign = False
        if np.isfinite(diag_emp) and np.isfinite(diag_pred) and np.isfinite(uniform_diag):
            wrong_sign = (diag_emp - uniform_diag) * (diag_pred - uniform_diag) < 0
        summary_row = {
            "q": q,
            "model": model_name,
            "residue_mode": residue_mode,
            "prime_csv": str(prime_csv),
            "n_raw_transitions_seen": int(total_transitions),
            "used_transitions": int(used[q]),
            "skipped_no_kernel": int(skipped_no_kernel[q]),
            "skipped_residue": int(skipped_residue[q]),
            "row_cosine_weighted_by_empirical_pi": rc,
            "weighted_row_L1": l1_weighted,
            "frobenius_norm": fro,
            "weighted_KL_empirical_to_model": kl,
            "spectral_gap_empirical": sg_emp,
            "spectral_gap_model": sg_pred,
            "spectral_gap_abs_error": abs(sg_pred - sg_emp) if np.isfinite(sg_emp) and np.isfinite(sg_pred) else float("nan"),
            "diagonal_probability_empirical": diag_emp,
            "diagonal_probability_model": diag_pred,
            "uniform_diagonal_probability": uniform_diag,
            "diagonal_wrong_sign_vs_uniform": bool(wrong_sign),
            "kernel_eta": meta.get("eta"),
            "kernel_target_mean_g2": meta.get("target_mean_g2"),
            "kernel_mean_g2_model": meta.get("mean_g2_model"),
        }
        summary_row.update(chi)
        summary_rows.append(summary_row)
        res = residues[q]
        for i, b in enumerate(res):
            for j, a in enumerate(res):
                matrix_rows.append({
                    "q": q,
                    "model": model_name,
                    "from_residue_b_prev_prime": b,
                    "to_residue_a_current_prime": a,
                    "empirical_count": counts[q][i, j],
                    "empirical_prob": emp[i, j],
                    "model_expected_count": row_counts[i] * pred[i, j],
                    "model_prob": pred[i, j],
                    "row_count": row_counts[i],
                })
    summary_df = pd.DataFrame(summary_rows)
    matrix_df = pd.DataFrame(matrix_rows)
    summary_df.to_csv(outdir / "chl2_os_prime_residue_summary.csv", index=False)
    matrix_df.to_csv(outdir / "chl2_os_prime_residue_transition_by_mod.csv", index=False)
    return summary_df, matrix_df

def main() -> None:
    ap = argparse.ArgumentParser(description="CHL2 consecutive exclusion audit")
    ap.add_argument("--config", required=True, help="Path to v46r2c_config.json or equivalent")
    ap.add_argument("--root", default=".", help="Root directory containing input_dir from config")
    ap.add_argument("--blocks", default=None, help="Blocks to run, e.g. '1-10' or '1,2,3'. Default from config")
    ap.add_argument("--output-dir", default="chl2_consecutive_exclusion_outputs")
    ap.add_argument("--pmax", type=int, default=None, help="Prime cutoff for singular series. Default config pmax")
    ap.add_argument("--y-mode", choices=["pmax", "logx", "sqrtlogx"], default="pmax")
    ap.add_argument("--filters", default=",".join(FILTERS.keys()), help="Comma-separated filters")
    ap.add_argument("--path-exclusion", action="store_true", help="Also compute expensive path-sensitive H4/H3 exclusion factor")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="Number of worker processes. Use 0 for all detected cores.")
    ap.add_argument("--parallel-mode", choices=["auto", "blocks", "path", "none"], default="auto", help="auto: build path cache in parallel, then evaluate blocks in parallel; path: path cache parallel but block loop serial; blocks: block parallel; none: serial")
    ap.add_argument("--path-chunk-size", type=int, default=0, help="Unique (g1,g2) pairs per path-cache task; 0 enables automatic sizing")
    ap.add_argument("--path-target-tasks-per-worker", type=int, default=6, help="Automatic path-cache chunks target tasks per worker")
    ap.add_argument("--cache-maxsize", type=int, default=2_000_000, help="Max singular-series cache entries per worker; 0 means unlimited")
    ap.add_argument("--path-cache-file", default=None, help="Optional persistent CSV/CSV.GZ cache for path-sensitive logE values")
    ap.add_argument("--path-cache-source", choices=["first", "all"], default="all", help="Use first block or all blocks to build unique-pair path cache")
    ap.add_argument("--reuse-path-cache", action="store_true", help="Reuse existing path cache if it exists")
    ap.add_argument("--prime-csv", default=None, help="Chronological prime CSV for absolute prime-residue Oliver--Soundararajan test. Use AUTO to read config real_prime_sequence.")
    ap.add_argument("--os-prime-mods", default="3,5,7", help="Comma-separated moduli for absolute prime-residue test")
    ap.add_argument("--os-model", default="auto", help="Model used for prime-residue prediction; default auto chooses CHL2_path when --path-exclusion is enabled, otherwise CHL2_gap. Also supports order-zero baselines such as Cramer_Granville_gap_excl_order0_exp.")
    ap.add_argument("--os-residue-mode", choices=["reduced", "all"], default="reduced", help="Use reduced residue classes or all residues in OS matrices")
    ap.add_argument("--os-max-transitions", type=int, default=0, help="Optional cap on prime transitions for OS test; 0 means all")
    ap.add_argument("--os-prime-chunksize", type=int, default=1_000_000, help="Rows per chunk while streaming prime-csv")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    telemetry = telemetry_start()
    telemetry["script"] = "chl2_consecutive_exclusion_audit"
    telemetry["args"] = vars(args)

    config_path = Path(args.config)
    root = Path(args.root)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    blocks = parse_blocks_arg(args.blocks, config.get("blocks", list(range(1, 11))))
    block_files = resolve_block_files(config, root, blocks)
    start_x = float(config.get("start_x", 1e11))
    log_x = math.log(start_x)
    pmax_cfg = int(config.get("pmax", 47))
    if args.y_mode == "pmax":
        Y = int(args.pmax or pmax_cfg)
    elif args.y_mode == "logx":
        Y = int(math.floor(log_x))
    else:
        Y = int(math.floor(math.sqrt(log_x)))
    primes = primes_upto(Y)
    filters = [x.strip() for x in args.filters.split(',') if x.strip()]
    for f in filters:
        if f not in FILTERS:
            raise ValueError(f"Unknown filter {f}; available {sorted(FILTERS)}")

    workers = (os.cpu_count() or 1) if int(args.workers) == 0 else max(1, int(args.workers))
    path_cache_path: Optional[Path] = None
    if args.path_exclusion:
        path_cache_path = Path(args.path_cache_file) if args.path_cache_file else outdir / f"chl2_path_exclusion_cache_Y{Y}_logx{int(round(log_x*1_000_000))}.csv.gz"
        if args.dry_run:
            print(f"[CHL2-path] dry-run: path cache would be {path_cache_path}", flush=True)
        elif path_cache_path.exists() and args.reuse_path_cache:
            print(f"[CHL2-path] reusing path cache: {path_cache_path}", flush=True)
        else:
            pair_df = collect_unique_pairs(block_files, source=args.path_cache_source)
            path_workers = workers if args.parallel_mode in ("auto", "path") else 1
            cache_df = compute_path_exclusion_cache_parallel(
                pair_df,
                primes=primes,
                log_x=log_x,
                workers=path_workers,
                chunk_size=args.path_chunk_size,
                cache_maxsize=args.cache_maxsize,
                target_tasks_per_worker=args.path_target_tasks_per_worker,
            )
            path_cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_df.to_csv(path_cache_path, index=False)
            print(f"[CHL2-path] wrote path cache: {path_cache_path}", flush=True)

    diag_rows = [{"block": b, "path": str(path), "exists": path.exists()} for b, path in block_files]
    pd.DataFrame(diag_rows).to_csv(outdir / "chl2_block_diagnostics.csv", index=False)

    model_list = [
        "CHL2_gap_excl_cond_eta",
        "CHL1_ratio_only_cond_eta",
        "CHL2_cramer_excl_cond_eta",
        "noPhi_gap_excl_cond_eta",
        "noPhi_cond_eta",
        "HL2_gap_excl_order0_eta",
        "HL2_order0_eta",
        "noPhi_order0_eta",
        "Cramer_order0_exp",
        "Cramer_Granville_order0_exp",
        "Cramer_Granville_gap_excl_order0_exp",
    ]
    if args.path_exclusion:
        model_list.insert(1, "CHL2_path_excl_cond_eta")
    run_config = {
        "phase": "CHL2_CONSECUTIVE_EXCLUSION_AUDIT",
        "config": str(config_path),
        "root": str(root),
        "blocks": blocks,
        "block_files": [str(p) for _, p in block_files],
        "Y_mode": args.y_mode,
        "Y": Y,
        "primes": primes,
        "log_x": log_x,
        "path_exclusion_enabled": bool(args.path_exclusion),
        "workers": workers,
        "parallel_mode": args.parallel_mode,
        "path_chunk_size": int(args.path_chunk_size),
        "path_target_tasks_per_worker": int(args.path_target_tasks_per_worker),
        "cache_maxsize_per_worker": int(args.cache_maxsize),
        "path_cache_file": str(path_cache_path) if path_cache_path is not None else None,
        "path_cache_source": args.path_cache_source,
        "prime_csv": str(resolve_prime_csv(args.prime_csv, config, root, config_path=config_path)) if args.prime_csv else None,
        "candidate_prime_csv_paths": [str(x) for x in prime_csv_candidates(args.prime_csv, config, root, config_path=config_path)] if args.prime_csv else [],
        "os_prime_mods": [int(x) for x in args.os_prime_mods.split(',') if x.strip()],
        "filters": filters,
        "models": model_list,
        "core_factor": "E_no_interior(g;x)=exp(-sum_{2<=u<=g-2,u even} S_Y({0,u,g})/S_Y({0,g})/log(x))",
        "path_factor_optional": "E_path(g1,g2;x)=exp(-sum_u S_Y({0,g1,g1+u,g1+g2})/S_Y({0,g1,g1+g2})/log(x))",
        "note": "CHL2 tests whether adding a parameter-free no-interior-prime survival factor to the CHL singular-series ratio improves consecutive-gap Markov prediction. No Shape/Balance/Tail geometry is used in core models.",
    }
    with open(outdir / "chl2_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    if args.dry_run:
        print(json.dumps(run_config, indent=2))
        print(pd.DataFrame(diag_rows).to_string(index=False))
        return

    all_eval_dicts: List[dict] = []
    all_excl_diag: List[pd.DataFrame] = []
    block_tasks = [
        (b, str(path), tuple(primes), float(log_x), tuple(filters), bool(args.path_exclusion), str(path_cache_path) if path_cache_path is not None else None)
        for b, path in block_files
    ]
    use_block_parallel = workers > 1 and args.parallel_mode in ("auto", "blocks") and len(block_tasks) > 1
    if use_block_parallel:
        print(f"[CHL2] evaluating {len(block_tasks)} blocks with {min(workers, len(block_tasks))} worker(s)", flush=True)
        with ProcessPoolExecutor(max_workers=min(workers, len(block_tasks))) as ex:
            futures = [ex.submit(analyze_block_file_task, task) for task in block_tasks]
            done = 0
            for fut in as_completed(futures):
                eval_dicts, excl_diag = fut.result()
                all_eval_dicts.extend(eval_dicts)
                all_excl_diag.append(excl_diag)
                done += 1
                print(f"[CHL2] completed block task {done}/{len(block_tasks)}", flush=True)
    else:
        for task in block_tasks:
            eval_dicts, excl_diag = analyze_block_file_task(task)
            all_eval_dicts.extend(eval_dicts)
            all_excl_diag.append(excl_diag)

    block_df = pd.DataFrame(all_eval_dicts)
    block_df.to_csv(outdir / "chl2_metrics_by_block.csv", index=False)
    summary = summarize_metrics(block_df)
    summary.to_csv(outdir / "chl2_conditional_summary.csv", index=False)
    gains = pairwise_gains(summary)
    gains.to_csv(outdir / "chl2_pairwise_gains.csv", index=False)
    main_model = "CHL2_path_excl_cond_eta" if args.path_exclusion else "CHL2_gap_excl_cond_eta"
    mem = memory_irreducibility(block_df, main_model=main_model)
    mem.to_csv(outdir / "chl2_memory_irreducibility.csv", index=False)
    if all_excl_diag:
        pd.concat(all_excl_diag, ignore_index=True).to_csv(outdir / "chl2_exclusion_diagnostics.csv", index=False)

    os_summary = None
    if args.prime_csv:
        prime_path = resolve_prime_csv(args.prime_csv, config, root, config_path=config_path)
        if prime_path is None:
            print("[CHL2-OS] prime-csv requested but no path could be resolved; skipping", flush=True)
        elif not prime_path.exists():
            print(f"[CHL2-OS] prime-csv not found: {prime_path}; skipping", flush=True)
        else:
            os_model = args.os_model
            if os_model == "auto":
                os_model = "CHL2_path_excl_cond_eta" if args.path_exclusion else "CHL2_gap_excl_cond_eta"
            os_mods = [int(x.strip()) for x in args.os_prime_mods.split(',') if x.strip()]
            os_summary, _os_matrix = run_os_prime_residue_test(
                prime_csv=prime_path,
                block_files=block_files,
                primes=primes,
                log_x=log_x,
                model_name=os_model,
                mods=os_mods,
                residue_mode=args.os_residue_mode,
                path_cache_file=path_cache_path,
                max_transitions=int(args.os_max_transitions),
                chunksize=int(args.os_prime_chunksize),
                outdir=outdir,
            )
            print("[CHL2-OS] wrote prime-residue OS outputs", flush=True)

    best_lines = []
    for f, grp in summary.groupby("filter"):
        best = grp.sort_values("loglik_per_event", ascending=False).iloc[0]
        best_lines.append(f"- `{f}`: best `{best['model']}` with loglik/event `{best['loglik_per_event']:.8g}`, KL `{best['conditional_kl']:.8g}`.")
    gain_lines = []
    for f in filters:
        row = gains[(gains["filter"].eq(f)) & (gains["model"].eq(main_model)) & (gains["baseline"].eq("CHL1_ratio_only_cond_eta"))]
        if not row.empty:
            r = row.iloc[0]
            gain_lines.append(f"- `{f}`: {main_model} vs CHL1 ratio-only Δloglik/event `{r['delta_loglik_model_minus_baseline']:.8g}`.")
    interp = f"""# CHL2 consecutive exclusion audit — quick interpretation

Singular-series horizon: `Y={Y}` with primes `{primes}`.

## Model definition

The core CHL2 model is

`CHL2_gap_excl_cond_eta = CHL1_ratio_only_cond_eta × E_no_interior(g2;x)`,

where

`E_no_interior(g;x)=exp(-sum_{{2<=u<=g-2,u even}} S_Y({{0,u,g}})/S_Y({{0,g}})/log(x))`.

This factor is parameter-free. It penalizes candidate gaps whose interior statistically should contain primes after conditioning on the two endpoint primes of the candidate future gap.

## Best model by conditional log-likelihood

{os.linesep.join(best_lines)}

## Direct test of the exclusion factor against CHL1

{os.linesep.join(gain_lines) if gain_lines else 'No CHL2-vs-CHL1 gain rows available.'}

## Absolute-prime-residue Oliver--Soundararajan diagnostic

{('Prime-residue OS summary was written to `chl2_os_prime_residue_summary.csv`.' if os_summary is not None else 'Prime-residue OS test was not run. Pass `--prime-csv AUTO` or a CSV path to enable it.')}

## Reading rule

- Positive CHL2-vs-CHL1 `delta_loglik_model_minus_baseline` means that explicit no-interior exclusion improves the conditional singular-series Markov kernel.
- If `CHL2_gap_excl_cond_eta` beats `noPhi_gap_excl_cond_eta`, the singular-series ratio still adds information beyond the exclusion factor alone.
- If `HL2_gap_excl_order0_eta` remains behind CHL2, then memory `g1 -> g2` is not reducible to a one-gap consecutive correction.
- `Cramer_Granville_order0_exp` and `Cramer_Granville_gap_excl_order0_exp` are zero-memory sieve-corrected Cramer baselines. They test whether pair singular-series correction plus exponential spacing is enough without Markov conditioning.
- If CHL2 loses to CHL1, then the simple Poisson no-interior approximation is over-penalizing and CHL1 ratio-only remains the correct first-order kernel.

## Caveat

The default exclusion factor conditions on the endpoints of the future gap only. This is deliberately efficient and avoids adding free parameters. For the strict path-sensitive diagnostic, run with `--path-exclusion`. The optimized implementation computes the H4/H3 factor `S_Y({{0,g1,g1+u,g1+g2}})/S_Y({{0,g1,g1+g2}})` using multiprocessing and a persistent path cache over unique `(g1,g2)` pairs. This changes only the execution strategy, not the parameter-free mathematical factor.
"""
    with open(outdir / "chl2_interpretacion.md", "w", encoding="utf-8") as f:
        f.write(interp)

    write_telemetry(
        outdir / "chl2_runtime_telemetry.json",
        telemetry,
        config_path=str(config_path),
        output_dir=str(outdir),
        blocks=list(blocks),
        block_count=len(blocks),
        Y=int(Y),
        truncation_primes=list(map(int, primes)),
        log_x=float(log_x),
        filters=list(filters),
        path_exclusion=bool(args.path_exclusion),
        path_cache_file=str(path_cache_path) if path_cache_path is not None else None,
        path_cache_exists=bool(path_cache_path.exists()) if path_cache_path is not None else False,
        workers=int(workers),
        parallel_mode=str(args.parallel_mode),
        path_chunk_size=int(args.path_chunk_size),
        path_target_tasks_per_worker=int(args.path_target_tasks_per_worker),
        os_vectorized=True,
        models=list(model_list),
        metrics_rows=int(len(block_df)),
        summary_rows=int(len(summary)),
        gain_rows=int(len(gains)),
        memory_rows=int(len(mem)),
        os_executed=os_summary is not None,
        os_summary_rows=int(len(os_summary)) if os_summary is not None else 0,
        output_files=sorted([x.name for x in outdir.iterdir() if x.is_file()]),
    )
    print(f"[CHL2] wrote outputs to {outdir}")


if __name__ == "__main__":
    main()
