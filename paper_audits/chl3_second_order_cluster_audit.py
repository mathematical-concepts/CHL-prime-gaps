#!/usr/bin/env python3
"""
CHL3_SECOND_ORDER_CLUSTER_AUDIT
===============================

Experimental second-order cluster correction for the CHL2 conditional
Hardy--Littlewood Markov kernel.

This script is intentionally separate from the stable CHL2 audit.  It tests
whether a low-wheel second-order cumulant correction can reduce the q=3
absolute prime-residue anomaly without degrading the CHL2 conditional
likelihood and the q=5,7 Oliver--Soundararajan diagnostics.

The CHL2 path no-interior correction is

    log E^(1)(g1,g2) = - sum_u p_u,

where

    p_u = (1/log x) * S_Y({0,g1,g1+u,g1+g2}) / S_Y({0,g1,g1+g2}).

CHL3 tests the second-order cluster form

    log E^(2) ~= - sum_u p_u + sum_{u<v} kappa_uv,
    kappa_uv = p_uv - p_u p_v.

The practical low-wheel implementation keeps the full CHL2 first-order
intensities p_u but approximates cross-dependence using local singular-series
couplings restricted to low primes, e.g. {3}, {3,5}, {3,5,7}.  This is an
experimental branch and does not alter the CHL2 v1.4 stable release.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Allow direct execution from paper_audits/ or repository root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

try:
    import chl2_consecutive_exclusion_audit as chl2
except Exception as exc:  # pragma: no cover - fail loudly for users
    raise RuntimeError(
        "CHL3 audit expects chl2_consecutive_exclusion_audit.py to be present "
        "in the same directory (paper_audits/) or on PYTHONPATH."
    ) from exc

from chl_kernel.primes import primes_upto




# ---------------------------------------------------------------------------
# Prime CSV resolution helpers
# ---------------------------------------------------------------------------

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
    """Return plausible chronological-prime CSV paths.

    The stable CHL2 script has had several variants of ``resolve_prime_csv``
    during the project.  The CHL3 experimental audit must not depend on a
    private helper that may not exist in a user's checkout.  This resolver is
    intentionally local and conservative.

    ``--prime-csv AUTO`` searches common config keys and common generated-data
    locations.  Explicit paths are also resolved relative to the current
    working directory, the repository root argument, and the config directory.
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

        # Common filenames emitted by data_generation and by the earlier DS1
        # pipeline.  These are candidates only; no recursive search is done.
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


def resolve_prime_csv_local(
    prime_arg: Optional[str],
    config: dict,
    root: Path,
    *,
    config_path: Optional[Path] = None,
) -> Optional[Path]:
    """Resolve a chronological-prime CSV path for the OS diagnostic.

    Returns the first existing candidate.  If none exists, returns the first
    candidate so that status JSON can show what path was attempted.
    """
    candidates = prime_csv_candidates(prime_arg, config, root, config_path=config_path)
    for path in candidates:
        if Path(path).exists():
            return Path(path)
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Low-wheel second-order cluster cache
# ---------------------------------------------------------------------------

_WORKER_PRIMES: Tuple[int, ...] = tuple()
_WORKER_LOG_X: float = 1.0
_WORKER_CACHE_MAXSIZE: int = 2_000_000
_WORKER_SINGULAR_CACHE: Dict[Tuple[int, ...], float] = {}
_WORKER_LOWWHEEL_SETS: Tuple[Tuple[int, ...], ...] = tuple()


def parse_lowwheel_sets(text: str) -> List[Tuple[int, ...]]:
    """Parse low-wheel sets such as ``"3;3,5;3,5,7"``."""
    out: List[Tuple[int, ...]] = []
    for part in str(text).split(";"):
        part = part.strip()
        if not part:
            continue
        vals = tuple(sorted({int(x.strip()) for x in part.split(",") if x.strip()}))
        if vals:
            out.append(vals)
    # Preserve order, remove duplicates.
    seen = set()
    uniq: List[Tuple[int, ...]] = []
    for vals in out:
        if vals not in seen:
            uniq.append(vals)
            seen.add(vals)
    return uniq


def lowwheel_label(low_primes: Sequence[int]) -> str:
    """Return a compact label, e.g. [3,5] -> p3p5."""
    if not low_primes:
        return "empty"
    return "p" + "p".join(str(int(p)) for p in low_primes)


def even_interior_offsets(g2: int) -> range:
    """Even interior offsets 2,4,...,g2-2."""
    return range(2, int(g2), 2)


def local_factor_from_residues(residues: Sequence[int], tuple_size: int, p: int) -> float:
    """Local Hardy--Littlewood factor using tuple cardinality ``tuple_size``."""
    pf = float(p)
    nu = len({int(r) % int(p) for r in residues})
    return (1.0 - nu / pf) / ((1.0 - 1.0 / pf) ** int(tuple_size))


def lowwheel_covers_all_classes(residues: Sequence[int], p: int) -> bool:
    """Return True if residues occupy every class modulo ``p``.

    This is the hard-zero condition for low-wheel admissibility.  It is kept
    explicit because the CHL3 q=3 correction depends precisely on preserving
    the absolute zero of inadmissible H5 tuples instead of diluting it into a
    small averaged factor.
    """
    p = int(p)
    return len({int(r) % p for r in residues}) >= p


def lowwheel_is_inadmissible(residues: Sequence[int], low_primes: Sequence[int]) -> bool:
    """Return True if the tuple is inadmissible for at least one low prime."""
    return any(lowwheel_covers_all_classes(residues, int(q)) for q in low_primes)


def lowwheel_singular_from_residues(residues: Sequence[int], tuple_size: int, low_primes: Sequence[int]) -> float:
    """Product of local singular-series factors over ``low_primes`` only.

    If a tuple covers all residue classes modulo any selected low prime, the
    singular factor is a hard zero.  This is not an approximation: it is exactly
    the low-wheel Möbius/admissibility obstruction.
    """
    if lowwheel_is_inadmissible(residues, low_primes):
        return 0.0
    prod = 1.0
    for p in low_primes:
        factor = local_factor_from_residues(residues, tuple_size, int(p))
        # At this point factor should be strictly positive.  Keep the guard for
        # numerical safety but do not silently turn hard zeros into tiny values.
        if factor <= 0.0:
            return 0.0
        prod *= factor
    return float(prod)


def _cluster_worker_init(
    primes: Sequence[int],
    log_x: float,
    lowwheel_sets: Sequence[Sequence[int]],
    cache_maxsize: int,
) -> None:
    global _WORKER_PRIMES, _WORKER_LOG_X, _WORKER_LOWWHEEL_SETS, _WORKER_CACHE_MAXSIZE, _WORKER_SINGULAR_CACHE
    _WORKER_PRIMES = tuple(int(p) for p in primes)
    _WORKER_LOG_X = max(float(log_x), 1.0)
    _WORKER_LOWWHEEL_SETS = tuple(tuple(int(q) for q in s) for s in lowwheel_sets)
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
            acc = -math.inf
            break
        acc += math.log(f)
    val = float(acc)
    if _WORKER_CACHE_MAXSIZE <= 0 or len(_WORKER_SINGULAR_CACHE) < _WORKER_CACHE_MAXSIZE:
        _WORKER_SINGULAR_CACHE[key] = val
    else:
        _WORKER_SINGULAR_CACHE.clear()
        _WORKER_SINGULAR_CACHE[key] = val
    return val


def _omega_cluster_for_pair(g1: int, g2: int) -> Dict[str, float]:
    """Compute CHL2 path, Bernoulli, and low-wheel CHL3 factors for one pair."""
    g1 = int(g1)
    g2 = int(g2)
    L = max(float(_WORKER_LOG_X), 1.0)
    inv_log = 1.0 / L

    log_s3 = _worker_singular_log((0, g1, g1 + g2))
    offsets = list(even_interior_offsets(g2))
    n_int = len(offsets)

    out: Dict[str, float] = {
        "n_even_interior": float(n_int),
        "omega_path_poisson": 0.0,
        "logE_chl2_path": 0.0,
        "omega_path_bernoulli": 0.0,
        "logE_bernoulli_path": 0.0,
        "bernoulli_clipped_pu": 0.0,
        "max_pu": 0.0,
    }

    if not math.isfinite(log_s3) or n_int == 0:
        for low in _WORKER_LOWWHEEL_SETS:
            label = lowwheel_label(low)
            out[f"omega_lowwheel2_{label}"] = 0.0
            out[f"logE_lowwheel2_{label}"] = 0.0
            out[f"omega_self_lowwheel2_{label}"] = 0.0
            out[f"logE_self_lowwheel2_{label}"] = 0.0
            out[f"self_term_{label}"] = 0.0
            out[f"kappa_sum_{label}"] = 0.0
            out[f"active_residue_pairs_{label}"] = 0.0
            out[f"nonzero_h5_pairs_{label}"] = 0.0
            out[f"hard_zero_h5_pairs_{label}"] = 0.0
            out[f"hard_zero_h5_weight_{label}"] = 0.0
            out[f"hard_zero_h5_kappa_mass_{label}"] = 0.0
            out[f"kappa_negative_mass_{label}"] = 0.0
            out[f"kappa_positive_mass_{label}"] = 0.0
            out[f"skipped_zero_h4_pairs_{label}"] = 0.0
            out[f"M_low_{label}"] = float(math.prod(low) if low else 1)
        return out

    # First-order probabilities p_u.
    p_by_u: Dict[int, float] = {}
    omega1 = 0.0
    omega_bernoulli = 0.0
    clipped = 0
    max_p = 0.0
    eps = 1e-12
    for u in offsets:
        log_s4 = _worker_singular_log((0, g1, g1 + u, g1 + g2))
        if math.isfinite(log_s4):
            p_u = math.exp(log_s4 - log_s3) * inv_log
        else:
            p_u = 0.0
        if p_u < 0.0:
            p_u = 0.0
            clipped += 1
        max_p = max(max_p, p_u)
        p_by_u[u] = p_u
        omega1 += p_u
        p_clip = p_u
        if p_clip >= 1.0:
            p_clip = 1.0 - eps
            clipped += 1
        omega_bernoulli += -math.log(max(eps, 1.0 - p_clip))

    out["omega_path_poisson"] = float(omega1)
    out["logE_chl2_path"] = -float(omega1)
    out["omega_path_bernoulli"] = float(omega_bernoulli)
    out["logE_bernoulli_path"] = -float(omega_bernoulli)
    out["bernoulli_clipped_pu"] = float(clipped)
    out["max_pu"] = float(max_p)

    # Low-wheel second-order cumulants.
    #
    # IMPORTANT CHL3/Möbius correction:
    # The previous aggregate coupling used full-horizon p_u masses and a
    # low-wheel coupling factor.  That captures some signal, but it dilutes the
    # strict low-wheel hard zeros that are decisive in q=3.  Here the base
    # CHL2 intensity omega1 remains full-horizon, while the cross-cumulant is
    # computed exactly on the selected low wheel.  The sum is still exact over
    # all interior pairs because low-wheel singular ratios depend only on the
    # residues of u and v modulo M_low; we aggregate by residue classes rather
    # than looping over O(g2^2) concrete pairs.
    for low in _WORKER_LOWWHEEL_SETS:
        label = lowwheel_label(low)
        if not low:
            self_term = 0.5 * sum(p * p for p in p_by_u.values())
            omega_self = omega1 + self_term
            out[f"omega_lowwheel2_{label}"] = float(omega1)
            out[f"logE_lowwheel2_{label}"] = -float(omega1)
            out[f"omega_self_lowwheel2_{label}"] = float(omega_self)
            out[f"logE_self_lowwheel2_{label}"] = -float(omega_self)
            out[f"self_term_{label}"] = float(self_term)
            out[f"kappa_sum_{label}"] = 0.0
            out[f"active_residue_pairs_{label}"] = 0.0
            out[f"nonzero_h5_pairs_{label}"] = 0.0
            out[f"hard_zero_h5_pairs_{label}"] = 0.0
            out[f"hard_zero_h5_weight_{label}"] = 0.0
            out[f"hard_zero_h5_kappa_mass_{label}"] = 0.0
            out[f"kappa_negative_mass_{label}"] = 0.0
            out[f"kappa_positive_mass_{label}"] = 0.0
            out[f"skipped_zero_h4_pairs_{label}"] = 0.0
            out[f"M_low_{label}"] = 1.0
            out[f"mobius_low_pair_count_{label}"] = 0.0
            out[f"mobius_low_kappa_method_{label}"] = 1.0
            continue

        M = 1
        for p_low in low:
            M *= int(p_low)

        h3_res = (0 % M, g1 % M, (g1 + g2) % M)
        s3_low = lowwheel_singular_from_residues(h3_res, 3, low)

        # Low-wheel interior intensities p_u^low and counts by residue.
        # N counts the exact number of concrete interior offsets in each
        # low-wheel residue.  S_low and Q_low are sums of p_u^low and
        # (p_u^low)^2; these are sufficient to reconstruct exactly the sum over
        # all unordered pairs u < v for any function depending only on residues.
        N = {r: 0 for r in range(M)}
        S_low = {r: 0.0 for r in range(M)}
        Q_low = {r: 0.0 for r in range(M)}
        skipped_zero_h4_pairs = 0

        if s3_low > 0.0:
            for u in offsets:
                r = int(u % M)
                N[r] += 1
                h4_r = (0 % M, g1 % M, (g1 + r) % M, (g1 + g2) % M)
                s4_r = lowwheel_singular_from_residues(h4_r, 4, low)
                if s4_r <= 0.0:
                    # Individual interior candidate inadmissible under the low
                    # wheel.  Its p_u^low is zero, and it contributes no pair
                    # covariance mass except through impossible H5 pairs, which
                    # also have zero p_uv and zero p_u p_v.
                    skipped_zero_h4_pairs += 1
                    continue
                p_low_u = inv_log * (s4_r / s3_low)
                if p_low_u < 0.0 or not math.isfinite(p_low_u):
                    p_low_u = 0.0
                S_low[r] += p_low_u
                Q_low[r] += p_low_u * p_low_u

        residues = [r for r in range(M) if N[r] > 0]
        kappa_sum = 0.0
        active_pairs = 0.0
        hard_zero_h5_pairs = 0.0
        hard_zero_h5_weight = 0.0
        hard_zero_h5_kappa_mass = 0.0
        nonzero_h5_pairs = 0.0
        kappa_negative_mass = 0.0
        kappa_positive_mass = 0.0

        if s3_low > 0.0:
            inv_log2 = inv_log * inv_log
            for i, r in enumerate(residues):
                for s_res in residues[i:]:
                    if r != s_res:
                        pair_count = float(N[r] * N[s_res])
                        product_mass = S_low[r] * S_low[s_res]
                    else:
                        pair_count = float(N[r] * (N[r] - 1) // 2)
                        product_mass = 0.5 * (S_low[r] * S_low[r] - Q_low[r])

                    if pair_count <= 0.0:
                        continue

                    h5_rs = (
                        0 % M,
                        g1 % M,
                        (g1 + r) % M,
                        (g1 + s_res) % M,
                        (g1 + g2) % M,
                    )

                    # Strict low-wheel Möbius zero: if H5 covers every residue
                    # class modulo any low prime, S_low(H5)=0 exactly.  Do not
                    # skip this pair; it contributes -p_u p_v to the cumulant.
                    if lowwheel_is_inadmissible(h5_rs, low):
                        p_uv_low = 0.0
                        hard_zero_h5_pairs += pair_count
                        hard_zero_h5_weight += product_mass
                        nonzero = False
                    else:
                        s5 = lowwheel_singular_from_residues(h5_rs, 5, low)
                        p_uv_low = inv_log2 * (s5 / s3_low) if s5 > 0.0 else 0.0
                        nonzero = p_uv_low > 0.0
                        if nonzero:
                            nonzero_h5_pairs += pair_count

                    kappa_contrib = pair_count * p_uv_low - product_mass
                    if not nonzero and product_mass > 0.0:
                        hard_zero_h5_kappa_mass += kappa_contrib
                    if kappa_contrib < 0.0:
                        kappa_negative_mass += kappa_contrib
                    elif kappa_contrib > 0.0:
                        kappa_positive_mass += kappa_contrib
                    kappa_sum += kappa_contrib
                    active_pairs += pair_count

        # Base self-interaction is still computed from the full-horizon p_u,
        # because it corrects the Bernoulli self term of the CHL2 path intensity.
        self_term = 0.5 * sum(p * p for p in p_by_u.values())
        omega2 = omega1 - kappa_sum
        omega_self = omega1 + self_term - kappa_sum
        out[f"omega_lowwheel2_{label}"] = float(omega2)
        out[f"logE_lowwheel2_{label}"] = -float(omega2)
        out[f"omega_self_lowwheel2_{label}"] = float(omega_self)
        out[f"logE_self_lowwheel2_{label}"] = -float(omega_self)
        out[f"self_term_{label}"] = float(self_term)
        out[f"kappa_sum_{label}"] = float(kappa_sum)
        out[f"active_residue_pairs_{label}"] = float(active_pairs)
        out[f"nonzero_h5_pairs_{label}"] = float(nonzero_h5_pairs)
        out[f"hard_zero_h5_pairs_{label}"] = float(hard_zero_h5_pairs)
        out[f"hard_zero_h5_weight_{label}"] = float(hard_zero_h5_weight)
        out[f"hard_zero_h5_kappa_mass_{label}"] = float(hard_zero_h5_kappa_mass)
        out[f"kappa_negative_mass_{label}"] = float(kappa_negative_mass)
        out[f"kappa_positive_mass_{label}"] = float(kappa_positive_mass)
        out[f"skipped_zero_h4_pairs_{label}"] = float(skipped_zero_h4_pairs)
        out[f"M_low_{label}"] = float(M)
        out[f"mobius_low_pair_count_{label}"] = float(active_pairs)
        out[f"mobius_low_kappa_method_{label}"] = 2.0

    return out


def _cluster_cache_chunk(task: Tuple[int, List[Tuple[int, int]]]) -> Tuple[int, List[dict]]:
    chunk_id, pairs = task
    rows: List[dict] = []
    for g1, g2 in pairs:
        row = {"g1": int(g1), "g2": int(g2)}
        row.update(_omega_cluster_for_pair(int(g1), int(g2)))
        rows.append(row)
    return chunk_id, rows


def _iter_chunks(seq: Sequence[Tuple[int, int]], chunk_size: int) -> Iterable[List[Tuple[int, int]]]:
    chunk_size = max(1, int(chunk_size))
    for i in range(0, len(seq), chunk_size):
        yield list(seq[i:i + chunk_size])


def auto_cluster_chunk_size(
    n_pairs: int,
    workers: int,
    requested_chunk_size: int,
    target_tasks_per_worker: int = 6,
    min_chunk_size: int = 32,
    max_chunk_size: int = 512,
) -> int:
    """Choose a cluster-cache chunk size that keeps all workers busy.

    The earlier experimental default used large chunks (for example 5000).
    On B01 there are only about 6--7k unique pair states, so that default
    creates only two tasks and leaves most CPU cores idle.  A requested
    chunk size > 0 is respected exactly; requested_chunk_size <= 0 enables
    automatic sizing.
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


def collect_unique_pairs(block_files: Sequence[Tuple[int, Path]], source: str) -> pd.DataFrame:
    frames = []
    chosen = block_files if source == "all" else block_files[:1]
    for b, path in chosen:
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"[CHL3] scanning unique pairs from block {b}: {path}", flush=True)
        frames.append(pd.read_csv(path, usecols=["g1", "g2"]).drop_duplicates())
    if not frames:
        return pd.DataFrame(columns=["g1", "g2"])
    return pd.concat(frames, ignore_index=True).drop_duplicates().sort_values(["g1", "g2"]).reset_index(drop=True)


def compute_cluster_cache_parallel(
    pairs_df: pd.DataFrame,
    primes: Sequence[int],
    log_x: float,
    lowwheel_sets: Sequence[Sequence[int]],
    workers: int,
    chunk_size: int,
    cache_maxsize: int,
    target_tasks_per_worker: int = 6,
) -> pd.DataFrame:
    pairs_df = pairs_df[["g1", "g2"]].drop_duplicates().sort_values(["g1", "g2"]).reset_index(drop=True)
    pairs = [(int(a), int(b)) for a, b in pairs_df[["g1", "g2"]].itertuples(index=False, name=None)]
    workers = max(1, int(workers))
    effective_chunk_size = auto_cluster_chunk_size(
        n_pairs=len(pairs),
        workers=workers,
        requested_chunk_size=int(chunk_size),
        target_tasks_per_worker=int(target_tasks_per_worker),
    )
    tasks = [(i, chunk) for i, chunk in enumerate(_iter_chunks(pairs, effective_chunk_size))]
    print(
        f"[CHL3] unique pairs={len(pairs):,}; chunks={len(tasks):,}; "
        f"workers={workers}; cluster_chunk_size={effective_chunk_size}; "
        f"lowwheel={list(map(list, lowwheel_sets))}",
        flush=True,
    )
    t0 = time.time()
    results: List[Tuple[int, List[dict]]] = []
    if workers == 1:
        _cluster_worker_init(tuple(primes), float(log_x), lowwheel_sets, int(cache_maxsize))
        for task in tasks:
            results.append(_cluster_cache_chunk(task))
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_cluster_worker_init, initargs=(tuple(primes), float(log_x), tuple(tuple(s) for s in lowwheel_sets), int(cache_maxsize))) as ex:
            futs = [ex.submit(_cluster_cache_chunk, task) for task in tasks]
            done = 0
            for fut in as_completed(futs):
                results.append(fut.result())
                done += 1
                if done == len(tasks) or done % max(1, len(tasks) // 20) == 0:
                    print(f"[CHL3] completed {done:,}/{len(tasks):,} chunks ({100.0 * done / len(tasks):.1f}%)", flush=True)
    results.sort(key=lambda x: x[0])
    rows: List[dict] = []
    for _, part in results:
        rows.extend(part)
    df = pd.DataFrame(rows)
    print(f"[CHL3] cluster cache computed in {time.time() - t0:.1f}s", flush=True)
    return df


# ---------------------------------------------------------------------------
# Conditional likelihood audit
# ---------------------------------------------------------------------------


def analyze_block_chl3(
    df_all: pd.DataFrame,
    block: int,
    primes: Sequence[int],
    log_x: float,
    filters: Sequence[str],
    cluster_cache_file: Path,
    lowwheel_sets: Sequence[Sequence[int]],
) -> Tuple[List[dict], pd.DataFrame]:
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

    _, _, log_ratio_all, adm3_all = chl2.singular_logs_for_pairs(df_all["g1"].to_numpy(), df_all["g2"].to_numpy(), primes)
    df_all["_log_ratio"] = log_ratio_all
    df_all["_adm3"] = adm3_all

    all_g2 = sorted(df_all["g2"].astype(int).unique())
    gap_excl_map, gap_excl_diag = chl2.compute_gap_exclusion_logs(all_g2, primes, log_x)
    cramer_excl_map = chl2.compute_cramer_exclusion_logs(all_g2, log_x)
    df_all["_logE_gap"] = df_all["g2"].astype(int).map(gap_excl_map).astype(float)
    df_all["_logE_cramer"] = df_all["g2"].astype(int).map(cramer_excl_map).astype(float)

    cache_cols = ["g1", "g2", "logE_chl2_path", "logE_bernoulli_path"]
    for low in lowwheel_sets:
        label = lowwheel_label(low)
        cache_cols.append(f"logE_lowwheel2_{label}")
        cache_cols.append(f"logE_self_lowwheel2_{label}")
    telemetry_prefixes = (
        "omega_lowwheel2_", "omega_self_lowwheel2_", "self_term_",
        "kappa_sum_", "active_residue_pairs_", "nonzero_h5_pairs_",
        "hard_zero_h5_pairs_", "hard_zero_h5_weight_",
        "hard_zero_h5_kappa_mass_", "kappa_negative_mass_",
        "kappa_positive_mass_", "skipped_zero_h4_pairs_",
    )
    cache_df = pd.read_csv(
        cluster_cache_file,
        usecols=lambda c: c in set(cache_cols)
        or c in {"omega_path_poisson", "omega_path_bernoulli"}
        or c.startswith(telemetry_prefixes)
        or c in {"g1", "g2", "n_even_interior", "max_pu", "bernoulli_clipped_pu"},
    )
    df_all = df_all.merge(cache_df, on=["g1", "g2"], how="left")
    if df_all["logE_chl2_path"].isna().any():
        raise ValueError(f"Block {block}: missing cluster-cache rows after merge")

    diag_cols = [c for c in cache_df.columns if c not in ("g1", "g2")]
    diag = cache_df.head(10000).copy().assign(block=block, kind="cluster_cache_sample")
    gap_excl_diag = gap_excl_diag.assign(block=block, kind="gap_endpoint")
    diag = pd.concat([gap_excl_diag, diag], ignore_index=True, sort=False)

    evals = []
    for fname in filters:
        mask = chl2.filter_mask(df_all, fname)
        df = df_all.loc[mask].reset_index(drop=True)
        if df.empty or df["H"].sum() <= 0:
            continue
        log_ratio = df["_log_ratio"].to_numpy(np.float64)
        adm3 = np.isfinite(log_ratio)
        zero_cond = np.where(adm3, 0.0, -np.inf)
        logE_gap = df["_logE_gap"].to_numpy(np.float64)
        logE_cramer = df["_logE_cramer"].to_numpy(np.float64)
        logE_path = df["logE_chl2_path"].to_numpy(np.float64)
        logE_bernoulli = df["logE_bernoulli_path"].to_numpy(np.float64)

        # Conditional / Markov models.
        evals.append(chl2.eval_conditional_model(df, log_ratio, "CHL1_ratio_only_cond_eta", fname, block).__dict__)
        evals.append(chl2.eval_conditional_model(df, np.where(adm3, log_ratio + logE_gap, -np.inf), "CHL2_gap_excl_cond_eta", fname, block).__dict__)
        evals.append(chl2.eval_conditional_model(df, np.where(adm3, log_ratio + logE_path, -np.inf), "CHL2_path_excl_cond_eta", fname, block).__dict__)
        evals.append(chl2.eval_conditional_model(df, np.where(adm3, log_ratio + logE_bernoulli, -np.inf), "CHL2_bernoulli_path_cond_eta", fname, block).__dict__)
        for low in lowwheel_sets:
            label = lowwheel_label(low)
            col = f"logE_lowwheel2_{label}"
            logE_low = df[col].to_numpy(np.float64)
            evals.append(chl2.eval_conditional_model(df, np.where(adm3, log_ratio + logE_low, -np.inf), f"CHL3_lowwheel2_{label}_cond_eta", fname, block).__dict__)
            self_col = f"logE_self_lowwheel2_{label}"
            if self_col in df.columns:
                logE_self = df[self_col].to_numpy(np.float64)
                evals.append(chl2.eval_conditional_model(df, np.where(adm3, log_ratio + logE_self, -np.inf), f"CHL3_self_lowwheel2_{label}_cond_eta", fname, block).__dict__)
        evals.append(chl2.eval_conditional_model(df, np.where(adm3, log_ratio + logE_cramer, -np.inf), "CHL2_cramer_excl_cond_eta", fname, block).__dict__)
        evals.append(chl2.eval_conditional_model(df, np.where(adm3, zero_cond + logE_gap, -np.inf), "noPhi_gap_excl_cond_eta", fname, block).__dict__)
        evals.append(chl2.eval_conditional_model(df, zero_cond, "noPhi_cond_eta", fname, block).__dict__)

        # Order-zero baselines: candidate g2 only, no path-memory leakage.
        g_unique = np.array(sorted(df["g2"].astype(int).unique()), dtype=np.int64)
        log_s2_g, adm2_g = chl2.singular_log_s2_gap(g_unique, primes)
        logE_g = np.array([gap_excl_map[int(g)] for g in g_unique], dtype=np.float64)
        zero_g = np.where(adm2_g, 0.0, -np.inf)
        cramer_g = np.where(adm2_g, -g_unique.astype(np.float64) / log_x, -np.inf)
        cg_g = np.where(adm2_g, log_s2_g - g_unique.astype(np.float64) / log_x, -np.inf)
        cg_gap_g = np.where(adm2_g, log_s2_g + logE_g - g_unique.astype(np.float64) / log_x, -np.inf)
        evals.append(chl2.eval_order0_model(df, g_unique, log_s2_g, "HL2_order0_eta", fname, block, fit_eta=True).__dict__)
        evals.append(chl2.eval_order0_model(df, g_unique, np.where(adm2_g, log_s2_g + logE_g, -np.inf), "HL2_gap_excl_order0_eta", fname, block, fit_eta=True).__dict__)
        evals.append(chl2.eval_order0_model(df, g_unique, zero_g, "noPhi_order0_eta", fname, block, fit_eta=True).__dict__)
        evals.append(chl2.eval_order0_model(df, g_unique, cramer_g, "Cramer_order0_exp", fname, block, fit_eta=False).__dict__)
        evals.append(chl2.eval_order0_model(df, g_unique, cg_g, "Cramer_Granville_order0_exp", fname, block, fit_eta=False).__dict__)
        evals.append(chl2.eval_order0_model(df, g_unique, cg_gap_g, "Cramer_Granville_gap_excl_order0_exp", fname, block, fit_eta=False).__dict__)

    return evals, diag


def analyze_block_file_task(task: Tuple[int, str, Tuple[int, ...], float, Tuple[str, ...], str, Tuple[Tuple[int, ...], ...]]) -> Tuple[List[dict], pd.DataFrame]:
    b, path_str, primes, log_x, filters, cluster_cache_file, lowwheel_sets = task
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(path)
    print(f"[CHL3] worker reading block {b}: {path}", flush=True)
    df = pd.read_csv(path)
    return analyze_block_chl3(
        df,
        block=b,
        primes=primes,
        log_x=log_x,
        filters=filters,
        cluster_cache_file=Path(cluster_cache_file),
        lowwheel_sets=lowwheel_sets,
    )


def pairwise_gains_chl3(summary: pd.DataFrame) -> pd.DataFrame:
    baselines = [
        "CHL2_path_excl_cond_eta",
        "CHL2_gap_excl_cond_eta",
        "CHL2_bernoulli_path_cond_eta",
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
            for base in baselines:
                if base in d and model != base:
                    rows.append({
                        "filter": f,
                        "model": model,
                        "baseline": base,
                        "delta_loglik_model_minus_baseline": float(r["loglik_per_event"] - d[base]["loglik_per_event"]),
                        "delta_KL_baseline_minus_model": float(d[base]["conditional_kl"] - r["conditional_kl"]),
                    })
    return pd.DataFrame(rows).sort_values(["filter", "baseline", "delta_loglik_model_minus_baseline"], ascending=[True, True, False])


def memory_irreducibility_multi(block_df: pd.DataFrame, main_models: Sequence[str]) -> pd.DataFrame:
    frames = []
    for m in main_models:
        if m in set(block_df["model"].unique()):
            frames.append(chl2.memory_irreducibility(block_df, main_model=m).assign(main_model=m))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# OS absolute prime-residue diagnostic for CHL3 models
# ---------------------------------------------------------------------------


def load_parentwide_aggregate(block_files: Sequence[Tuple[int, Path]]) -> pd.DataFrame:
    frames = []
    for b, path in block_files:
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"[CHL3-OS] reading support for aggregate kernel from block {b}: {path}", flush=True)
        df = pd.read_csv(path, usecols=["g1", "g2", "H"])
        df["g1"] = df["g1"].astype(np.int64)
        df["g2"] = df["g2"].astype(np.int64)
        df["H"] = df["H"].astype(np.float64)
        frames.append(df)
    agg = pd.concat(frames, ignore_index=True).groupby(["g1", "g2"], as_index=False)["H"].sum()
    return agg


def build_chl3_kernel_mod_maps(
    block_files: Sequence[Tuple[int, Path]],
    primes: Sequence[int],
    log_x: float,
    model_name: str,
    mods: Sequence[int],
    cluster_cache_file: Path,
    lowwheel_sets: Sequence[Sequence[int]],
) -> Tuple[Dict[int, Dict[int, np.ndarray]], Dict[str, float]]:
    df = load_parentwide_aggregate(block_files)

    # Delegate order-zero baselines to CHL2 helper.
    order0_names = {
        "Cramer_order0_exp",
        "Cramer_Granville_order0_exp",
        "Cramer_Granville_gap_excl_order0_exp",
        "HL2_order0_eta",
        "HL2_gap_excl_order0_eta",
        "noPhi_order0_eta",
    }
    if model_name in order0_names:
        return chl2.build_order0_kernel_mod_maps(df, primes, log_x, model_name, mods)

    _, _, log_ratio, adm3 = chl2.singular_logs_for_pairs(df["g1"].to_numpy(), df["g2"].to_numpy(), primes)
    all_g2 = sorted(df["g2"].astype(int).unique())
    gap_excl_map, _ = chl2.compute_gap_exclusion_logs(all_g2, primes, log_x)
    cramer_excl_map = chl2.compute_cramer_exclusion_logs(all_g2, log_x)
    logE_gap = df["g2"].astype(int).map(gap_excl_map).astype(float).to_numpy()
    logE_cramer = df["g2"].astype(int).map(cramer_excl_map).astype(float).to_numpy()
    zero = np.where(np.isfinite(log_ratio), 0.0, -np.inf)

    cache_needed = ["g1", "g2", "logE_chl2_path", "logE_bernoulli_path"]
    for low in lowwheel_sets:
        label = lowwheel_label(low)
        cache_needed.append(f"logE_lowwheel2_{label}")
        cache_needed.append(f"logE_self_lowwheel2_{label}")
    cache_df = pd.read_csv(cluster_cache_file, usecols=lambda c: c in set(cache_needed))
    df = df.merge(cache_df, on=["g1", "g2"], how="left")
    if df["logE_chl2_path"].isna().any():
        raise ValueError("Aggregate OS support has pairs missing from CHL3 cluster cache")

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
        log_base = np.where(adm3, log_ratio + df["logE_chl2_path"].to_numpy(np.float64), -np.inf)
    elif model_name == "CHL2_bernoulli_path_cond_eta":
        log_base = np.where(adm3, log_ratio + df["logE_bernoulli_path"].to_numpy(np.float64), -np.inf)
    elif model_name.startswith("CHL3_lowwheel2_") and model_name.endswith("_cond_eta"):
        label = model_name[len("CHL3_lowwheel2_"):-len("_cond_eta")]
        col = f"logE_lowwheel2_{label}"
        if col not in df.columns:
            raise ValueError(f"CHL3 model {model_name} requires missing cache column {col}")
        log_base = np.where(adm3, log_ratio + df[col].to_numpy(np.float64), -np.inf)
    elif model_name.startswith("CHL3_self_lowwheel2_") and model_name.endswith("_cond_eta"):
        label = model_name[len("CHL3_self_lowwheel2_"):-len("_cond_eta")]
        col = f"logE_self_lowwheel2_{label}"
        if col not in df.columns:
            raise ValueError(f"CHL3 self+cluster model {model_name} requires missing cache column {col}")
        log_base = np.where(adm3, log_ratio + df[col].to_numpy(np.float64), -np.inf)
    else:
        raise ValueError(f"Unsupported CHL3 OS model {model_name}")

    obs = df[df["H"] > 0]
    target_mean = float((obs["H"] * obs["g2"]).sum() / obs["H"].sum()) if len(obs) else float("nan")
    eta, mean_model = chl2.solve_eta_conditional(df, log_base, target_mean)
    logp = chl2.conditional_log_probs(df, log_base, eta)

    maps: Dict[int, Dict[int, np.ndarray]] = {int(q): {} for q in mods}
    df_tmp = df.assign(_logp=logp)
    for g1, grp in df_tmp.groupby("g1"):
        lp = grp["_logp"].to_numpy(np.float64)
        finite = np.isfinite(lp)
        if not finite.any():
            continue
        probs = np.exp(lp[finite])
        g2vals = grp["g2"].to_numpy(np.int64)[finite]
        psum = float(probs.sum())
        if psum <= 0:
            continue
        probs = probs / psum
        for q in mods:
            arr = np.bincount((g2vals % int(q)).astype(np.int64), weights=probs, minlength=int(q)).astype(float)
            s = arr.sum()
            maps[int(q)][int(g1)] = arr / s if s > 0 else arr
    meta = {
        "eta": float(eta),
        "target_mean_g2": float(target_mean),
        "mean_g2_model": float(mean_model),
        "support_rows": float(len(df)),
        "empirical_events": float(df["H"].sum()),
    }
    return maps, meta


def pearson_chi_square_fast(obs_counts: np.ndarray, exp_counts: np.ndarray, row_counts: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Pearson chi-square table diagnostic without SciPy dependency.

    This intentionally omits p-values to keep the experimental CHL3 audit fast
    and dependency-light.  The statistic, degrees of freedom, and chi2/N are
    the quantities used for model comparison.
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


def run_os_prime_residue_test_chl3(
    prime_csv: Path,
    block_files: Sequence[Tuple[int, Path]],
    primes: Sequence[int],
    log_x: float,
    model_name: str,
    mods: Sequence[int],
    residue_mode: str,
    cluster_cache_file: Path,
    lowwheel_sets: Sequence[Sequence[int]],
    max_transitions: int,
    chunksize: int,
    outdir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not prime_csv.exists():
        raise FileNotFoundError(prime_csv)
    mods = [int(q) for q in mods]
    kernel_maps, meta = build_chl3_kernel_mod_maps(
        block_files=block_files,
        primes=primes,
        log_x=log_x,
        model_name=model_name,
        mods=mods,
        cluster_cache_file=cluster_cache_file,
        lowwheel_sets=lowwheel_sets,
    )
    counts = {q: np.zeros((len(chl2.reduced_residues_for_mod(q, residue_mode)), len(chl2.reduced_residues_for_mod(q, residue_mode))), dtype=np.float64) for q in mods}
    preds = {q: np.zeros_like(counts[q], dtype=np.float64) for q in mods}
    row_index = {q: {r: i for i, r in enumerate(chl2.reduced_residues_for_mod(q, residue_mode))} for q in mods}
    residues = {q: chl2.reduced_residues_for_mod(q, residue_mode) for q in mods}
    skipped_no_kernel = Counter()
    skipped_residue = Counter()
    used = Counter()

    prevprev: Optional[int] = None
    prev: Optional[int] = None
    total_transitions = 0
    stop = False
    print(f"[CHL3-OS] streaming prime sequence: {prime_csv}", flush=True)
    for vals in chl2.stream_prime_values(prime_csv, chunksize=chunksize):
        for pval in vals:
            pcur = int(pval)
            if prevprev is not None and prev is not None:
                g_prev = int(prev - prevprev)
                for q in mods:
                    b = int(prev % q)
                    a = int(pcur % q)
                    if b not in row_index[q] or a not in row_index[q]:
                        skipped_residue[q] += 1
                        continue
                    km = kernel_maps.get(q, {}).get(g_prev)
                    if km is None:
                        skipped_no_kernel[q] += 1
                        continue
                    bi = row_index[q][b]
                    ai = row_index[q][a]
                    counts[q][bi, ai] += 1.0
                    for d, prob in enumerate(km):
                        if prob <= 0:
                            continue
                        target = (b + d) % q
                        j = row_index[q].get(target)
                        if j is not None:
                            preds[q][bi, j] += float(prob)
                    used[q] += 1
                total_transitions += 1
                if max_transitions and total_transitions >= max_transitions:
                    stop = True
                    break
            prevprev, prev = prev, pcur
        if stop:
            break

    rows = []
    matrix_rows = []
    for q in mods:
        emp, row_counts = chl2.normalize_rows_from_counts(counts[q])
        pred, _pred_row_counts = chl2.normalize_rows_from_counts(preds[q])
        chi = pearson_chi_square_fast(counts[q], preds[q], row_counts=row_counts)
        rc = chl2.row_cosine_weighted(emp, pred, row_counts)
        kl = chl2.matrix_kl_weighted(emp, pred, row_counts)
        l1 = 0.0
        total = float(row_counts.sum())
        if total > 0:
            for i in range(emp.shape[0]):
                l1 += (row_counts[i] / total) * float(np.sum(np.abs(emp[i] - pred[i])))
        sg_emp = chl2.spectral_gap_row_stochastic(emp)
        sg_pred = chl2.spectral_gap_row_stochastic(pred)
        diag_emp = float(np.average(np.diag(emp), weights=row_counts)) if row_counts.sum() > 0 else float("nan")
        diag_pred = float(np.average(np.diag(pred), weights=row_counts)) if row_counts.sum() > 0 else float("nan")
        uniform_diag = 1.0 / len(residues[q]) if residues[q] else float("nan")
        wrong_sign = False
        if np.isfinite(diag_emp) and np.isfinite(diag_pred) and np.isfinite(uniform_diag):
            wrong_sign = (diag_emp - uniform_diag) * (diag_pred - uniform_diag) < 0
        row = {
            "q": q,
            "model": model_name,
            "residue_mode": residue_mode,
            "n_used_transitions": int(used[q]),
            "skipped_no_kernel": int(skipped_no_kernel[q]),
            "skipped_residue": int(skipped_residue[q]),
            "row_cosine_weighted": rc,
            "weighted_KL_empirical_to_model": kl,
            "weighted_L1": float(l1),
            "spectral_gap_empirical": sg_emp,
            "spectral_gap_model": sg_pred,
            "spectral_gap_abs_error": abs(sg_emp - sg_pred) if np.isfinite(sg_emp) and np.isfinite(sg_pred) else float("nan"),
            "diagonal_probability_empirical": diag_emp,
            "diagonal_probability_model": diag_pred,
            "uniform_diagonal_probability": uniform_diag,
            "diagonal_wrong_sign_vs_uniform": bool(wrong_sign),
        }
        row.update(chi)
        row.update({f"kernel_{k}": v for k, v in meta.items()})
        rows.append(row)
        for i, b in enumerate(residues[q]):
            for j, a in enumerate(residues[q]):
                matrix_rows.append({
                    "q": q,
                    "model": model_name,
                    "from_residue_b": b,
                    "to_residue_a": a,
                    "observed_count": counts[q][i, j],
                    "expected_count_model": row_counts[i] * pred[i, j],
                    "empirical_probability": emp[i, j],
                    "model_probability": pred[i, j],
                    "row_count": row_counts[i],
                })
    summary = pd.DataFrame(rows)
    matrix = pd.DataFrame(matrix_rows)
    summary.to_csv(outdir / "chl3_os_prime_residue_summary.csv", index=False)
    matrix.to_csv(outdir / "chl3_os_prime_residue_transition_by_mod.csv", index=False)
    return summary, matrix




def run_os_prime_residue_test_chl3_multi(
    prime_csv: Path,
    block_files: Sequence[Tuple[int, Path]],
    primes: Sequence[int],
    log_x: float,
    model_names: Sequence[str],
    mods: Sequence[int],
    residue_mode: str,
    cluster_cache_file: Path,
    lowwheel_sets: Sequence[Sequence[int]],
    max_transitions: int,
    chunksize: int,
    outdir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fast absolute prime-residue OS diagnostic for multiple models.

    This is the performance-critical stage after the cluster cache has been
    built.  The previous implementation streamed the prime CSV and updated the
    model prediction one transition at a time, nested inside loops over
    q, model, and gap residues.  That is intentionally simple but very slow and
    appears CPU-light because much of the time is spent in Python iteration and
    gzip I/O.

    This implementation keeps the same mathematics and the same outputs, but
    processes each prime chunk vectorially:

    1. Build arrays of triples (p_{i-2}, p_{i-1}, p_i).
    2. For each modulus q, group transitions by (g_prev, from_residue,
       to_residue) using numpy.unique.
    3. For each model, add observed counts and predicted rows by group count,
       not event by event.

    The expected count written to the matrix CSV is exactly

        row_count * model_probability,

    so the Pearson chi-square can be recomputed directly from the CSV.
    """
    if not prime_csv.exists():
        raise FileNotFoundError(prime_csv)
    model_names = [str(m) for m in model_names]
    if not model_names:
        raise ValueError("model_names cannot be empty")
    mods = [int(q) for q in mods]

    # Build model kernels once.
    kernel_maps_by_model: Dict[str, Dict[int, Dict[int, np.ndarray]]] = {}
    meta_by_model: Dict[str, Dict[str, float]] = {}
    for model_name in model_names:
        print(f"[CHL3-OS] building kernel maps for {model_name}", flush=True)
        maps, meta = build_chl3_kernel_mod_maps(
            block_files=block_files,
            primes=primes,
            log_x=log_x,
            model_name=model_name,
            mods=mods,
            cluster_cache_file=cluster_cache_file,
            lowwheel_sets=lowwheel_sets,
        )
        kernel_maps_by_model[model_name] = maps
        meta_by_model[model_name] = meta

    residues = {q: chl2.reduced_residues_for_mod(q, residue_mode) for q in mods}
    row_index = {q: {r: i for i, r in enumerate(residues[q])} for q in mods}
    dense_index: Dict[int, np.ndarray] = {}
    for q in mods:
        idx = np.full(q, -1, dtype=np.int16)
        for i, r in enumerate(residues[q]):
            idx[int(r) % q] = i
        dense_index[q] = idx

    # Precompute P(a | b, g_prev, model, q) as a row-stochastic matrix for
    # every previous gap in the support.  This removes the innermost loop over
    # gap residues from the prime-streaming stage.
    transition_rows: Dict[str, Dict[int, Dict[int, np.ndarray]]] = {}
    for model_name in model_names:
        transition_rows[model_name] = {}
        for q in mods:
            n = len(residues[q])
            transition_rows[model_name][q] = {}
            for gp, km in kernel_maps_by_model[model_name].get(q, {}).items():
                mat = np.zeros((n, n), dtype=np.float64)
                for bi, b in enumerate(residues[q]):
                    for d, prob in enumerate(km):
                        if prob <= 0.0:
                            continue
                        j = row_index[q].get((int(b) + int(d)) % q)
                        if j is not None:
                            mat[bi, j] += float(prob)
                transition_rows[model_name][q][int(gp)] = mat

    counts = {
        model: {q: np.zeros((len(residues[q]), len(residues[q])), dtype=np.float64) for q in mods}
        for model in model_names
    }
    preds = {
        model: {q: np.zeros_like(counts[model][q], dtype=np.float64) for q in mods}
        for model in model_names
    }
    skipped_no_kernel = {model: Counter() for model in model_names}
    skipped_residue = {model: Counter() for model in model_names}
    used = {model: Counter() for model in model_names}

    history = np.array([], dtype=np.int64)
    total_transitions = 0
    chunk_no = 0
    t0 = time.time()
    print(f"[CHL3-OS] vectorized streaming prime sequence: {prime_csv}", flush=True)
    for vals in chl2.stream_prime_values(prime_csv, chunksize=chunksize):
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
                for model in model_names:
                    skipped_residue[model][q] += invalid
            if not np.any(valid):
                continue

            g_valid = g_prev_arr[valid].astype(np.int64, copy=False)
            bi_valid = bi_all[valid].astype(np.int64, copy=False)
            ai_valid = ai_all[valid].astype(np.int64, copy=False)

            # Encode (g_prev, bi, ai) in a single int64 key.
            key = g_valid * (n * n) + bi_valid * n + ai_valid
            uniq, freq = np.unique(key, return_counts=True)

            decoded_gp = uniq // (n * n)
            rem = uniq - decoded_gp * (n * n)
            decoded_bi = rem // n
            decoded_ai = rem - decoded_bi * n

            for model in model_names:
                row_cache = transition_rows[model][q]
                for gp, bi, ai, cnt in zip(decoded_gp, decoded_bi, decoded_ai, freq):
                    mat = row_cache.get(int(gp))
                    if mat is None:
                        skipped_no_kernel[model][q] += int(cnt)
                        continue
                    c = float(cnt)
                    counts[model][q][int(bi), int(ai)] += c
                    preds[model][q][int(bi), :] += c * mat[int(bi), :]
                    used[model][q] += int(cnt)

        total_transitions += n_events
        chunk_no += 1
        if chunk_no % 5 == 0:
            elapsed = time.time() - t0
            rate = total_transitions / elapsed if elapsed > 0 else 0.0
            print(f"[CHL3-OS] streamed {total_transitions:,} transitions in {elapsed:.1f}s ({rate:,.0f}/s)", flush=True)

        history = arr[-2:].copy()
        if max_transitions and total_transitions >= int(max_transitions):
            break

    rows = []
    matrix_rows = []
    for model_name in model_names:
        for q in mods:
            emp, row_counts = chl2.normalize_rows_from_counts(counts[model_name][q])
            pred, _pred_row_counts = chl2.normalize_rows_from_counts(preds[model_name][q])
            chi = pearson_chi_square_fast(counts[model_name][q], preds[model_name][q], row_counts=row_counts)
            rc = chl2.row_cosine_weighted(emp, pred, row_counts)
            kl = chl2.matrix_kl_weighted(emp, pred, row_counts)
            l1 = 0.0
            total = float(row_counts.sum())
            if total > 0:
                for i in range(emp.shape[0]):
                    l1 += (row_counts[i] / total) * float(np.sum(np.abs(emp[i] - pred[i])))
            sg_emp = chl2.spectral_gap_row_stochastic(emp)
            sg_pred = chl2.spectral_gap_row_stochastic(pred)
            diag_emp = float(np.average(np.diag(emp), weights=row_counts)) if row_counts.sum() > 0 else float("nan")
            diag_pred = float(np.average(np.diag(pred), weights=row_counts)) if row_counts.sum() > 0 else float("nan")
            uniform_diag = 1.0 / len(residues[q]) if residues[q] else float("nan")
            wrong_sign = False
            if np.isfinite(diag_emp) and np.isfinite(diag_pred) and np.isfinite(uniform_diag):
                wrong_sign = (diag_emp - uniform_diag) * (diag_pred - uniform_diag) < 0
            row = {
                "q": q,
                "model": model_name,
                "residue_mode": residue_mode,
                "n_used_transitions": int(used[model_name][q]),
                "skipped_no_kernel": int(skipped_no_kernel[model_name][q]),
                "skipped_residue": int(skipped_residue[model_name][q]),
                "row_cosine_weighted": rc,
                "weighted_KL_empirical_to_model": kl,
                "weighted_L1": float(l1),
                "spectral_gap_empirical": sg_emp,
                "spectral_gap_model": sg_pred,
                "spectral_gap_abs_error": abs(sg_emp - sg_pred) if np.isfinite(sg_emp) and np.isfinite(sg_pred) else float("nan"),
                "diagonal_probability_empirical": diag_emp,
                "diagonal_probability_model": diag_pred,
                "uniform_diagonal_probability": uniform_diag,
                "diagonal_wrong_sign_vs_uniform": bool(wrong_sign),
            }
            row.update(chi)
            row.update({f"kernel_{k}": v for k, v in meta_by_model[model_name].items()})
            rows.append(row)
            for i, b in enumerate(residues[q]):
                for j, a in enumerate(residues[q]):
                    matrix_rows.append({
                        "q": q,
                        "model": model_name,
                        "from_residue_b": b,
                        "to_residue_a": a,
                        "observed_count": counts[model_name][q][i, j],
                        "expected_count_model": row_counts[i] * pred[i, j],
                        "empirical_probability": emp[i, j],
                        "model_probability": pred[i, j],
                        "row_count": row_counts[i],
                    })
    summary = pd.DataFrame(rows)
    matrix = pd.DataFrame(matrix_rows)
    summary.to_csv(outdir / "chl3_os_prime_residue_summary.csv", index=False)
    matrix.to_csv(outdir / "chl3_os_prime_residue_transition_by_mod.csv", index=False)
    return summary, matrix


def q3_diagnostic_from_os(os_summary: Optional[pd.DataFrame]) -> pd.DataFrame:
    if os_summary is None or os_summary.empty:
        return pd.DataFrame()
    rows = []
    for _, r in os_summary.iterrows():
        q = int(r["q"])
        uniform = float(r["uniform_diagonal_probability"])
        diag_emp = float(r["diagonal_probability_empirical"])
        diag_mod = float(r["diagonal_probability_model"])
        rows.append({
            "q": q,
            "model": r.get("model", ""),
            "diag_empirical_minus_uniform": diag_emp - uniform,
            "diag_model_minus_uniform": diag_mod - uniform,
            "wrong_sign": bool(r.get("diagonal_wrong_sign_vs_uniform", False)),
            "pearson_chi2_per_transition": float(r.get("pearson_chi2_per_transition", float("nan"))),
            "row_cosine_weighted": float(r.get("row_cosine_weighted", float("nan"))),
            "interpretation": "primary q=3 target" if q == 3 else "non-degradation check",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def resolve_cluster_cache_path(outdir: Path, Y: int, log_x: float, lowwheel_sets: Sequence[Sequence[int]], explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit)
    labels = "_".join(lowwheel_label(s) for s in lowwheel_sets) or "none"
    return outdir / f"chl3_cluster_cache_Y{Y}_logx{int(round(log_x * 1_000_000))}_{labels}.csv.gz"




def expand_os_model_names(os_models_arg: str, os_model_arg: str, lowwheel_sets: Sequence[Sequence[int]]) -> List[str]:
    """Resolve OS model arguments to an explicit list."""
    default_primary = "CHL3_lowwheel2_p3_cond_eta" if (3,) in lowwheel_sets else f"CHL3_lowwheel2_{lowwheel_label(lowwheel_sets[0])}_cond_eta"
    all_models = [
        "CHL2_path_excl_cond_eta",
        "CHL2_bernoulli_path_cond_eta",
        *[f"CHL3_lowwheel2_{lowwheel_label(s)}_cond_eta" for s in lowwheel_sets],
        *[f"CHL3_self_lowwheel2_{lowwheel_label(s)}_cond_eta" for s in lowwheel_sets],
    ]
    arg = str(os_models_arg or "").strip()
    if arg:
        if arg.lower() == "all":
            return all_models
        return [x.strip() for x in arg.split(",") if x.strip()]
    if str(os_model_arg).strip().lower() == "auto":
        return [default_primary]
    return [str(os_model_arg).strip()]

def main() -> None:
    ap = argparse.ArgumentParser(description="CHL3 second-order low-wheel cluster audit")
    ap.add_argument("--config", required=True, help="Path to generated/config JSON")
    ap.add_argument("--root", default=".", help="Root directory containing input_dir from config")
    ap.add_argument("--blocks", default=None, help="Blocks to run, e.g. '1-10' or '1,2,3'")
    ap.add_argument("--output-dir", default="chl3_second_order_cluster_outputs")
    ap.add_argument("--pmax", type=int, default=None, help="Prime cutoff for singular series; default config pmax")
    ap.add_argument("--y-mode", choices=["pmax", "logx", "sqrtlogx"], default="pmax")
    ap.add_argument("--filters", default=",".join(chl2.FILTERS.keys()))
    ap.add_argument("--lowwheel-sets", default="3;3,5;3,5,7", help="Semicolon-separated low-wheel sets, e.g. '3;3,5;3,5,7'")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="Number of workers; 0 means all cores")
    ap.add_argument("--parallel-mode", choices=["auto", "blocks", "cluster", "none"], default="auto")
    ap.add_argument("--cluster-chunk-size", type=int, default=0, help="Cluster-cache chunk size; 0 enables automatic sizing to keep workers busy")
    ap.add_argument("--cluster-target-tasks-per-worker", type=int, default=6, help="Automatic cluster chunks target tasks per worker")
    ap.add_argument("--cache-maxsize", type=int, default=2_000_000)
    ap.add_argument("--cluster-cache-file", default=None)
    ap.add_argument("--cluster-cache-source", choices=["first", "all"], default="all")
    ap.add_argument("--reuse-cluster-cache", action="store_true")
    ap.add_argument("--prime-csv", default=None, help="Chronological prime CSV for absolute OS test; AUTO reads config real_prime_sequence")
    ap.add_argument("--os-prime-mods", default="3,5,7")
    ap.add_argument("--os-model", default="auto", help="Backward-compatible single OS model; default auto uses CHL3_lowwheel2_p3_cond_eta")
    ap.add_argument("--os-models", default="", help="Comma-separated OS models, or 'all' for CHL2/CHL3 main candidates. Overrides --os-model when provided.")
    ap.add_argument("--os-residue-mode", choices=["reduced", "all"], default="reduced")
    ap.add_argument("--os-max-transitions", type=int, default=0)
    ap.add_argument("--os-prime-chunksize", type=int, default=1_000_000)
    ap.add_argument("--require-os", action="store_true")
    ap.add_argument("--os-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    config_path = Path(args.config)
    root = Path(args.root)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    config = chl2.load_config(config_path)
    blocks = chl2.parse_blocks_arg(args.blocks, config.get("blocks", list(range(1, 11))))
    block_files = chl2.resolve_block_files(config, root, blocks)
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
    filters = [x.strip() for x in str(args.filters).split(",") if x.strip()]
    for f in filters:
        if f not in chl2.FILTERS:
            raise ValueError(f"Unknown filter {f}; available {sorted(chl2.FILTERS)}")
    lowwheel_sets = parse_lowwheel_sets(args.lowwheel_sets)
    workers = (os.cpu_count() or 1) if int(args.workers) == 0 else max(1, int(args.workers))

    cluster_cache_path = resolve_cluster_cache_path(outdir, Y, log_x, lowwheel_sets, args.cluster_cache_file)
    diag_rows = [{"block": b, "path": str(path), "exists": path.exists()} for b, path in block_files]
    pd.DataFrame(diag_rows).to_csv(outdir / "chl3_block_diagnostics.csv", index=False)

    if args.dry_run:
        print(json.dumps({
            "phase": "CHL3_SECOND_ORDER_CLUSTER_AUDIT",
            "Y": Y,
            "primes": primes,
            "lowwheel_sets": [list(s) for s in lowwheel_sets],
            "cluster_cache_file": str(cluster_cache_path),
            "blocks": blocks,
            "block_files": [str(p) for _, p in block_files],
        }, indent=2))
        return

    # Build or reuse cluster cache.
    cache_t0 = time.time()
    if cluster_cache_path.exists() and args.reuse_cluster_cache:
        print(f"[CHL3] reusing cluster cache: {cluster_cache_path}", flush=True)
    else:
        pair_df = collect_unique_pairs(block_files, source=args.cluster_cache_source)
        cluster_workers = workers if args.parallel_mode in ("auto", "cluster") else 1
        cache_df = compute_cluster_cache_parallel(
            pair_df,
            primes=primes,
            log_x=log_x,
            lowwheel_sets=lowwheel_sets,
            workers=cluster_workers,
            chunk_size=int(args.cluster_chunk_size),
            cache_maxsize=int(args.cache_maxsize),
            target_tasks_per_worker=int(args.cluster_target_tasks_per_worker),
        )
        cluster_cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_df.to_csv(cluster_cache_path, index=False)
        print(f"[CHL3] wrote cluster cache: {cluster_cache_path}", flush=True)
    cache_seconds = time.time() - cache_t0

    model_list = [
        "CHL2_path_excl_cond_eta",
        "CHL2_bernoulli_path_cond_eta",
        *[f"CHL3_lowwheel2_{lowwheel_label(s)}_cond_eta" for s in lowwheel_sets],
        *[f"CHL3_self_lowwheel2_{lowwheel_label(s)}_cond_eta" for s in lowwheel_sets],
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

    run_config = {
        "phase": "CHL3_SECOND_ORDER_CLUSTER_AUDIT",
        "experimental_branch": True,
        "config": str(config_path),
        "root": str(root),
        "blocks": blocks,
        "block_files": [str(p) for _, p in block_files],
        "Y_mode": args.y_mode,
        "Y": Y,
        "primes": primes,
        "log_x": log_x,
        "lowwheel_sets": [list(s) for s in lowwheel_sets],
        "workers": workers,
        "parallel_mode": args.parallel_mode,
        "cluster_cache_file": str(cluster_cache_path),
        "cluster_cache_source": args.cluster_cache_source,
        "filters": filters,
        "models": model_list,
        "primary_question": "Does a second-order low-wheel cumulant reduce the q=3 anomaly without degrading CHL2?",
    }
    with open(outdir / "chl3_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    # OS-only mode.
    def resolve_prime_path() -> Optional[Path]:
        return resolve_prime_csv_local(args.prime_csv, config, root, config_path=config_path) if args.prime_csv else None

    os_summary: Optional[pd.DataFrame] = None
    os_status = {
        "requested": bool(args.prime_csv),
        "prime_arg": args.prime_csv,
        "resolved_prime_csv": None,
        "candidate_prime_csv_paths": [str(x) for x in prime_csv_candidates(args.prime_csv, config, root, config_path=config_path)] if args.prime_csv else [],
        "ran": False,
        "skip_reason": None,
        "model": None,
        "model_names": None,
        "mods": [int(x.strip()) for x in str(args.os_prime_mods).split(",") if x.strip()],
        "summary_csv": None,
        "matrix_csv": None,
    }

    def run_os_if_requested() -> Optional[pd.DataFrame]:
        nonlocal os_status
        if not args.prime_csv:
            os_status["skip_reason"] = "--prime-csv was not provided"
            return None
        prime_path = resolve_prime_path()
        os_status["resolved_prime_csv"] = str(prime_path) if prime_path is not None else None
        if prime_path is None or not prime_path.exists():
            os_status["skip_reason"] = f"prime CSV not found: {prime_path}"
            with open(outdir / "chl3_os_prime_residue_status.json", "w", encoding="utf-8") as f:
                json.dump(os_status, f, indent=2)
            if args.require_os:
                raise RuntimeError(os_status["skip_reason"])
            return None
        os_model_names = expand_os_model_names(args.os_models, args.os_model, lowwheel_sets)
        os_status["model"] = os_model_names[0] if len(os_model_names) == 1 else "MULTI"
        os_status["model_names"] = os_model_names
        if len(os_model_names) == 1:
            summary, _matrix = run_os_prime_residue_test_chl3(
                prime_csv=prime_path,
                block_files=block_files,
                primes=primes,
                log_x=log_x,
                model_name=os_model_names[0],
                mods=os_status["mods"],
                residue_mode=args.os_residue_mode,
                cluster_cache_file=cluster_cache_path,
                lowwheel_sets=lowwheel_sets,
                max_transitions=int(args.os_max_transitions),
                chunksize=int(args.os_prime_chunksize),
                outdir=outdir,
            )
        else:
            summary, _matrix = run_os_prime_residue_test_chl3_multi(
                prime_csv=prime_path,
                block_files=block_files,
                primes=primes,
                log_x=log_x,
                model_names=os_model_names,
                mods=os_status["mods"],
                residue_mode=args.os_residue_mode,
                cluster_cache_file=cluster_cache_path,
                lowwheel_sets=lowwheel_sets,
                max_transitions=int(args.os_max_transitions),
                chunksize=int(args.os_prime_chunksize),
                outdir=outdir,
            )
        os_status["ran"] = True
        os_status["summary_csv"] = str(outdir / "chl3_os_prime_residue_summary.csv")
        os_status["matrix_csv"] = str(outdir / "chl3_os_prime_residue_transition_by_mod.csv")
        os_status["n_summary_rows"] = int(len(summary))
        with open(outdir / "chl3_os_prime_residue_status.json", "w", encoding="utf-8") as f:
            json.dump(os_status, f, indent=2)
        q3 = q3_diagnostic_from_os(summary)
        if not q3.empty:
            q3.to_csv(outdir / "chl3_q3_diagnostic.csv", index=False)
        return summary

    if args.os_only:
        os_summary = run_os_if_requested()
        with open(outdir / "chl3_interpretacion.md", "w", encoding="utf-8") as f:
            f.write(f"# CHL3 OS-only diagnostic\n\nRan: `{os_status.get('ran')}`. Model: `{os_status.get('model')}`.\n")
        return

    all_eval_dicts: List[dict] = []
    all_diag: List[pd.DataFrame] = []
    block_tasks = [
        (b, str(path), tuple(primes), float(log_x), tuple(filters), str(cluster_cache_path), tuple(tuple(s) for s in lowwheel_sets))
        for b, path in block_files
    ]
    use_block_parallel = workers > 1 and args.parallel_mode in ("auto", "blocks") and len(block_tasks) > 1
    eval_t0 = time.time()
    if use_block_parallel:
        print(f"[CHL3] evaluating {len(block_tasks)} blocks with {min(workers, len(block_tasks))} worker(s)", flush=True)
        with ProcessPoolExecutor(max_workers=min(workers, len(block_tasks))) as ex:
            futures = [ex.submit(analyze_block_file_task, task) for task in block_tasks]
            done = 0
            for fut in as_completed(futures):
                eval_dicts, diag = fut.result()
                all_eval_dicts.extend(eval_dicts)
                all_diag.append(diag)
                done += 1
                print(f"[CHL3] completed block task {done}/{len(block_tasks)}", flush=True)
    else:
        for task in block_tasks:
            eval_dicts, diag = analyze_block_file_task(task)
            all_eval_dicts.extend(eval_dicts)
            all_diag.append(diag)
    eval_seconds = time.time() - eval_t0

    block_df = pd.DataFrame(all_eval_dicts)
    block_df.to_csv(outdir / "chl3_metrics_by_block.csv", index=False)
    summary = chl2.summarize_metrics(block_df)
    summary.to_csv(outdir / "chl3_conditional_summary.csv", index=False)
    gains = pairwise_gains_chl3(summary)
    gains.to_csv(outdir / "chl3_pairwise_gains.csv", index=False)
    main_models = (
        ["CHL2_path_excl_cond_eta", "CHL2_bernoulli_path_cond_eta"]
        + [f"CHL3_lowwheel2_{lowwheel_label(s)}_cond_eta" for s in lowwheel_sets]
        + [f"CHL3_self_lowwheel2_{lowwheel_label(s)}_cond_eta" for s in lowwheel_sets]
    )
    mem = memory_irreducibility_multi(block_df, main_models)
    if not mem.empty:
        mem.to_csv(outdir / "chl3_memory_irreducibility.csv", index=False)
    if all_diag:
        pd.concat(all_diag, ignore_index=True, sort=False).to_csv(outdir / "chl3_cluster_diagnostics.csv", index=False)

    os_summary = run_os_if_requested()

    # Telemetry.
    cache_rows = 0
    try:
        cache_rows = int(sum(len(chunk) for chunk in pd.read_csv(cluster_cache_path, usecols=["g1"], chunksize=1_000_000)))
    except Exception:
        cache_rows = -1
    cache_column_means: Dict[str, float] = {}
    try:
        telemetry_cols = [
            c for c in pd.read_csv(cluster_cache_path, nrows=0).columns
            if c.startswith(("hard_zero_h5_", "kappa_negative_mass_", "kappa_positive_mass_", "kappa_sum_", "self_term_", "omega_lowwheel2_", "omega_self_lowwheel2_"))
        ]
        if telemetry_cols:
            tele_df = pd.read_csv(cluster_cache_path, usecols=telemetry_cols)
            cache_column_means = {f"mean_{c}": float(tele_df[c].mean()) for c in telemetry_cols if pd.api.types.is_numeric_dtype(tele_df[c])}
            cache_column_means.update({f"sum_{c}": float(tele_df[c].sum()) for c in telemetry_cols if c.startswith(("hard_zero_h5_", "kappa_negative_mass_", "kappa_positive_mass_")) and pd.api.types.is_numeric_dtype(tele_df[c])})
    except Exception as exc:
        cache_column_means = {"telemetry_read_error": str(exc)}

    telemetry = {
        "Y": Y,
        "lowwheel_sets": [list(s) for s in lowwheel_sets],
        "total_runtime_seconds": time.time() - t_start,
        "cluster_cache_seconds": cache_seconds,
        "block_eval_seconds": eval_seconds,
        "unique_pair_cache_file": str(cluster_cache_path),
        "cache_rows_estimate_chunks": cache_rows,
        "workers": workers,
        "models": model_list,
        "os_ran": bool(os_status.get("ran")),
        "os_model": os_status.get("model"),
        "os_model_names": os_status.get("model_names"),
        "cluster_cache_column_means_and_sums": cache_column_means,
    }
    with open(outdir / "chl3_cluster_telemetry.json", "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)

    # Interpretation.
    lines = []
    for f_name, grp in summary.groupby("filter"):
        best = grp.sort_values("loglik_per_event", ascending=False).iloc[0]
        lines.append(f"- `{f_name}`: best `{best['model']}` with loglik/event `{best['loglik_per_event']:.8g}`.")
    gain_lines = []
    for model in [m for m in main_models if m.startswith("CHL3_") or m == "CHL2_bernoulli_path_cond_eta"]:
        for f_name in filters:
            row = gains[(gains["filter"].eq(f_name)) & (gains["model"].eq(model)) & (gains["baseline"].eq("CHL2_path_excl_cond_eta"))]
            if not row.empty:
                r = row.iloc[0]
                gain_lines.append(f"- `{f_name}` `{model}` vs CHL2 path Δloglik/event `{r['delta_loglik_model_minus_baseline']:.8g}`.")
    os_text = "OS absolute prime-residue test was not requested."
    if os_summary is not None and not os_summary.empty:
        os_text = os_summary.to_string(index=False)
    interp = f"""# CHL3 second-order cluster audit — quick interpretation

This is an experimental branch.  The stable paper result remains CHL2 v1.4 unless a CHL3 model reduces the q=3 anomaly without degrading CHL2 likelihood and q=5,7 diagnostics.

## Best model by conditional log-likelihood

{os.linesep.join(lines)}

## CHL3/Bernoulli gains against CHL2 path

{os.linesep.join(gain_lines) if gain_lines else 'No CHL3-vs-CHL2 gain rows available.'}

## Absolute prime-residue OS diagnostic

{os_text}

## PASS rule

A CHL3 model should only be considered a replacement candidate if it reduces q=3 wrong-sign/chi-square, preserves q=5 and q=7 direction, and does not materially degrade conditional log-likelihood against CHL2_path_excl_cond_eta.
"""
    with open(outdir / "chl3_interpretacion.md", "w", encoding="utf-8") as f:
        f.write(interp)
    print(f"[CHL3] wrote outputs to {outdir}")


if __name__ == "__main__":
    main()
