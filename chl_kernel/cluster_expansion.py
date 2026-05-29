from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from math import exp, log
from typing import Iterable, Sequence


def even_interior_offsets(g2: int) -> list[int]:
    """Return even interior offsets u with 2 <= u <= g2-2."""
    if g2 <= 2:
        return []
    return list(range(2, g2, 2))


def h3_tuple(g1: int, g2: int) -> tuple[int, int, int]:
    """Endpoint triple H3 = {0, g1, g1+g2}."""
    return (0, g1, g1 + g2)


def h4_tuple(g1: int, g2: int, u: int) -> tuple[int, int, int, int]:
    """Endpoint triple plus one interior candidate."""
    return (0, g1, g1 + u, g1 + g2)


def h5_tuple(g1: int, g2: int, u: int, v: int) -> tuple[int, int, int, int, int]:
    """Endpoint triple plus two interior candidates."""
    return (0, g1, g1 + u, g1 + v, g1 + g2)

def omega_path_bernoulli(
    g1: int,
    g2: int,
    log_x: float,
    singular_ratio_h4_h3,
    eps: float = 1e-12,
) -> tuple[float, dict]:
    """
    Bernoulli no-interior correction.

    Replaces the first-order Poisson approximation sum(p_u)
    by -sum(log(1-p_u)). This does not introduce cross-dependence,
    but avoids the continuous-Poisson approximation for moderately
    large p_u.

    singular_ratio_h4_h3(g1, g2, u) must return S(H4(u))/S(H3).
    """
    inv_log = 1.0 / log_x
    omega = 0.0
    clipped = 0
    max_p = 0.0

    for u in even_interior_offsets(g2):
        p_u = inv_log * singular_ratio_h4_h3(g1, g2, u)
        max_p = max(max_p, p_u)
        if p_u >= 1.0:
            p_u = 1.0 - eps
            clipped += 1
        elif p_u < 0.0:
            p_u = 0.0
            clipped += 1
        omega += -log(1.0 - p_u)

    return omega, {"clipped_pu": clipped, "max_pu": max_p}

def omega_second_order_full(
    g1: int,
    g2: int,
    log_x: float,
    singular_ratio_h4_h3,
    singular_ratio_h5_h3,
) -> tuple[float, dict]:
    """
    Full second-order cluster correction.

    Omega^(2) = sum_u p_u - sum_{u<v} kappa_uv,
    where kappa_uv = p_uv - p_u p_v.
    """
    inv_log = 1.0 / log_x
    inv_log2 = inv_log * inv_log

    offsets = even_interior_offsets(g2)
    p = {}
    omega1 = 0.0

    for u in offsets:
        p_u = inv_log * singular_ratio_h4_h3(g1, g2, u)
        p[u] = p_u
        omega1 += p_u

    kappa_sum = 0.0
    pairs = 0
    max_abs_kappa = 0.0

    for u, v in combinations(offsets, 2):
        p_uv = inv_log2 * singular_ratio_h5_h3(g1, g2, u, v)
        kappa = p_uv - p[u] * p[v]
        kappa_sum += kappa
        max_abs_kappa = max(max_abs_kappa, abs(kappa))
        pairs += 1

    omega2 = omega1 - kappa_sum

    return omega2, {
        "omega1": omega1,
        "kappa_sum": kappa_sum,
        "pairs": pairs,
        "max_abs_kappa": max_abs_kappa,
    }


def lowwheel_covers_all_classes(residues: Sequence[int], p: int) -> bool:
    """Return True if ``residues`` occupy every class modulo ``p``.

    This is the exact low-wheel hard-zero obstruction.  It is used explicitly
    in CHL3 because the q=3 anomaly is precisely a failure mode where an
    absolute Möbius zero can be diluted by averaged approximations.
    """
    p = int(p)
    return len({int(r) % p for r in residues}) >= p


def lowwheel_is_inadmissible(residues: Sequence[int], low_primes: Sequence[int]) -> bool:
    """Return True if the tuple is inadmissible for at least one low prime."""
    return any(lowwheel_covers_all_classes(residues, int(q)) for q in low_primes)


def local_factor_from_residues(residues: Sequence[int], tuple_size: int, p: int) -> float:
    """
    Local Hardy--Littlewood factor for a tuple of fixed cardinality.

    tuple_size must be the number of tuple positions, not the number of
    distinct residues. This is essential for H5 with repeated low-wheel
    residues.
    """
    p = int(p)
    nu = len({int(r) % p for r in residues})
    return (1.0 - nu / p) / ((1.0 - 1.0 / p) ** tuple_size)


def lowwheel_singular_from_residues(
    residues: Sequence[int],
    tuple_size: int,
    low_primes: Sequence[int],
) -> float:
    """Product of local factors over low primes, preserving hard zeros."""
    if lowwheel_is_inadmissible(residues, low_primes):
        return 0.0
    prod = 1.0
    for p in low_primes:
        factor = local_factor_from_residues(residues, tuple_size, p)
        if factor <= 0.0:
            return 0.0
        prod *= factor
    return prod

def omega_second_order_lowwheel(
    g1: int,
    g2: int,
    log_x: float,
    low_primes: Sequence[int],
    singular_ratio_h4_h3,
) -> tuple[float, dict]:
    """
    Low-wheel second-order cluster correction.

    Uses full CHL2 marginal interior intensities p_u but approximates
    cross-dependence through low-prime local cumulants.
    """
    if not low_primes:
        # Empty low-wheel reproduces CHL2 first-order Poisson.
        omega1 = 0.0
        for u in even_interior_offsets(g2):
            omega1 += (1.0 / log_x) * singular_ratio_h4_h3(g1, g2, u)
        return omega1, {"low_primes": [], "kappa_sum": 0.0}

    M = 1
    for p in low_primes:
        M *= p

    inv_log = 1.0 / log_x
    offsets = even_interior_offsets(g2)

    # Aggregate p_u by low-wheel residue.
    S = {r: 0.0 for r in range(M)}
    Q = {r: 0.0 for r in range(M)}
    omega1 = 0.0

    for u in offsets:
        p_u = inv_log * singular_ratio_h4_h3(g1, g2, u)
        r = u % M
        S[r] += p_u
        Q[r] += p_u * p_u
        omega1 += p_u

    # Low-wheel residues of endpoints.
    h3_res = (0 % M, g1 % M, (g1 + g2) % M)
    s3 = lowwheel_singular_from_residues(h3_res, 3, low_primes)

    kappa_sum = 0.0
    active_pairs = 0

    residues = [r for r in range(M) if S[r] != 0.0]

    for i, r in enumerate(residues):
        for s in residues[i:]:
            if r != s:
                weight = S[r] * S[s]
            else:
                weight = 0.5 * (S[r] * S[r] - Q[r])

            if weight == 0.0:
                continue

            h4_r = (0 % M, g1 % M, (g1 + r) % M, (g1 + g2) % M)
            h4_s = (0 % M, g1 % M, (g1 + s) % M, (g1 + g2) % M)
            h5_rs = (0 % M, g1 % M, (g1 + r) % M, (g1 + s) % M, (g1 + g2) % M)

            s4_r = lowwheel_singular_from_residues(h4_r, 4, low_primes)
            s4_s = lowwheel_singular_from_residues(h4_s, 4, low_primes)
            if s4_r <= 0.0 or s4_s <= 0.0 or s3 <= 0.0:
                continue

            if lowwheel_is_inadmissible(h5_rs, low_primes):
                # Exact Möbius hard zero: S_low(H5)=0, hence coupling=0 and
                # kappa contribution equals -weight.  Do not skip this case.
                coupling = 0.0
            else:
                s5 = lowwheel_singular_from_residues(h5_rs, 5, low_primes)
                coupling = (s5 * s3) / (s4_r * s4_s)

            kappa_sum += weight * (coupling - 1.0)
            active_pairs += 1

    omega2 = omega1 - kappa_sum

    return omega2, {
        "low_primes": list(low_primes),
        "M_low": M,
        "omega1": omega1,
        "kappa_sum": kappa_sum,
        "active_residue_pairs": active_pairs,
    }

