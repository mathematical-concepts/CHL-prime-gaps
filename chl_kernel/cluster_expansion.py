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


def local_factor_from_residues(residues: Sequence[int], tuple_size: int, p: int) -> float:
    """
    Local Hardy--Littlewood factor for a tuple of fixed cardinality.

    tuple_size must be the number of tuple positions, not the number of
    distinct residues. This is essential for H5 with repeated low-wheel
    residues.
    """
    nu = len({r % p for r in residues})
    return (1.0 - nu / p) / ((1.0 - 1.0 / p) ** tuple_size)


def lowwheel_singular_from_residues(
    residues: Sequence[int],
    tuple_size: int,
    low_primes: Sequence[int],
) -> float:
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
    Strict low-wheel Möbius second-order correction.

    The base first-order intensity is the full-horizon CHL2 quantity

        Omega^(1) = sum_u S_Y(H4(u))/S_Y(H3)/log(x).

    The cross-cumulant is computed on the selected low wheel only.  This is
    intentionally not an ``early exit'' approximation: all unordered interior
    pairs u < v are included.  The computation remains fast because, on a fixed
    low wheel, the ratios S_low(H4)/S_low(H3) and S_low(H5)/S_low(H3) depend only
    on residues modulo M_low = prod(low_primes), so the pair sum is aggregated
    exactly by residue classes.

    If H5 is inadmissible modulo any selected low prime, S_low(H5)=0 exactly,
    giving the hard negative cumulant -p_u p_v.  This is the intended Möbius
    strict correction for compressed low-wheel systems such as q=3.
    """
    inv_log = 1.0 / max(float(log_x), 1.0)
    offsets = even_interior_offsets(g2)

    # Full-horizon first-order intensity and Bernoulli self term.
    omega1 = 0.0
    self_sum_full = 0.0
    for u in offsets:
        p_full = inv_log * float(singular_ratio_h4_h3(g1, g2, u))
        if not (p_full > 0.0):
            p_full = 0.0
        omega1 += p_full
        self_sum_full += p_full * p_full
    self_term = 0.5 * self_sum_full

    if not low_primes:
        return omega1, {
            "low_primes": [],
            "M_low": 1,
            "omega1": omega1,
            "kappa_sum": 0.0,
            "self_term": self_term,
            "omega_self": omega1 + self_term,
            "hard_zero_h5_pairs": 0.0,
            "hard_zero_h5_weight": 0.0,
            "hard_zero_h5_kappa_mass": 0.0,
            "kappa_negative_mass": 0.0,
            "kappa_positive_mass": 0.0,
            "mobius_low_kappa_method": 2.0,
        }

    M = 1
    for p in low_primes:
        M *= int(p)

    h3_res = (0 % M, int(g1) % M, (int(g1) + int(g2)) % M)
    s3_low = lowwheel_singular_from_residues(h3_res, 3, low_primes)
    if s3_low <= 0.0:
        # If the endpoint triple is already inadmissible on the low wheel, the
        # caller should normally have zeroed the CHL weight through log_R.  We
        # return a finite diagnostic rather than raising.
        return omega1, {
            "low_primes": list(low_primes),
            "M_low": M,
            "omega1": omega1,
            "kappa_sum": 0.0,
            "self_term": self_term,
            "omega_self": omega1 + self_term,
            "hard_zero_h5_pairs": 0.0,
            "hard_zero_h5_weight": 0.0,
            "hard_zero_h5_kappa_mass": 0.0,
            "kappa_negative_mass": 0.0,
            "kappa_positive_mass": 0.0,
            "mobius_low_kappa_method": 2.0,
        }

    # Aggregate p_u^low by residue.
    N = {r: 0 for r in range(M)}
    S_low = {r: 0.0 for r in range(M)}
    Q_low = {r: 0.0 for r in range(M)}
    skipped_zero_h4 = 0
    for u in offsets:
        r = int(u) % M
        N[r] += 1
        h4_res = (0 % M, int(g1) % M, (int(g1) + r) % M, (int(g1) + int(g2)) % M)
        s4_low = lowwheel_singular_from_residues(h4_res, 4, low_primes)
        if s4_low <= 0.0:
            skipped_zero_h4 += 1
            continue
        p_low = inv_log * (s4_low / s3_low)
        S_low[r] += p_low
        Q_low[r] += p_low * p_low

    kappa_sum = 0.0
    active_pairs = 0.0
    hard_zero_pairs = 0.0
    hard_zero_weight = 0.0
    hard_zero_kappa_mass = 0.0
    nonzero_h5_pairs = 0.0
    kappa_negative_mass = 0.0
    kappa_positive_mass = 0.0
    inv_log2 = inv_log * inv_log
    residues = [r for r in range(M) if N[r] > 0]

    for i, r in enumerate(residues):
        for s in residues[i:]:
            if r != s:
                pair_count = float(N[r] * N[s])
                product_mass = S_low[r] * S_low[s]
            else:
                pair_count = float(N[r] * (N[r] - 1) // 2)
                product_mass = 0.5 * (S_low[r] * S_low[r] - Q_low[r])
            if pair_count <= 0.0:
                continue
            h5_res = (
                0 % M,
                int(g1) % M,
                (int(g1) + r) % M,
                (int(g1) + s) % M,
                (int(g1) + int(g2)) % M,
            )
            s5_low = lowwheel_singular_from_residues(h5_res, 5, low_primes)
            if s5_low <= 0.0:
                p_uv_low = 0.0
                hard_zero_pairs += pair_count
                hard_zero_weight += product_mass
            else:
                p_uv_low = inv_log2 * (s5_low / s3_low)
                nonzero_h5_pairs += pair_count
            kappa = pair_count * p_uv_low - product_mass
            if s5_low <= 0.0 and product_mass > 0.0:
                hard_zero_kappa_mass += kappa
            if kappa < 0.0:
                kappa_negative_mass += kappa
            elif kappa > 0.0:
                kappa_positive_mass += kappa
            kappa_sum += kappa
            active_pairs += pair_count

    omega2 = omega1 - kappa_sum
    omega_self = omega1 + self_term - kappa_sum
    return omega2, {
        "low_primes": list(low_primes),
        "M_low": M,
        "omega1": omega1,
        "kappa_sum": kappa_sum,
        "self_term": self_term,
        "omega_self": omega_self,
        "active_pairs": active_pairs,
        "nonzero_h5_pairs": nonzero_h5_pairs,
        "hard_zero_h5_pairs": hard_zero_pairs,
        "hard_zero_h5_weight": hard_zero_weight,
        "hard_zero_h5_kappa_mass": hard_zero_kappa_mass,
        "kappa_negative_mass": kappa_negative_mass,
        "kappa_positive_mass": kappa_positive_mass,
        "skipped_zero_h4_pairs": float(skipped_zero_h4),
        "mobius_low_kappa_method": 2.0,
    }
