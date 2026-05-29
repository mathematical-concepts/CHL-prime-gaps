"""Parameter-free CHL1/CHL2 kernels and classical baselines.

This module contains no CSV reading and no command-line code.  It is the frozen
mathematical engine of the repository.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .singular_series import SingularSeriesCache
from .cluster_expansion import omega_path_bernoulli, omega_second_order_full, omega_second_order_lowwheel


def even_candidates(gmax: int, start: int = 2) -> range:
    """Even candidate gaps ``start, start+2, ..., gmax``."""
    start = 2 if int(start) < 2 else int(start)
    if start % 2:
        start += 1
    return range(start, int(gmax) + 1, 2)


def survives_actual_wheel(p_current: int, g: int, primes: Sequence[int]) -> bool:
    """Return whether the concrete candidate ``p_current + g`` survives q <= Y.

    This is useful for search-oracle applications.  The statistical CHL audits
    operate on aggregated gap pairs and use tuple admissibility rather than a
    concrete current residue.
    """
    n = int(p_current) + int(g)
    for q in primes:
        if n != q and n % q == 0:
            return False
    return True


@dataclass
class CHLKernel:
    """Conditional Hardy--Littlewood Markov kernel with CHL2 exclusion.

    Parameters
    ----------
    Y:
        Truncation horizon for singular-series products.  This is not a sieve
        level in the classical asymptotic sense; it is an auditable finite
        horizon used by the empirical kernel.
    log_x:
        Local value of ``log x`` used in the first-order Poisson no-interior
        intensity.
    max_cache_size:
        Singular-series cache size.  ``0`` means unlimited.
    """
    Y: int = 47
    log_x: float = 25.328436
    max_cache_size: int = 1_000_000
    series: SingularSeriesCache = field(init=False)

    def __post_init__(self) -> None:
        self.log_x = max(float(self.log_x), 1.0)
        self.series = SingularSeriesCache(self.Y, self.max_cache_size)

    # --- CHL1: conditional singular-series ratio -------------------------
    def log_R(self, g1: int, g2: int) -> float:
        """Return ``log R_Y(g2|g1)`` for CHL1.

        R_Y(g2|g1) = S_Y({0,g1,g1+g2}) / S_Y({0,g1}).
        """
        g1, g2 = int(g1), int(g2)
        return self.series.log_ratio((0, g1, g1 + g2), (0, g1))

    # --- CHL2: no-interior-prime survival correction ---------------------
    def omega_gap(self, g: int) -> float:
        """Endpoint-only first-order no-interior intensity.

        Omega_Y(g;x) = sum_{2 <= u <= g-2, u even}
                       S_Y({0,u,g}) / S_Y({0,g}) / log(x).
        """
        g = int(g)
        if g <= 2:
            return 0.0
        log_s2 = self.series.log_singular((0, g))
        if not math.isfinite(log_s2):
            return math.inf
        omega = 0.0
        for u in range(2, g, 2):
            log_s3 = self.series.log_singular((0, u, g))
            if math.isfinite(log_s3):
                omega += math.exp(log_s3 - log_s2) / self.log_x
        return float(omega)

    def omega_path(self, g1: int, g2: int) -> float:
        """Path-sensitive first-order no-interior intensity.

        Omega_Y^path(g1,g2;x) = sum_u S_Y({0,g1,g1+u,g1+g2})
                                / S_Y({0,g1,g1+g2}) / log(x).
        """
        g1, g2 = int(g1), int(g2)
        if g2 <= 2:
            return 0.0
        log_s3 = self.series.log_singular((0, g1, g1 + g2))
        if not math.isfinite(log_s3):
            return math.inf
        omega = 0.0
        for u in range(2, g2, 2):
            log_s4 = self.series.log_singular((0, g1, g1 + u, g1 + g2))
            if math.isfinite(log_s4):
                omega += math.exp(log_s4 - log_s3) / self.log_x
        return float(omega)

    def singular_ratio_h4_h3(self, g1: int, g2: int, u: int) -> float:
        """Return S_Y(H4(u)) / S_Y(H3) for the path interior candidate."""
        g1, g2, u = int(g1), int(g2), int(u)
        z = self.series.log_ratio((0, g1, g1 + u, g1 + g2), (0, g1, g1 + g2))
        return 0.0 if not math.isfinite(z) else float(math.exp(z))

    def singular_ratio_h5_h3(self, g1: int, g2: int, u: int, v: int) -> float:
        """Return S_Y(H5(u,v)) / S_Y(H3) for two interior candidates."""
        g1, g2, u, v = int(g1), int(g2), int(u), int(v)
        z = self.series.log_ratio((0, g1, g1 + u, g1 + v, g1 + g2), (0, g1, g1 + g2))
        return 0.0 if not math.isfinite(z) else float(math.exp(z))

    def omega_path_bernoulli(self, g1: int, g2: int) -> tuple[float, dict]:
        """Bernoulli self-correction of the CHL2 path exclusion factor."""
        return omega_path_bernoulli(g1, g2, self.log_x, self.singular_ratio_h4_h3)

    def omega_second_order_full(self, g1: int, g2: int) -> tuple[float, dict]:
        """Full second-order H5/H3 cluster correction; diagnostic only."""
        return omega_second_order_full(g1, g2, self.log_x, self.singular_ratio_h4_h3, self.singular_ratio_h5_h3)

    def omega_second_order_lowwheel(self, g1: int, g2: int, low_primes: Sequence[int]) -> tuple[float, dict]:
        """Strict low-wheel Möbius second-order correction."""
        return omega_second_order_lowwheel(g1, g2, self.log_x, low_primes, self.singular_ratio_h4_h3)

    def log_weight_chl2_bernoulli_path(self, g1: int, g2: int, eta: float = 0.0) -> float:
        """Unnormalized CHL2 path log-weight with Bernoulli self correction."""
        z = self.log_R(g1, g2)
        if not math.isfinite(z):
            return -math.inf
        om, _ = self.omega_path_bernoulli(g1, g2)
        return -math.inf if not math.isfinite(om) else float(z - om + eta * int(g2))

    def log_weight_chl3_lowwheel(self, g1: int, g2: int, low_primes: Sequence[int], eta: float = 0.0) -> float:
        """Unnormalized strict low-wheel CHL3 log-weight without self term."""
        z = self.log_R(g1, g2)
        if not math.isfinite(z):
            return -math.inf
        om, _ = self.omega_second_order_lowwheel(g1, g2, low_primes)
        return -math.inf if not math.isfinite(om) else float(z - om + eta * int(g2))

    def log_weight_chl3_self_lowwheel(self, g1: int, g2: int, low_primes: Sequence[int], eta: float = 0.0) -> float:
        """Unnormalized strict low-wheel CHL3 log-weight with Bernoulli self term."""
        z = self.log_R(g1, g2)
        if not math.isfinite(z):
            return -math.inf
        _, info = self.omega_second_order_lowwheel(g1, g2, low_primes)
        om = float(info.get("omega_self", float("nan")))
        return -math.inf if not math.isfinite(om) else float(z - om + eta * int(g2))

    def log_weight_chl1(self, g1: int, g2: int, eta: float = 0.0) -> float:
        """Unnormalized CHL1 log-weight for candidate ``g2`` after ``g1``."""
        z = self.log_R(g1, g2)
        return -math.inf if not math.isfinite(z) else float(z + eta * int(g2))

    def log_weight_chl2_path(self, g1: int, g2: int, eta: float = 0.0) -> float:
        """Unnormalized CHL2 path-exclusion log-weight."""
        z = self.log_R(g1, g2)
        if not math.isfinite(z):
            return -math.inf
        om = self.omega_path(g1, g2)
        return -math.inf if not math.isfinite(om) else float(z - om + eta * int(g2))

    def log_weight_chl2_gap(self, g1: int, g2: int, eta: float = 0.0) -> float:
        """Unnormalized CHL2 endpoint-gap-exclusion log-weight."""
        z = self.log_R(g1, g2)
        if not math.isfinite(z):
            return -math.inf
        om = self.omega_gap(g2)
        return -math.inf if not math.isfinite(om) else float(z - om + eta * int(g2))

    def cramer_granville_log_weight(self, g: int, use_gap_exclusion: bool = False) -> float:
        """Order-zero Cramer--Granville log-weight for candidate gap ``g``.

        P(g) ∝ S_Y({0,g}) exp(-g/log x), optionally multiplied by the endpoint
        no-interior survival factor ``exp(-Omega_Y(g;x))``.
        """
        g = int(g)
        z = self.series.log_singular((0, g))
        if not math.isfinite(z):
            return -math.inf
        val = z - g / self.log_x
        if use_gap_exclusion:
            om = self.omega_gap(g)
            if not math.isfinite(om):
                return -math.inf
            val -= om
        return float(val)

    def hit_cost(self, g1: int, g2: int) -> float:
        """Heuristic search cost C_Y(g2|g1) = -log R_Y + Omega_Y^path."""
        r = self.log_R(g1, g2)
        if not math.isfinite(r):
            return math.inf
        om = self.omega_path(g1, g2)
        return math.inf if not math.isfinite(om) else float(-r + om)

    def normalized_distribution(self, g1: int, candidates: Iterable[int], model: str = "chl2_path", eta: float = 0.0) -> dict[int, float]:
        """Return a normalized distribution over candidate gaps.

        Parameters
        ----------
        model:
            One of ``'chl2_path'``, ``'chl2_gap'``, ``'chl1'``.
        """
        weights: dict[int, float] = {}
        max_log = -math.inf
        for h in candidates:
            if model == "chl2_path":
                z = self.log_weight_chl2_path(g1, h, eta)
            elif model == "chl2_gap":
                z = self.log_weight_chl2_gap(g1, h, eta)
            elif model == "chl1":
                z = self.log_weight_chl1(g1, h, eta)
            else:
                raise ValueError(f"unknown model: {model}")
            weights[int(h)] = z
            if z > max_log:
                max_log = z
        if not math.isfinite(max_log):
            return {int(h): 0.0 for h in weights}
        norm = sum(math.exp(z - max_log) for z in weights.values() if math.isfinite(z))
        if norm <= 0:
            return {int(h): 0.0 for h in weights}
        return {h: (math.exp(z - max_log) / norm if math.isfinite(z) else 0.0) for h, z in weights.items()}

def chl3_log_weight_lowwheel(
    kernel,
    g1: int,
    g2: int,
    low_primes: Sequence[int],
    eta: float = 0.0,
) -> tuple[float, dict]:
    """
    Log unnormalized CHL3 low-wheel second-order weight.
    """
    log_R = kernel.log_R(g1, g2)
    omega2, info = omega_second_order_lowwheel(
        g1=g1,
        g2=g2,
        log_x=kernel.log_x,
        low_primes=low_primes,
        singular_ratio_h4_h3=kernel.singular_ratio_h4_h3,
    )
    return log_R - omega2 + eta * g2, info