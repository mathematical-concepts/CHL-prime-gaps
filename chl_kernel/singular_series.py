"""Truncated Hardy--Littlewood singular series utilities.

The central object is the truncated k-tuple singular series

    S_Y(H) = prod_{q <= Y} (1 - nu_q(H)/q) / (1 - 1/q)^|H|,

where ``nu_q(H)`` is the number of residue classes occupied by the offsets in
``H`` modulo q.  The code works in log space for numerical stability.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence, Tuple

from .primes import primes_upto

OffsetTuple = Tuple[int, ...]


def canonical_offsets(offsets: Iterable[int]) -> OffsetTuple:
    """Return a sorted tuple of unique integer offsets."""
    return tuple(sorted({int(x) for x in offsets}))


def nu_mod_q(offsets: Sequence[int], q: int) -> int:
    """Number of distinct residues occupied by ``offsets`` modulo ``q``."""
    return len({int(o) % int(q) for o in offsets})


def is_admissible(offsets: Iterable[int], Y: int | None = None) -> bool:
    """Return whether a tuple is admissible up to the horizon ``Y``.

    A tuple H is admissible if ``nu_q(H) < q`` for every prime q.  With finite
    ``Y`` this checks the condition for primes ``q <= Y``.
    """
    offs = canonical_offsets(offsets)
    if len(offs) == 0:
        return True
    if Y is None:
        # It is enough to check primes up to max span + 1 for a finite set of
        # integer offsets in many practical situations, but this function is
        # used in the truncated setting.  We keep the explicit Y requirement for
        # mathematical clarity.
        raise ValueError("Y must be provided for a finite admissibility check")
    for q in primes_upto(int(Y)):
        if nu_mod_q(offs, q) >= q:
            return False
    return True


@dataclass
class SingularSeriesCache:
    """Cached evaluator for truncated singular-series products.

    Parameters
    ----------
    Y:
        Truncation horizon.  Only primes ``q <= Y`` enter the product.
    max_cache_size:
        Maximum number of log-values retained.  ``0`` means unlimited.
    """
    Y: int = 47
    max_cache_size: int = 1_000_000
    primes: tuple[int, ...] = field(init=False)
    _cache: dict[OffsetTuple, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.Y = int(self.Y)
        self.primes = primes_upto(self.Y)

    def log_singular(self, offsets: Iterable[int]) -> float:
        """Return ``log S_Y(offsets)``.

        If the tuple is inadmissible under the truncation, ``-inf`` is returned.
        """
        key = canonical_offsets(offsets)
        val = self._cache.get(key)
        if val is not None:
            return val
        k = len(key)
        acc = 0.0
        for q in self.primes:
            qf = float(q)
            nu = nu_mod_q(key, q)
            local = (1.0 - nu / qf) / ((1.0 - 1.0 / qf) ** k)
            if local <= 0.0:
                acc = -math.inf
                break
            acc += math.log(local)
        if self.max_cache_size <= 0 or len(self._cache) < self.max_cache_size:
            self._cache[key] = float(acc)
        else:
            # Clearing the cache only affects speed, never values.
            self._cache.clear()
            self._cache[key] = float(acc)
        return float(acc)

    def singular(self, offsets: Iterable[int]) -> float:
        """Return ``S_Y(offsets)`` in ordinary scale."""
        z = self.log_singular(offsets)
        return 0.0 if not math.isfinite(z) else math.exp(z)

    def log_ratio(self, numerator_offsets: Iterable[int], denominator_offsets: Iterable[int]) -> float:
        """Return the log of a singular-series ratio."""
        a = self.log_singular(numerator_offsets)
        b = self.log_singular(denominator_offsets)
        if not math.isfinite(a):
            return -math.inf
        if not math.isfinite(b):
            return math.inf
        return float(a - b)
