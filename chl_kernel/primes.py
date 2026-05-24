"""Prime utilities used by the CHL kernels.

The functions in this file are intentionally small and dependency-free.  They
are suitable for use in notebooks, independent projects, and the repository's
CLI scripts.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Iterable, List


@lru_cache(maxsize=None)
def primes_upto(n: int) -> tuple[int, ...]:
    """Return all primes ``<= n`` as a cached tuple.

    Parameters
    ----------
    n:
        Inclusive upper bound.
    """
    n = int(n)
    if n < 2:
        return tuple()
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[0:2] = b"\x00\x00"
    r = math.isqrt(n)
    for p in range(2, r + 1):
        if sieve[p]:
            start = p * p
            sieve[start:n + 1:p] = b"\x00" * (((n - start) // p) + 1)
    return tuple(i for i in range(n + 1) if sieve[i])


def is_probable_prime_trial(n: int) -> bool:
    """Deterministic trial-division primality check for small/medium integers.

    This is not meant for record-scale primality testing.  It is used only in
    quick examples and smoke tests.  The data generator uses a segmented sieve.
    """
    n = int(n)
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    r = math.isqrt(n)
    f = 5
    while f <= r:
        if n % f == 0 or n % (f + 2) == 0:
            return False
        f += 6
    return True


def reduced_residues(q: int) -> tuple[int, ...]:
    """Return residues modulo ``q`` coprime to ``q``."""
    return tuple(a for a in range(q) if math.gcd(a, q) == 1)
