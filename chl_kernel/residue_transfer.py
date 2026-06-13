"""Direct modular-transfer utilities for consecutive prime residues.

This module is intentionally independent from CSV files and command-line code.
It provides small linear-algebra helpers for the CHL4 residual-transfer programme:

* empirical and model transfer matrices over reduced residue classes;
* row-centered log residuals between empirical and CHL2-induced transitions;
* the q=3 diagonal/off-diagonal scalar;
* a Dirichlet-character Fourier basis for prime moduli.

The objects here do not alter the CHL2 gap-survival kernel.  They are a new
layer for studying direct state-to-state residue transfer:

    T_q(b,a) = P(p_n = a mod q | p_{n-1} = b mod q).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TransferMatrix:
    """A row-stochastic transfer matrix on reduced residue classes.

    Parameters
    ----------
    q:
        Modulus.
    residues:
        Reduced residue classes used for rows and columns, in order.
    probabilities:
        Square array ``P[b_index, a_index]``.
    row_counts:
        Optional empirical row counts.  Used for weighted summaries.
    counts:
        Optional empirical count matrix.
    label:
        Human-readable label, e.g. ``"empirical"`` or ``"CHL2"``.
    """

    q: int
    residues: tuple[int, ...]
    probabilities: np.ndarray
    row_counts: np.ndarray | None = None
    counts: np.ndarray | None = None
    label: str = "matrix"

    def __post_init__(self) -> None:
        p = np.asarray(self.probabilities, dtype=float)
        object.__setattr__(self, "probabilities", p)
        if p.shape != (len(self.residues), len(self.residues)):
            raise ValueError("probabilities must be square with shape len(residues)^2")
        if self.row_counts is not None:
            object.__setattr__(self, "row_counts", np.asarray(self.row_counts, dtype=float))
        if self.counts is not None:
            object.__setattr__(self, "counts", np.asarray(self.counts, dtype=float))


def reduced_residues(q: int) -> list[int]:
    """Return the reduced residue classes modulo ``q`` in increasing order."""
    q = int(q)
    return [a for a in range(1, q) if math.gcd(a, q) == 1]


def is_prime_int(n: int) -> bool:
    """Small deterministic primality check for helper validation."""
    n = int(n)
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    r = int(math.isqrt(n))
    for d in range(3, r + 1, 2):
        if n % d == 0:
            return False
    return True


def primitive_root_prime_modulus(q: int) -> int:
    """Return a primitive root modulo a prime ``q``.

    The first CHL4 implementation targets prime moduli such as 3, 5, 7, 11,
    and 13.  For composite moduli, a more general finite-abelian-group Fourier
    basis would be needed.
    """
    q = int(q)
    if not is_prime_int(q):
        raise ValueError("primitive_root_prime_modulus currently requires prime q")
    phi = q - 1
    # Factor phi.
    factors: set[int] = set()
    n = phi
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.add(n)
    for g in range(2, q):
        if all(pow(g, phi // p, q) != 1 for p in factors):
            return g
    raise RuntimeError(f"no primitive root found modulo {q}")


def dirichlet_character_table_prime_modulus(q: int) -> tuple[list[int], np.ndarray]:
    """Return residues and a character table for a prime modulus.

    Returns
    -------
    residues:
        Reduced residue classes in increasing order.
    table:
        Complex array of shape ``(phi(q), phi(q))`` where ``table[i,k]`` is the
        value of the k-th Dirichlet character at ``residues[i]``.  Character
        index 0 is the principal character.
    """
    q = int(q)
    residues = reduced_residues(q)
    phi = len(residues)
    if phi != q - 1 or not is_prime_int(q):
        raise ValueError("only prime q is supported by this character table")
    g = primitive_root_prime_modulus(q)
    exponent_of: dict[int, int] = {}
    x = 1
    for m in range(phi):
        exponent_of[x] = m
        x = (x * g) % q
    table = np.zeros((phi, phi), dtype=complex)
    for i, a in enumerate(residues):
        m = exponent_of[a % q]
        for k in range(phi):
            table[i, k] = np.exp(2j * np.pi * k * m / phi)
    return residues, table


def normalize_rows(counts: np.ndarray, smoothing: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Normalize a count matrix row-wise.

    Returns ``(probabilities, row_counts)``.  Rows with zero mass become uniform
    after smoothing or zero if no smoothing is supplied.
    """
    c = np.asarray(counts, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("counts must be a square matrix")
    if smoothing:
        c = c + float(smoothing)
    row_counts = c.sum(axis=1)
    probs = np.zeros_like(c, dtype=float)
    nz = row_counts > 0
    probs[nz] = c[nz] / row_counts[nz, None]
    return probs, row_counts


def transfer_from_counts(q: int, counts: np.ndarray, residues: Sequence[int] | None = None, label: str = "empirical") -> TransferMatrix:
    """Build a :class:`TransferMatrix` from a count matrix."""
    residues_t = tuple(int(x) for x in (residues or reduced_residues(q)))
    probs, row_counts = normalize_rows(np.asarray(counts, dtype=float))
    return TransferMatrix(q=int(q), residues=residues_t, probabilities=probs, row_counts=row_counts, counts=counts, label=label)


def build_count_matrix_from_pairs(q: int, from_residues: np.ndarray, to_residues: np.ndarray) -> TransferMatrix:
    """Build an empirical transfer matrix from arrays of residue pairs."""
    residues = reduced_residues(q)
    idx = {r: i for i, r in enumerate(residues)}
    counts = np.zeros((len(residues), len(residues)), dtype=float)
    for b, a in zip(from_residues, to_residues):
        bi = idx.get(int(b) % q)
        ai = idx.get(int(a) % q)
        if bi is not None and ai is not None:
            counts[bi, ai] += 1.0
    return transfer_from_counts(q=q, counts=counts, residues=residues, label="empirical")


def row_centered_log_residual(empirical: TransferMatrix, model: TransferMatrix, eps: float = 1e-12) -> np.ndarray:
    """Compute row-centered log-ratio residual ``log(emp/model)``.

    The row-centering removes row-normalization offsets and leaves relative
    transfer preference inside each previous-residue state.
    """
    if empirical.q != model.q or empirical.residues != model.residues:
        raise ValueError("empirical and model matrices must use the same q and residues")
    r = np.log(empirical.probabilities + eps) - np.log(model.probabilities + eps)
    return r - r.mean(axis=1, keepdims=True)


def diagonal_probability(matrix: TransferMatrix, weighted: bool = True) -> float:
    """Return average diagonal probability.

    If ``weighted`` and row counts are available, rows are weighted by their
    empirical mass; otherwise rows are averaged uniformly.
    """
    diag = np.diag(matrix.probabilities)
    if weighted and matrix.row_counts is not None and matrix.row_counts.sum() > 0:
        w = matrix.row_counts / matrix.row_counts.sum()
        return float(np.dot(w, diag))
    return float(diag.mean())


def diagonal_log_odds_q3(matrix: TransferMatrix, eps: float = 1e-12) -> float:
    """Return ``D3 = log(T11*T22/(T12*T21))`` for q=3."""
    if int(matrix.q) != 3:
        raise ValueError("diagonal_log_odds_q3 requires q=3")
    residues = list(matrix.residues)
    if residues != [1, 2]:
        raise ValueError("q=3 matrix must use residues (1,2)")
    t = matrix.probabilities
    return float(np.log((t[0, 0] + eps) * (t[1, 1] + eps) / ((t[0, 1] + eps) * (t[1, 0] + eps))))


def character_spectrum(residual: np.ndarray, q: int, residues: Sequence[int] | None = None) -> list[dict[str, float | int]]:
    """Project a row-centered residual matrix onto Dirichlet-character modes.

    The formula used is the finite Fourier projection over the reduced residue group.

    Only prime moduli are currently supported.
    """
    residues_table, chars = dirichlet_character_table_prime_modulus(q)
    if residues is not None and list(residues) != residues_table:
        raise ValueError("residue order does not match the prime-modulus character table")
    R = np.asarray(residual, dtype=float)
    phi = len(residues_table)
    if R.shape != (phi, phi):
        raise ValueError("residual shape must be phi(q) x phi(q)")
    out: list[dict[str, float | int]] = []
    coeffs: list[complex] = []
    for chi_idx in range(phi):
        for psi_idx in range(phi):
            acc = 0j
            for bi in range(phi):
                for ai in range(phi):
                    acc += R[bi, ai] * np.conjugate(chars[ai, chi_idx]) * chars[bi, psi_idx]
            acc /= phi * phi
            coeffs.append(acc)
            out.append({
                "q": int(q),
                "chi_index": int(chi_idx),
                "psi_index": int(psi_idx),
                "coefficient_real": float(acc.real),
                "coefficient_imag": float(acc.imag),
                "energy": float(abs(acc) ** 2),
            })
    total = sum(float(abs(c) ** 2) for c in coeffs)
    for row in out:
        row["energy_fraction"] = 0.0 if total <= 0 else float(row["energy"] / total)
    return out


def effective_rank_from_energy(energies: Iterable[float], eps: float = 1e-15) -> float:
    """Entropy effective rank from nonnegative mode energies."""
    e = np.asarray(list(energies), dtype=float)
    s = e.sum()
    if s <= 0:
        return 0.0
    p = e / s
    p = p[p > eps]
    return float(np.exp(-np.sum(p * np.log(p))))
