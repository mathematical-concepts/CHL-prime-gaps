#!/usr/bin/env python3
"""CHL4 direct modular-transfer residual audit.

This script begins the CHL4 programme without modifying the CHL2 gap-survival
kernel. It measures the direct residue-transfer residual left by CHL2:

    T_q(b,a) = P(p_n = a mod q | p_{n-1} = b mod q),

    R_q(b,a) = row-centered log(T_emp(b,a) / T_CHL2(b,a)).

The first milestone is the q=3 diagonal scalar

    D3 = log(T(1,1) T(2,2) / (T(1,2) T(2,1))).

A negative D3 means diagonal repulsion; a positive D3 means diagonal persistence.

Input modes
-----------
1. from-os-csv: read a CHL2 OS transition CSV already produced by
   chl2_consecutive_exclusion_audit.py.
2. from-prime-csv: stream a chronological prime CSV to compute empirical
   transfer matrices by block, while using a CHL2 transition CSV as the model
   reference. This mode can align blocks to the real ``parent_wide_B*.csv.gz``
   block counts, avoiding artificial B11 tail blocks.

No Oliver--Soundararajan term is fitted here. This is the residual-measurement
stage that decides whether CHL4 has a stable object to explain.
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
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# Ensure repository root imports work when called as a script.
ROOT_HINT = Path(__file__).resolve().parents[1]
if str(ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(ROOT_HINT))

from chl_kernel.residue_transfer import (
    TransferMatrix,
    character_spectrum,
    diagonal_log_odds_q3,
    diagonal_probability,
    effective_rank_from_energy,
    reduced_residues,
    row_centered_log_residual,
    transfer_from_counts,
)
try:
    from chl_kernel.telemetry import telemetry_start, write_telemetry
except Exception:  # pragma: no cover - compatibility with older repository checkouts
    def telemetry_start():
        return {"start_time": time.time()}

    def write_telemetry(path, tele, **kwargs):
        start = float(tele.get("start_time", time.time()))
        payload = dict(kwargs)
        payload["elapsed_seconds"] = time.time() - start
        payload["argv"] = sys.argv
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


@dataclass
class LoadedMatrixSet:
    q: int
    block: str
    model_name: str
    empirical: TransferMatrix
    model: TransferMatrix


@dataclass
class BlockPlan:
    labels: list[str]
    counts: list[int]
    mode: str
    source_files: list[str]
    total_expected_transitions: int


def parse_int_list(s: str) -> list[int]:
    """Parse comma-separated integers."""
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_block_spec(s: str | None) -> list[int]:
    """Parse block specs like '1-10,12'."""
    if not s:
        return []
    out: list[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(dict.fromkeys(out))


def first_existing_column(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    """Return the first available column from a list of candidates."""
    available = set(columns)
    for c in candidates:
        if c in available:
            return c
    return None


def normalize_matrix_rows(mat: np.ndarray) -> np.ndarray:
    """Return row-normalized copy of a square matrix."""
    m = np.asarray(mat, dtype=float).copy()
    s = m.sum(axis=1)
    nz = s > 0
    m[nz] /= s[nz, None]
    return m


def safe_to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    """Return markdown table when tabulate exists, otherwise plain text.

    This keeps the script runnable in minimal environments where pandas'
    optional ``tabulate`` dependency is not installed.
    """
    try:
        return df.to_markdown(index=index)
    except Exception:
        return "```\n" + df.to_string(index=index) + "\n```"


def read_json(path: str | Path | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def resolve_base_dir(config: dict, config_path: str | Path | None, root: str | Path) -> Path:
    """Resolve the dataset base directory.

    Preference order:
    - root/input_dir from config, if present;
    - config file parent directory;
    - root.
    """
    root_p = Path(root)
    input_dir = config.get("input_dir") or config.get("dataset_dir") or config.get("data_dir")
    if input_dir:
        cand = Path(input_dir)
        if not cand.is_absolute():
            cand = root_p / cand
        return cand
    if config_path:
        return Path(config_path).resolve().parent
    return root_p


def block_file_candidates(base_dir: Path, config: dict, block: int) -> list[Path]:
    """Return likely parent-wide file paths for a given block."""
    blocks_dir = config.get("blocks_dir", "blocks")
    block_glob = config.get("block_glob") or config.get("block_pattern") or "parent_wide_B{block:02d}.csv.gz"
    candidates: list[Path] = []
    # Python format pattern.
    try:
        candidates.append(base_dir / blocks_dir / block_glob.format(block=block))
    except Exception:
        pass
    # Common formats.
    candidates.extend([
        base_dir / blocks_dir / f"parent_wide_B{block:02d}.csv.gz",
        base_dir / blocks_dir / f"v46t12_ds_pilot_parent_wide_B{block:02d}.csv.gz",
        base_dir / f"parent_wide_B{block:02d}.csv.gz",
        base_dir / f"v46t12_ds_pilot_parent_wide_B{block:02d}.csv.gz",
    ])
    # Remove duplicates while preserving order.
    seen: set[str] = set()
    out: list[Path] = []
    for c in candidates:
        s = str(c)
        if s not in seen:
            out.append(c)
            seen.add(s)
    return out


def find_block_file(base_dir: Path, config: dict, block: int) -> Path:
    for c in block_file_candidates(base_dir, config, block):
        if c.exists():
            return c
    tried = "\n  ".join(str(x) for x in block_file_candidates(base_dir, config, block))
    raise FileNotFoundError(f"Could not resolve parent-wide file for block {block}. Tried:\n  {tried}")


def read_parent_wide_event_count(path: Path, count_col: str = "H") -> int:
    """Read parent-wide block event count.

    The usual count column is ``H``. If it is absent, fall back to row count.
    """
    header = pd.read_csv(path, nrows=0)
    if count_col in header.columns:
        total = 0
        for chunk in pd.read_csv(path, usecols=[count_col], chunksize=1_000_000):
            total += int(pd.to_numeric(chunk[count_col], errors="coerce").fillna(0).sum())
        return int(total)
    return int(sum(len(c) for c in pd.read_csv(path, chunksize=1_000_000)))


def build_parent_wide_block_plan(
    *,
    config: dict,
    config_path: str | Path | None,
    root: str | Path,
    blocks: Sequence[int],
    count_col: str = "H",
) -> BlockPlan:
    """Build block boundaries from real parent-wide block files."""
    base = resolve_base_dir(config, config_path, root)
    labels: list[str] = []
    counts: list[int] = []
    files: list[str] = []
    for b in blocks:
        path = find_block_file(base, config, int(b))
        n = read_parent_wide_event_count(path, count_col=count_col)
        labels.append(f"B{int(b):02d}")
        counts.append(int(n))
        files.append(str(path))
    return BlockPlan(labels=labels, counts=counts, mode="parent-wide", source_files=files, total_expected_transitions=int(sum(counts)))


def build_fixed_block_plan(block_size: int, max_blocks: int | None = None) -> BlockPlan | None:
    if not block_size or block_size <= 0:
        return None
    n = int(max_blocks or 0)
    if n <= 0:
        # Labels are generated on the fly in fixed mode.
        return BlockPlan(labels=[], counts=[], mode="fixed", source_files=[], total_expected_transitions=0)
    labels = [f"B{i:02d}" for i in range(1, n + 1)]
    counts = [int(block_size)] * n
    return BlockPlan(labels=labels, counts=counts, mode="fixed", source_files=[], total_expected_transitions=int(sum(counts)))


def read_prime_column(path: str | Path, chunksize: int = 1_000_000):
    """Yield numpy arrays of primes from a CSV/CSV.GZ file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    reader = pd.read_csv(path, chunksize=int(chunksize))
    prime_col: str | None = None
    for chunk in reader:
        if prime_col is None:
            prime_col = first_existing_column(chunk.columns, ["prime", "p", "p_n", "current_prime"])
            if prime_col is None:
                if len(chunk.columns) == 1:
                    prime_col = chunk.columns[0]
                else:
                    raise ValueError(f"cannot infer prime column from {list(chunk.columns)}")
        arr = chunk[prime_col].to_numpy(dtype=np.int64)
        if len(arr):
            yield arr


def build_model_matrices_from_os_csv(os_csv: str | Path, mods: Sequence[int], model: str | None = None) -> dict[int, TransferMatrix]:
    """Load CHL2 model transfer matrices from an OS transition CSV."""
    df = pd.read_csv(os_csv)
    q_col = first_existing_column(df.columns, ["q", "modulus"])
    if q_col is None:
        raise ValueError("OS CSV must contain a q/modulus column")
    model_col = first_existing_column(df.columns, ["model", "model_name"])
    from_col = first_existing_column(df.columns, ["from_residue_b_prev_prime", "from_residue", "b"])
    to_col = first_existing_column(df.columns, ["to_residue_a_current_prime", "to_residue", "a"])
    prob_col = first_existing_column(df.columns, ["model_prob", "model_probability", "probability_model"])
    count_col = first_existing_column(df.columns, ["row_count", "from_row_count"])
    if from_col is None or to_col is None or prob_col is None:
        raise ValueError("OS CSV must contain from/to residue columns and a model probability column")
    if model_col and model:
        df = df[df[model_col].astype(str) == str(model)]
    elif model_col:
        choices = list(df[model_col].astype(str).unique())
        preferred = "CHL2_path_excl_cond_eta"
        selected = preferred if preferred in choices else choices[0]
        df = df[df[model_col].astype(str) == selected]
        model = selected
    else:
        model = model or "model"

    out: dict[int, TransferMatrix] = {}
    for q in mods:
        sub = df[df[q_col].astype(int) == int(q)]
        residues = reduced_residues(q)
        idx = {r: i for i, r in enumerate(residues)}
        mat = np.zeros((len(residues), len(residues)), dtype=float)
        row_counts = np.zeros(len(residues), dtype=float)
        for _, row in sub.iterrows():
            bi = idx.get(int(row[from_col]) % int(q))
            ai = idx.get(int(row[to_col]) % int(q))
            if bi is None or ai is None:
                continue
            mat[bi, ai] = float(row[prob_col])
            if count_col is not None:
                row_counts[bi] = max(row_counts[bi], float(row[count_col]))
        rs = mat.sum(axis=1)
        nz = rs > 0
        mat[nz] = mat[nz] / rs[nz, None]
        out[int(q)] = TransferMatrix(q=int(q), residues=tuple(residues), probabilities=mat, row_counts=row_counts if count_col else None, label=str(model))
    return out


def load_matrix_sets_from_os_csv(os_csv: str | Path, mods: Sequence[int], model: str | None, eps: float = 1e-12) -> list[LoadedMatrixSet]:
    """Read empirical and model transfer matrices from a CHL2 OS CSV."""
    df = pd.read_csv(os_csv)
    q_col = first_existing_column(df.columns, ["q", "modulus"])
    model_col = first_existing_column(df.columns, ["model", "model_name"])
    from_col = first_existing_column(df.columns, ["from_residue_b_prev_prime", "from_residue", "b"])
    to_col = first_existing_column(df.columns, ["to_residue_a_current_prime", "to_residue", "a"])
    emp_prob_col = first_existing_column(df.columns, ["empirical_prob", "empirical_probability", "probability_empirical"])
    model_prob_col = first_existing_column(df.columns, ["model_prob", "model_probability", "probability_model"])
    emp_count_col = first_existing_column(df.columns, ["empirical_count", "count_empirical", "observed_count"])
    row_count_col = first_existing_column(df.columns, ["row_count", "from_row_count"])
    block_col = first_existing_column(df.columns, ["block", "block_id", "window"])
    if q_col is None or from_col is None or to_col is None:
        raise ValueError("OS CSV missing q/from/to columns")
    if emp_prob_col is None and emp_count_col is None:
        raise ValueError("OS CSV must contain empirical probabilities or counts")
    if model_prob_col is None:
        raise ValueError("OS CSV must contain model probabilities")
    if model_col and model:
        df = df[df[model_col].astype(str) == str(model)]
    elif model_col:
        choices = list(df[model_col].astype(str).unique())
        selected = "CHL2_path_excl_cond_eta" if "CHL2_path_excl_cond_eta" in choices else choices[0]
        df = df[df[model_col].astype(str) == selected]
        model = selected
    else:
        model = model or "model"

    block_values = ["ALL"] if block_col is None else list(df[block_col].astype(str).unique())
    out: list[LoadedMatrixSet] = []
    for block in block_values:
        dfb = df if block_col is None else df[df[block_col].astype(str) == block]
        for q in mods:
            sub = dfb[dfb[q_col].astype(int) == int(q)]
            residues = reduced_residues(q)
            idx = {r: i for i, r in enumerate(residues)}
            emp_counts = np.zeros((len(residues), len(residues)), dtype=float)
            emp_probs = np.zeros_like(emp_counts)
            model_probs = np.zeros_like(emp_counts)
            row_counts = np.zeros(len(residues), dtype=float)
            for _, row in sub.iterrows():
                bi = idx.get(int(row[from_col]) % int(q))
                ai = idx.get(int(row[to_col]) % int(q))
                if bi is None or ai is None:
                    continue
                if emp_count_col is not None:
                    emp_counts[bi, ai] = float(row[emp_count_col])
                if emp_prob_col is not None:
                    emp_probs[bi, ai] = float(row[emp_prob_col])
                model_probs[bi, ai] = float(row[model_prob_col])
                if row_count_col is not None:
                    row_counts[bi] = max(row_counts[bi], float(row[row_count_col]))
            if emp_count_col is not None:
                emp = transfer_from_counts(q, emp_counts, residues, label="empirical")
            else:
                emp_probs = normalize_matrix_rows(emp_probs)
                emp = TransferMatrix(q=int(q), residues=tuple(residues), probabilities=emp_probs, row_counts=row_counts if row_count_col else None, counts=None, label="empirical")
            model_probs = normalize_matrix_rows(model_probs)
            model_mat = TransferMatrix(q=int(q), residues=tuple(residues), probabilities=model_probs, row_counts=emp.row_counts, label=str(model))
            out.append(LoadedMatrixSet(q=int(q), block=str(block), model_name=str(model), empirical=emp, model=model_mat))
    return out


def assign_parentwide_block_ids(indices: np.ndarray, cumulative: np.ndarray, drop_tail: bool) -> tuple[np.ndarray, int]:
    """Assign event indices to real parent-wide block IDs.

    Returns block IDs in 0..nblocks-1. Tail events beyond the last real block are
    set to -1 when ``drop_tail`` is true, or nblocks otherwise.
    """
    block_ids = np.searchsorted(cumulative, indices, side="right")
    nblocks = len(cumulative)
    tail_count = int((block_ids >= nblocks).sum())
    if drop_tail:
        block_ids = block_ids.astype(np.int64, copy=True)
        block_ids[block_ids >= nblocks] = -1
    return block_ids.astype(np.int64, copy=False), tail_count


def worker_transition_counts(task: tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]) -> dict[tuple[int, int], np.ndarray]:
    """Worker: count residue transitions by block ID and modulus."""
    from_p, to_p, block_ids, mods = task
    out: dict[tuple[int, int], np.ndarray] = {}
    valid = block_ids >= 0
    if not np.any(valid):
        return out
    from_p = from_p[valid]
    to_p = to_p[valid]
    block_ids = block_ids[valid]
    for q in mods:
        q = int(q)
        residues = reduced_residues(q)
        idx = {r: i for i, r in enumerate(residues)}
        b = (from_p % q).astype(np.int64)
        a = (to_p % q).astype(np.int64)
        keys = np.stack([block_ids, b, a], axis=1)
        unique, cnt = np.unique(keys, axis=0, return_counts=True)
        for (block_id, br, ar), c in zip(unique, cnt):
            bi = idx.get(int(br))
            ai = idx.get(int(ar))
            if bi is None or ai is None:
                continue
            key = (int(block_id), q)
            mat = out.get(key)
            if mat is None:
                mat = np.zeros((len(residues), len(residues)), dtype=float)
                out[key] = mat
            mat[bi, ai] += float(c)
    return out


def merge_counts_dict(dest: dict[tuple[int, int], np.ndarray], src: dict[tuple[int, int], np.ndarray]) -> None:
    for key, mat in src.items():
        if key not in dest:
            dest[key] = mat.copy()
        else:
            dest[key] += mat


def empirical_counts_from_prime_csv(
    prime_csv: str | Path,
    mods: Sequence[int],
    *,
    chunksize: int = 1_000_000,
    workers: int = 1,
    block_plan: BlockPlan | None = None,
    fixed_block_size: int = 0,
    drop_partial_blocks: bool = True,
    skip_first_transition: bool = False,
    max_pending_factor: int = 2,
) -> tuple[dict[tuple[str, int], np.ndarray], dict]:
    """Stream a prime CSV and return empirical residue-transition counts.

    Parallel processing is applied to vectorized transition-count tasks. Reading
    gzip CSV remains serial, but the grouping by ``(block, from, to)`` can use
    multiple worker processes.
    """
    mods = [int(q) for q in mods]
    counts_by_id: dict[tuple[int, int], np.ndarray] = {}
    prev: int | None = None
    transition_index = 0
    skipped_initial = 0
    tail_events_seen = 0
    total_events_counted = 0
    chunks_submitted = 0
    chunks_completed = 0
    workers_eff = os.cpu_count() or 1 if int(workers) == 0 else max(1, int(workers))

    cumulative: np.ndarray | None = None
    if block_plan and block_plan.mode == "parent-wide":
        cumulative = np.cumsum(np.asarray(block_plan.counts, dtype=np.int64))

    def submit_or_run(task, pool=None):
        nonlocal chunks_submitted, chunks_completed
        chunks_submitted += 1
        if pool is None:
            res = worker_transition_counts(task)
            merge_counts_dict(counts_by_id, res)
            chunks_completed += 1
            return None
        return pool.submit(worker_transition_counts, task)

    futures = []
    max_pending = max(1, workers_eff * max(1, int(max_pending_factor)))
    pool = ProcessPoolExecutor(max_workers=workers_eff) if workers_eff > 1 else None
    try:
        for arr in read_prime_column(prime_csv, chunksize=chunksize):
            if prev is not None:
                arr2 = np.concatenate([np.array([prev], dtype=np.int64), arr])
            else:
                arr2 = arr
            if len(arr2) < 2:
                if len(arr):
                    prev = int(arr[-1])
                continue
            from_p = arr2[:-1]
            to_p = arr2[1:]
            if skip_first_transition and transition_index == 0 and len(from_p):
                from_p = from_p[1:]
                to_p = to_p[1:]
                skipped_initial += 1
            n = len(from_p)
            if n <= 0:
                prev = int(arr[-1]) if len(arr) else prev
                continue
            idx = np.arange(transition_index, transition_index + n, dtype=np.int64)
            if cumulative is not None:
                block_ids, tail = assign_parentwide_block_ids(idx, cumulative, drop_tail=drop_partial_blocks)
                tail_events_seen += int(tail)
            elif fixed_block_size and fixed_block_size > 0:
                block_ids = (idx // int(fixed_block_size)).astype(np.int64)
            else:
                block_ids = np.zeros(n, dtype=np.int64)
            total_events_counted += int((block_ids >= 0).sum())
            task = (from_p, to_p, block_ids, mods)
            fut = submit_or_run(task, pool=pool)
            if fut is not None:
                futures.append(fut)
                while len(futures) >= max_pending:
                    done = []
                    for f in as_completed(futures, timeout=None):
                        merge_counts_dict(counts_by_id, f.result())
                        chunks_completed += 1
                        done.append(f)
                        break
                    futures = [f for f in futures if f not in done]
            transition_index += n
            prev = int(arr[-1]) if len(arr) else prev
        if pool is not None:
            for f in as_completed(futures):
                merge_counts_dict(counts_by_id, f.result())
                chunks_completed += 1
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    # Convert numeric block IDs to labels.
    out: dict[tuple[str, int], np.ndarray] = {}
    if block_plan and block_plan.mode == "parent-wide":
        labels = list(block_plan.labels)
    else:
        labels = []
    for (bid, q), mat in counts_by_id.items():
        if block_plan and block_plan.mode == "parent-wide":
            if 0 <= bid < len(labels):
                label = labels[bid]
            else:
                label = "TAIL_PARTIAL"
        elif fixed_block_size and fixed_block_size > 0:
            label = f"B{bid + 1:02d}"
        else:
            label = "ALL"
        out[(label, q)] = mat

    meta = {
        "workers_effective": workers_eff,
        "chunks_submitted": chunks_submitted,
        "chunks_completed": chunks_completed,
        "raw_adjacent_transitions_seen": int(transition_index + skipped_initial),
        "skipped_initial_transition": int(skipped_initial),
        "counted_transitions": int(total_events_counted),
        "tail_events_seen": int(tail_events_seen),
        "tail_events_dropped": int(tail_events_seen if drop_partial_blocks else 0),
        "block_mode": block_plan.mode if block_plan else ("fixed" if fixed_block_size else "all"),
    }
    return out, meta


def matrix_sets_from_prime_csv(
    prime_csv: str | Path,
    os_csv: str | Path,
    mods: Sequence[int],
    model: str | None,
    *,
    chunksize: int,
    workers: int,
    block_plan: BlockPlan | None,
    fixed_block_size: int,
    drop_partial_blocks: bool,
    skip_first_transition: bool,
) -> tuple[list[LoadedMatrixSet], dict]:
    """Build empirical block matrices from prime CSV and use OS CSV model matrices."""
    model_mats = build_model_matrices_from_os_csv(os_csv, mods=mods, model=model)
    emp_counts, meta = empirical_counts_from_prime_csv(
        prime_csv,
        mods=mods,
        chunksize=chunksize,
        workers=workers,
        block_plan=block_plan,
        fixed_block_size=fixed_block_size,
        drop_partial_blocks=drop_partial_blocks,
        skip_first_transition=skip_first_transition,
    )
    out: list[LoadedMatrixSet] = []
    for (block, q), counts in sorted(emp_counts.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if block == "TAIL_PARTIAL" and drop_partial_blocks:
            continue
        emp = transfer_from_counts(q, counts, reduced_residues(q), label="empirical")
        model_mat = model_mats[q]
        model2 = TransferMatrix(q=q, residues=model_mat.residues, probabilities=model_mat.probabilities, row_counts=emp.row_counts, label=model_mat.label)
        out.append(LoadedMatrixSet(q=q, block=block, model_name=model_mat.label, empirical=emp, model=model2))
    return out, meta


def write_outputs(matrix_sets: list[LoadedMatrixSet], output_dir: Path, eps: float) -> dict[str, int]:
    """Compute residuals/spectra and write all CHL4 audit CSVs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    empirical_rows = []
    model_rows = []
    residual_rows = []
    q3_rows = []
    spectrum_rows = []
    stability_rows = []

    for ms in matrix_sets:
        emp = ms.empirical
        model = ms.model
        residual = row_centered_log_residual(emp, model, eps=eps)
        residues = list(emp.residues)
        for i, b in enumerate(residues):
            row_count = 0.0 if emp.row_counts is None else float(emp.row_counts[i])
            for j, a in enumerate(residues):
                count = 0.0 if emp.counts is None else float(emp.counts[i, j])
                empirical_rows.append({
                    "block": ms.block,
                    "q": ms.q,
                    "from_residue_b": b,
                    "to_residue_a": a,
                    "empirical_count": count,
                    "row_count": row_count,
                    "empirical_probability": float(emp.probabilities[i, j]),
                })
                model_rows.append({
                    "block": ms.block,
                    "q": ms.q,
                    "model": ms.model_name,
                    "from_residue_b": b,
                    "to_residue_a": a,
                    "model_probability": float(model.probabilities[i, j]),
                    "model_expected_count": float(row_count * model.probabilities[i, j]),
                })
                residual_rows.append({
                    "block": ms.block,
                    "q": ms.q,
                    "from_residue_b": b,
                    "to_residue_a": a,
                    "row_centered_log_residual_emp_over_chl2": float(residual[i, j]),
                })
        if ms.q == 3:
            try:
                d_emp = diagonal_log_odds_q3(emp, eps=eps)
                d_model = diagonal_log_odds_q3(model, eps=eps)
                q3_rows.append({
                    "block": ms.block,
                    "q": ms.q,
                    "D_empirical": d_emp,
                    "D_chl2": d_model,
                    "D_residual_emp_minus_chl2": d_emp - d_model,
                    "diagonal_probability_empirical": diagonal_probability(emp, weighted=True),
                    "diagonal_probability_chl2": diagonal_probability(model, weighted=True),
                    "wrong_sign_chl2": bool(d_model > 0 and d_emp < 0),
                    "row_count_total": float(emp.row_counts.sum() if emp.row_counts is not None else 0.0),
                })
            except Exception as exc:
                q3_rows.append({"block": ms.block, "q": ms.q, "error": str(exc)})
        try:
            spec = character_spectrum(residual, q=ms.q, residues=emp.residues)
            erank = effective_rank_from_energy([r["energy"] for r in spec])
            for r in spec:
                row = dict(r)
                row.update({"block": ms.block, "effective_rank": erank})
                spectrum_rows.append(row)
        except Exception as exc:
            spectrum_rows.append({"block": ms.block, "q": ms.q, "error": str(exc)})

    q3_df = pd.DataFrame(q3_rows)
    if not q3_df.empty and "D_residual_emp_minus_chl2" in q3_df.columns:
        stable_q3 = q3_df[~q3_df["block"].astype(str).str.contains("TAIL_PARTIAL", na=False)].copy()
        vals = pd.to_numeric(stable_q3.get("D_residual_emp_minus_chl2"), errors="coerce").dropna()
        if len(vals):
            stability_rows.append({
                "diagnostic": "q3_D_residual",
                "n_blocks": int(len(vals)),
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(vals.min()),
                "max": float(vals.max()),
                "positive_count": int((vals > 0).sum()),
                "negative_count": int((vals < 0).sum()),
            })
    spec_df_tmp = pd.DataFrame(spectrum_rows)
    if not spec_df_tmp.empty and "energy_fraction" in spec_df_tmp.columns:
        spec_df_tmp = spec_df_tmp[~spec_df_tmp.get("block", "").astype(str).str.contains("TAIL_PARTIAL", na=False)]
        for q, sub in spec_df_tmp.groupby("q"):
            if "error" in sub.columns and sub["error"].notna().all():
                continue
            sub2 = sub[(sub.get("chi_index", -1) != 0) | (sub.get("psi_index", -1) != 0)]
            if len(sub2):
                stability_rows.append({
                    "diagnostic": "nonprincipal_energy_fraction_max_by_block",
                    "q": int(q),
                    "n_rows": int(len(sub2)),
                    "mean": float(pd.to_numeric(sub2["energy_fraction"], errors="coerce").mean()),
                    "max": float(pd.to_numeric(sub2["energy_fraction"], errors="coerce").max()),
                })

    pd.DataFrame(empirical_rows).to_csv(output_dir / "chl4_transfer_empirical_matrices.csv", index=False)
    pd.DataFrame(model_rows).to_csv(output_dir / "chl4_transfer_chl2_matrices.csv", index=False)
    pd.DataFrame(residual_rows).to_csv(output_dir / "chl4_transfer_residual_logratio.csv", index=False)
    pd.DataFrame(q3_rows).to_csv(output_dir / "chl4_q3_diagonal_scalar.csv", index=False)
    pd.DataFrame(spectrum_rows).to_csv(output_dir / "chl4_character_spectrum.csv", index=False)
    pd.DataFrame(stability_rows).to_csv(output_dir / "chl4_block_stability.csv", index=False)
    write_interpretation(output_dir, q3_df, pd.DataFrame(spectrum_rows))
    return {
        "matrix_sets": len(matrix_sets),
        "empirical_rows": len(empirical_rows),
        "residual_rows": len(residual_rows),
        "q3_rows": len(q3_rows),
        "spectrum_rows": len(spectrum_rows),
    }


def write_interpretation(output_dir: Path, q3_df: pd.DataFrame, spectrum_df: pd.DataFrame) -> None:
    """Write a human-readable interpretation markdown."""
    lines: list[str] = []
    lines.append("# CHL4 modular-transfer residual audit\n")
    lines.append("This audit measures the direct prime-residue transfer residual left by CHL2. It is diagnostic only; it does not fit a new model.\n")
    if not q3_df.empty and "D_empirical" in q3_df.columns:
        cols = ["block", "D_empirical", "D_chl2", "D_residual_emp_minus_chl2", "diagonal_probability_empirical", "diagonal_probability_chl2", "wrong_sign_chl2"]
        show = q3_df[[c for c in cols if c in q3_df.columns]].copy()
        lines.append("## q=3 diagonal/off-diagonal scalar\n")
        lines.append(safe_to_markdown(show, index=False))
        lines.append("\n")
        stable = q3_df[~q3_df["block"].astype(str).str.contains("TAIL_PARTIAL", na=False)].copy() if "block" in q3_df.columns else q3_df
        vals = pd.to_numeric(stable.get("D_residual_emp_minus_chl2"), errors="coerce").dropna()
        if len(vals):
            lines.append(f"Mean residual scalar D_emp - D_CHL2 over full blocks: `{vals.mean():.6g}`. Negative empirical D with positive CHL2 D indicates the known binary-residue anomaly.\n")
    if not spectrum_df.empty and "energy_fraction" in spectrum_df.columns:
        lines.append("## Character-spectrum note\n")
        lines.append("The file `chl4_character_spectrum.csv` contains the Dirichlet-character Fourier decomposition of the row-centered residual. For prime q, character index 0 is principal. The first CHL4 milestone is to test whether q=3 residual energy concentrates in the single non-principal mode.\n")
    lines.append("## Next step\n")
    lines.append("If the q=3 residual is stable by block and concentrated in a low-dimensional character mode, the next CHL4 phase is a local Oliver--Soundararajan projection.\n")
    (output_dir / "chl4_interpretacion.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["from-os-csv", "from-prime-csv"], default="from-os-csv")
    parser.add_argument("--os-csv", required=True, help="CHL2 OS transition CSV with empirical and model probabilities.")
    parser.add_argument("--model", default="CHL2_path_excl_cond_eta", help="Model name to use from the OS CSV.")
    parser.add_argument("--mods", default="3,5,7,11,13", help="Comma-separated prime moduli.")
    parser.add_argument("--prime-csv", default=None, help="Chronological prime CSV for mode=from-prime-csv.")
    parser.add_argument("--config", default=None, help="Optional dataset config used to align real parent-wide blocks.")
    parser.add_argument("--root", default=".", help="Repository/data root for resolving config paths.")
    parser.add_argument("--blocks", default="", help="Real block spec for parent-wide boundaries, e.g. '1-10'.")
    parser.add_argument("--block-boundary-mode", choices=["auto", "parent-wide", "fixed", "none"], default="auto")
    parser.add_argument("--block-count-col", default="H")
    parser.add_argument("--drop-partial-blocks", action=argparse.BooleanOptionalAction, default=True, help="Drop prime-stream tail beyond real block boundaries.")
    parser.add_argument("--skip-first-transition", action=argparse.BooleanOptionalAction, default=None, help="Skip the first adjacent prime transition; default true for parent-wide block boundaries.")
    parser.add_argument("--block-size-transitions", type=int, default=0, help="Sequential fixed block size; used only when boundary mode is fixed or no parent-wide blocks are supplied.")
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=1, help="Workers for vectorized prime-stream grouping; 0 means all CPUs.")
    parser.add_argument("--max-pending-factor", type=int, default=2, help="Max pending chunk tasks per worker.")
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    tele = telemetry_start()
    outdir = Path(args.output_dir)
    mods = parse_int_list(args.mods)
    if any(not (m > 2) for m in mods):
        raise ValueError("CHL4 residue transfer expects odd moduli > 2")

    config = read_json(args.config)
    block_plan: BlockPlan | None = None
    boundary_meta: dict = {}
    if args.mode == "from-prime-csv":
        requested_blocks = parse_block_spec(args.blocks)
        mode = args.block_boundary_mode
        if mode == "auto":
            mode = "parent-wide" if (args.config and requested_blocks) else ("fixed" if args.block_size_transitions else "none")
        if mode == "parent-wide":
            if not args.config:
                raise ValueError("--config is required for --block-boundary-mode parent-wide")
            if not requested_blocks:
                # Try config range; fallback to 1-10.
                nb = int(config.get("num_blocks", config.get("n_blocks", 10)))
                requested_blocks = list(range(1, nb + 1))
            block_plan = build_parent_wide_block_plan(
                config=config,
                config_path=args.config,
                root=args.root,
                blocks=requested_blocks,
                count_col=args.block_count_col,
            )
            print(f"[CHL4] using parent-wide block boundaries: {list(zip(block_plan.labels, block_plan.counts))}")
        elif mode == "fixed":
            block_plan = build_fixed_block_plan(args.block_size_transitions)
        elif mode == "none":
            block_plan = None
        boundary_meta = {
            "block_boundary_mode_effective": mode,
            "block_plan_labels": block_plan.labels if block_plan else [],
            "block_plan_counts": block_plan.counts if block_plan else [],
            "block_plan_source_files": block_plan.source_files if block_plan else [],
            "block_plan_total_expected_transitions": block_plan.total_expected_transitions if block_plan else 0,
        }

    if args.mode == "from-os-csv":
        matrix_sets = load_matrix_sets_from_os_csv(args.os_csv, mods=mods, model=args.model, eps=args.eps)
        stream_meta = {}
    else:
        if not args.prime_csv:
            raise ValueError("--prime-csv is required for mode=from-prime-csv")
        skip_first = bool(args.skip_first_transition) if args.skip_first_transition is not None else bool(block_plan and block_plan.mode == "parent-wide")
        matrix_sets, stream_meta = matrix_sets_from_prime_csv(
            prime_csv=args.prime_csv,
            os_csv=args.os_csv,
            mods=mods,
            model=args.model,
            chunksize=args.chunksize,
            workers=args.workers,
            block_plan=block_plan,
            fixed_block_size=args.block_size_transitions if (block_plan and block_plan.mode == "fixed") else 0,
            drop_partial_blocks=args.drop_partial_blocks,
            skip_first_transition=skip_first,
        )
    counts = write_outputs(matrix_sets, outdir, eps=args.eps)
    cfg = {
        "mode": args.mode,
        "os_csv": str(args.os_csv),
        "model": args.model,
        "mods": mods,
        "prime_csv": args.prime_csv,
        "config": args.config,
        "root": str(args.root),
        "blocks": args.blocks,
        "block_boundary_mode": args.block_boundary_mode,
        "block_size_transitions": args.block_size_transitions,
        "chunksize": args.chunksize,
        "workers": args.workers,
        "drop_partial_blocks": args.drop_partial_blocks,
        "skip_first_transition_effective": (bool(args.skip_first_transition) if args.skip_first_transition is not None else bool(block_plan and block_plan.mode == "parent-wide")),
        "eps": args.eps,
        "output_dir": str(outdir),
        **boundary_meta,
        **stream_meta,
        **counts,
    }
    (outdir / "chl4_config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")
    write_telemetry(outdir / "chl4_runtime_telemetry.json", tele, **cfg)
    print(f"[CHL4] wrote outputs to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
