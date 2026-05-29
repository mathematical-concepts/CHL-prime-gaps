#!/usr/bin/env python3
"""Generate empirical consecutive-prime-gap blocks from scratch.

This script is intentionally theory-free.  It uses a segmented sieve of
Eratosthenes to generate primes in a user-specified interval and then exports
histograms of consecutive gap pairs ``(g1,g2)``.  These histograms are the input
format used by the CHL paper-audit scripts.

Default DS1-style use
---------------------
python data_generation/generate_prime_gap_blocks.py \
  --start 100000000000 \
  --end 102000000000 \
  --gmax 2400 \
  --num-blocks 10 \
  --workers 16 \
  --output-dir data/ds1_1e11_w2e9_g2400

Quick test
----------
python data_generation/generate_prime_gap_blocks.py --quick-test --output-dir data/quick_test
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

# Allow execution as a file from a fresh clone without requiring pip install -e . first.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from chl_kernel.telemetry import telemetry_start, write_telemetry


def ensure_dir(path: Path) -> None:
    """Create a directory if needed."""
    path.mkdir(parents=True, exist_ok=True)


def simple_sieve(limit: int) -> np.ndarray:
    """Return primes up to ``limit`` using a simple sieve."""
    limit = int(limit)
    if limit < 2:
        return np.array([], dtype=np.uint64)
    arr = np.ones(limit + 1, dtype=bool)
    arr[:2] = False
    r = int(math.isqrt(limit))
    for p in range(2, r + 1):
        if arr[p]:
            arr[p * p::p] = False
    return np.flatnonzero(arr).astype(np.uint64)


def segmented_sieve_to_file(payload: Tuple[int, int, int, str, np.ndarray]) -> Dict[str, object]:
    """Sieve one interval segment and save its prime array as ``.npy``."""
    idx, low, high, filepath, base_primes = payload
    path = Path(filepath)
    if path.exists():
        try:
            arr = np.load(path, mmap_mode="r")
            return {"idx": idx, "low": low, "high": high, "path": str(path), "count": int(len(arr))}
        except Exception:
            pass

    size = high - low + 1
    is_prime = np.ones(size, dtype=bool)
    if low == 0:
        if size > 0:
            is_prime[0] = False
        if size > 1:
            is_prime[1] = False
    elif low == 1:
        is_prime[0] = False

    for p in base_primes:
        pp = int(p)
        if pp * pp > high:
            break
        start = max(pp * pp, ((low + pp - 1) // pp) * pp)
        is_prime[start - low::pp] = False

    primes = np.flatnonzero(is_prime).astype(np.uint64) + np.uint64(low)
    np.save(path, primes)
    return {"idx": idx, "low": low, "high": high, "path": str(path), "count": int(len(primes))}


def load_prime_segments(segment_df: pd.DataFrame) -> np.ndarray:
    """Load sorted prime segments into one array.

    DS1-sized windows contain only a few million primes, so materializing this
    sequence is practical and simplifies exact chronological block construction.
    """
    arrays = []
    for _, row in segment_df.sort_values("idx").iterrows():
        arr = np.load(row["path"], mmap_mode="r")
        if len(arr):
            arrays.append(np.asarray(arr, dtype=np.uint64))
    if not arrays:
        return np.array([], dtype=np.uint64)
    return np.concatenate(arrays)


def compute_features(g1: int, g2: int, log_x_floor: int, alpha: float = 2.0, beta: float = 2.0) -> Dict[str, object]:
    """Return compatibility features for a gap pair.

    CHL2 does not use these empirical geometric features, but they are exported
    for historical compatibility and independent residual studies.
    """
    S = int(g1 + g2)
    D = int(g2 - g1)
    max_g = max(int(g1), int(g2))
    min_g = min(int(g1), int(g2))
    if S > 0:
        shape = abs(D) ** alpha / (S ** (alpha - 1.0))
        balance = -abs(D) / S
    else:
        shape = 0.0
        balance = 0.0
    L = max(1, int(log_x_floor))
    tail = -((S / L) ** beta)
    return {
        "S": S,
        "D": D,
        "max_g": max_g,
        "min_g": min_g,
        "shape": float(shape),
        "balance": float(balance),
        "tail": float(tail),
        "g1_mod30": int(g1 % 30),
        "g2_mod30": int(g2 % 30),
        "residue30_pair": f"{int(g1 % 30)}:{int(g2 % 30)}",
    }


def build_blocks(
    primes: np.ndarray,
    start: int,
    end: int,
    gmax: int,
    num_blocks: int,
    log_x_floor: int,
) -> Tuple[List[Dict[Tuple[int, int], int]], List[int], List[int], int]:
    """Build equal-mass blocks of observed consecutive gap pairs.

    A pair ``(g1,g2)`` is assigned to the window if the middle prime ``p_n`` lies
    in ``[start,end]`` and the two neighboring primes are present in the context.
    """
    if len(primes) < 3:
        raise RuntimeError("not enough primes in interval/context to build gap pairs")

    pairs: List[Tuple[int, int, int]] = []  # middle prime, g1, g2
    for i in range(1, len(primes) - 1):
        p_mid = int(primes[i])
        if start <= p_mid <= end:
            g1 = int(primes[i] - primes[i - 1])
            g2 = int(primes[i + 1] - primes[i])
            pairs.append((p_mid, g1, g2))
    n_pairs = len(pairs)
    if n_pairs == 0:
        raise RuntimeError("no gap pairs found in the requested empirical window")

    boundaries = [i * n_pairs // num_blocks for i in range(num_blocks)] + [n_pairs]
    H_blocks: List[Dict[Tuple[int, int], int]] = [dict() for _ in range(num_blocks)]
    raw_counts = [0] * num_blocks
    p_centers = [0] * num_blocks

    for b in range(num_blocks):
        lo, hi = boundaries[b], boundaries[b + 1]
        if lo >= hi:
            continue
        raw_counts[b] = hi - lo
        p_centers[b] = int(pairs[(lo + hi) // 2][0])
        for _, g1, g2 in pairs[lo:hi]:
            if g1 <= gmax and g2 <= gmax:
                key = (int(g1), int(g2))
                H_blocks[b][key] = H_blocks[b].get(key, 0) + 1
    return H_blocks, raw_counts, p_centers, n_pairs


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate CHL empirical prime-gap blocks from a segmented sieve.")
    ap.add_argument("--start", type=int, default=100_000_000_000, help="Window start; middle primes p_n are selected from [start,end].")
    ap.add_argument("--end", type=int, default=102_000_000_000, help="Window end, inclusive.")
    ap.add_argument("--context-margin", type=int, default=1_000_000, help="Extra range on both sides to capture neighboring primes.")
    ap.add_argument("--gmax", type=int, default=2400)
    ap.add_argument("--num-blocks", type=int, default=10)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--segment-size", type=int, default=50_000_000)
    ap.add_argument("--cache-dir", default=".prime_cache")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--quick-test", action="store_true", help="Use a small interval near 1e6 for smoke tests.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    telemetry = telemetry_start()
    telemetry["script"] = "generate_prime_gap_blocks"
    telemetry["args"] = vars(args)
    if args.quick_test:
        args.start = 1_000_000
        args.end = 1_200_000
        args.context_margin = 10_000
        args.gmax = 300
        args.num_blocks = 2
        args.segment_size = 50_000
        print(">>> QUICK TEST MODE: [1,000,000, 1,200,000] <<<", flush=True)

    start, end = int(args.start), int(args.end)
    if end <= start:
        raise ValueError("--end must exceed --start")
    sieve_low = max(2, start - int(args.context_margin))
    sieve_high = end + int(args.context_margin)
    outdir = Path(args.output_dir)
    blocks_dir = outdir / "blocks"
    cache_dir = Path(args.cache_dir) / f"segments_{sieve_low}_{sieve_high}"
    ensure_dir(outdir)
    ensure_dir(blocks_dir)
    ensure_dir(cache_dir)

    print(f"[data_generation] window=[{start:,},{end:,}], sieve=[{sieve_low:,},{sieve_high:,}]", flush=True)
    sqrt_high = int(math.isqrt(sieve_high)) + 1
    base_primes = simple_sieve(sqrt_high)

    tasks = []
    idx, low = 0, sieve_low
    while low <= sieve_high:
        high = min(sieve_high, low + int(args.segment_size) - 1)
        tasks.append((idx, low, high, str(cache_dir / f"seg_{idx:05d}.npy"), base_primes))
        idx += 1
        low = high + 1

    print(f"[data_generation] sieving {len(tasks)} segments with {args.workers} workers", flush=True)
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
        futures = [ex.submit(segmented_sieve_to_file, t) for t in tasks]
        for k, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if k == len(futures) or k % max(1, len(futures)//10) == 0:
                print(f"[data_generation] segments done {k}/{len(futures)}", flush=True)
    segment_df = pd.DataFrame(rows).sort_values("idx")
    segment_df.to_csv(outdir / "segment_manifest.csv", index=False)
    primes = load_prime_segments(segment_df)
    print(f"[data_generation] loaded {len(primes):,} primes in context in {time.time()-t0:.1f}s", flush=True)

    prime_path = outdir / "real_primes.csv.gz"
    pd.DataFrame({"p": primes[(primes >= start) & (primes <= end)].astype(np.uint64)}).to_csv(prime_path, index=False, compression="gzip")

    log_x_floor = int(math.floor(math.log((start + end) / 2.0)))
    H_blocks, raw_counts, p_centers, n_pairs = build_blocks(primes, start, end, int(args.gmax), int(args.num_blocks), log_x_floor)

    for b in range(int(args.num_blocks)):
        out_path = blocks_dir / f"parent_wide_B{b+1:02d}.csv.gz"
        rows_out = []
        for (g1, g2), count in sorted(H_blocks[b].items()):
            feat = compute_features(g1, g2, log_x_floor)
            rows_out.append({"block": b + 1, "p_center": p_centers[b], "g1": g1, "g2": g2, **feat, "H": count})
        pd.DataFrame(rows_out).to_csv(out_path, index=False, compression="gzip")
        print(f"[data_generation] wrote {out_path} rows={len(rows_out):,} raw_pairs={raw_counts[b]:,}", flush=True)

    cfg = {
        "dataset_name": outdir.name,
        "input_dir": str(outdir),
        "blocks_dir": "blocks",
        "block_glob": "parent_wide_B{block:02d}.csv.gz",
        "blocks": list(range(1, int(args.num_blocks) + 1)),
        "real_prime_sequence": "real_primes.csv.gz",
        "start": start,
        "end": end,
        "context_margin": int(args.context_margin),
        "gmax": int(args.gmax),
        "num_blocks": int(args.num_blocks),
        "log_x_floor": log_x_floor,
        "n_pairs_middle_prime_window": int(n_pairs),
    }
    with (outdir / "config.generated.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    write_telemetry(
        outdir / "data_generation_telemetry.json",
        telemetry,
        output_dir=str(outdir),
        start=int(start),
        end=int(end),
        sieve_low=int(sieve_low),
        sieve_high=int(sieve_high),
        workers=int(args.workers),
        segment_size=int(args.segment_size),
        segment_count=int(len(tasks)),
        context_prime_count=int(len(primes)),
        window_prime_count=int(((primes >= start) & (primes <= end)).sum()),
        gmax=int(args.gmax),
        num_blocks=int(args.num_blocks),
        n_pairs_middle_prime_window=int(n_pairs),
        raw_pairs_by_block=[int(x) for x in raw_counts],
        block_files=[str(blocks_dir / f"parent_wide_B{b+1:02d}.csv.gz") for b in range(int(args.num_blocks))],
    )
    print(f"[data_generation] complete. Config: {outdir/'config.generated.json'}", flush=True)


if __name__ == "__main__":
    main()
