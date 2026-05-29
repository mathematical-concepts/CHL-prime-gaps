#!/usr/bin/env python3
"""CHL2 Hit@K search oracle.

Given a current prime ``p_n`` and the previous gap ``g1 = p_n - p_{n-1}``, this
script ranks even candidate gaps ``g2`` by the parameter-free CHL2 state cost

    C_Y(g2 | g1, x) = -log R_Y(g2|g1) + Omega_Y^path(g1,g2;x).

Lower cost means a higher CHL2 path-exclusion score.  The script is a research
ranking tool; it does not prove primality and does not replace classical primality
tests.  Its purpose is to prioritize candidates before expensive verification.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import List

# Allow execution as a file from a fresh clone without requiring pip install -e . first.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chl_kernel import CHLKernel, even_candidates, survives_actual_wheel


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Rank next-gap candidates by CHL2 state cost.")
    ap.add_argument("--p-current", type=int, required=True, help="Current prime p_n.")
    ap.add_argument("--g-prev", type=int, required=True, help="Previous consecutive prime gap g1 = p_n - p_{n-1}.")
    ap.add_argument("--Y", type=int, default=47, help="Truncation horizon for singular-series products.")
    ap.add_argument("--gmax", type=int, default=2400, help="Maximum even candidate gap g2 to rank.")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--no-actual-wheel-mask", action="store_true", help="Do not discard candidates divisible by small q <= Y at the actual p_n residue.")
    ap.add_argument("--output-csv", default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    log_x = math.log(float(args.p_current))
    kernel = CHLKernel(Y=args.Y, log_x=log_x)
    rows: List[dict] = []
    for g2 in even_candidates(args.gmax):
        survives = True
        if not args.no_actual_wheel_mask:
            survives = survives_actual_wheel(args.p_current, g2, kernel.series.primes)
        if not survives:
            continue
        log_R = kernel.log_R(args.g_prev, g2)
        omega = kernel.omega_path(args.g_prev, g2)
        cost = math.inf if (not math.isfinite(log_R) or not math.isfinite(omega)) else -log_R + omega
        rows.append({
            "rank_cost": cost,
            "g2": int(g2),
            "candidate_n": int(args.p_current + g2),
            "log_R_Y": log_R,
            "Omega_path": omega,
            "score_exp_minus_cost": 0.0 if not math.isfinite(cost) else math.exp(-cost),
        })
    rows.sort(key=lambda r: (r["rank_cost"], r["g2"]))
    top = rows[: int(args.top_k)]

    if args.output_csv:
        path = Path(args.output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(top[0].keys()) if top else ["rank_cost", "g2", "candidate_n", "log_R_Y", "Omega_path", "score_exp_minus_cost"])
            writer.writeheader()
            writer.writerows(top)
        print(f"wrote {path}")
    else:
        print(f"CHL2 Hit@K oracle: p_n={args.p_current}, g1={args.g_prev}, Y={args.Y}, gmax={args.gmax}")
        for i, r in enumerate(top, 1):
            print(f"{i:4d}  g2={r['g2']:5d}  candidate={r['candidate_n']}  cost={r['rank_cost']:.8g}  logR={r['log_R_Y']:.8g}  Omega={r['Omega_path']:.8g}")


if __name__ == "__main__":
    main()
