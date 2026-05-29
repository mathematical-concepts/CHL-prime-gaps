#!/usr/bin/env python3
"""
CHL2 Y-Sweep runner
===================

Runs chl2_consecutive_exclusion_audit.py over a list of singular-series
horizons Y and aggregates the key stability diagnostics.

The sweep is deliberately a wrapper around the main audit script rather than a
separate model implementation.  This guarantees that CHL2, order-zero baselines,
Cramer--Granville baselines, and optional path-exclusion all use the same code
path for every horizon.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import pandas as pd


def parse_list(s: str) -> List[int]:
    out = []
    for part in str(s).split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            out.extend(range(int(a), int(b)+1))
        else:
            out.append(int(part))
    return sorted(set(out))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run CHL2 over multiple Y/Pmax horizons and aggregate stability metrics.")
    ap.add_argument("--script", default=None, help="Path to chl2_consecutive_exclusion_audit.py. Default: sibling script.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--blocks", default="1-10")
    ap.add_argument("--y-values", default="31,47,61,73", help="Comma-separated pmax/Y values, e.g. 31,47,61,73,97")
    ap.add_argument("--output-dir", default="chl2_y_sweep_outputs")
    ap.add_argument("--filters", default=None)
    ap.add_argument("--path-exclusion", action="store_true", help="Enable path-sensitive H4/H3 exclusion at every Y.")
    ap.add_argument("--workers", default="0", help="Workers passed to main audit. Use 0 for all cores.")
    ap.add_argument("--parallel-mode", default="auto", choices=["auto", "blocks", "path", "none"])
    ap.add_argument("--path-cache-source", default="all", choices=["first", "all"])
    ap.add_argument("--path-chunk-size", default="5000")
    ap.add_argument("--cache-maxsize", default="2000000")
    ap.add_argument("--prime-csv", default=None, help="Optional prime-csv argument passed to main audit for OS test; use AUTO to resolve config path.")
    ap.add_argument("--os-prime-mods", default="3,5,7")
    ap.add_argument("--os-model", default="auto")
    ap.add_argument("--reuse-existing", action="store_true", help="If a Y subdir already has chl2_conditional_summary.csv, do not rerun it.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    script = Path(args.script) if args.script else Path(__file__).resolve().parent / "chl2_consecutive_exclusion_audit.py"
    if not script.exists():
        raise FileNotFoundError(script)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    y_values = parse_list(args.y_values)

    run_rows = []
    for Y in y_values:
        subdir = outdir / f"Y{Y}"
        subdir.mkdir(parents=True, exist_ok=True)
        summary_path = subdir / "chl2_conditional_summary.csv"
        if args.reuse_existing and summary_path.exists():
            print(f"[Y-sweep] Reusing existing outputs for Y={Y}: {subdir}", flush=True)
        else:
            cmd = [
                sys.executable, str(script),
                "--config", str(args.config),
                "--root", str(args.root),
                "--blocks", str(args.blocks),
                "--y-mode", "pmax",
                "--pmax", str(Y),
                "--output-dir", str(subdir),
                "--workers", str(args.workers),
                "--parallel-mode", str(args.parallel_mode),
                "--path-cache-source", str(args.path_cache_source),
                "--path-chunk-size", str(args.path_chunk_size),
                "--cache-maxsize", str(args.cache_maxsize),
            ]
            if args.filters:
                cmd.extend(["--filters", str(args.filters)])
            if args.path_exclusion:
                cmd.append("--path-exclusion")
            if args.prime_csv:
                cmd.extend(["--prime-csv", str(args.prime_csv), "--os-prime-mods", str(args.os_prime_mods), "--os-model", str(args.os_model)])
            if args.dry_run:
                cmd.append("--dry-run")
            print("[Y-sweep]", " ".join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, check=True)
        run_rows.append({"Y": Y, "outdir": str(subdir), "summary_exists": summary_path.exists()})

    pd.DataFrame(run_rows).to_csv(outdir / "chl2_y_sweep_runs.csv", index=False)
    if args.dry_run:
        print(f"[Y-sweep] dry-run only. Wrote {outdir/'chl2_y_sweep_runs.csv'}")
        return

    frames = []
    gain_frames = []
    mem_frames = []
    os_frames = []
    for Y in y_values:
        subdir = outdir / f"Y{Y}"
        sp = subdir / "chl2_conditional_summary.csv"
        gp = subdir / "chl2_pairwise_gains.csv"
        mp = subdir / "chl2_memory_irreducibility.csv"
        osp = subdir / "chl2_os_prime_residue_summary.csv"
        if sp.exists():
            df = pd.read_csv(sp)
            df.insert(0, "Y", Y)
            frames.append(df)
        if gp.exists():
            df = pd.read_csv(gp)
            df.insert(0, "Y", Y)
            gain_frames.append(df)
        if mp.exists():
            df = pd.read_csv(mp)
            df.insert(0, "Y", Y)
            mem_frames.append(df)
        if osp.exists():
            df = pd.read_csv(osp)
            df.insert(0, "Y", Y)
            os_frames.append(df)
    if frames:
        all_summary = pd.concat(frames, ignore_index=True)
        all_summary.to_csv(outdir / "chl2_y_sweep_summary.csv", index=False)
    else:
        all_summary = pd.DataFrame()
    if gain_frames:
        all_gains = pd.concat(gain_frames, ignore_index=True)
        all_gains.to_csv(outdir / "chl2_y_sweep_gains.csv", index=False)
    else:
        all_gains = pd.DataFrame()
    if mem_frames:
        all_mem = pd.concat(mem_frames, ignore_index=True)
        all_mem.to_csv(outdir / "chl2_y_sweep_memory_irreducibility.csv", index=False)
    if os_frames:
        all_os = pd.concat(os_frames, ignore_index=True)
        all_os.to_csv(outdir / "chl2_y_sweep_os_prime_residue_summary.csv", index=False)

    rows = []
    main_candidates = ["CHL2_path_excl_cond_eta", "CHL2_gap_excl_cond_eta"]
    if not all_gains.empty:
        for f in sorted(all_gains["filter"].dropna().unique()):
            for baseline in ["CHL1_ratio_only_cond_eta", "Cramer_Granville_gap_excl_order0_exp", "Cramer_Granville_order0_exp", "Cramer_order0_exp", "HL2_gap_excl_order0_eta", "HL2_order0_eta"]:
                tmp = all_gains[(all_gains["filter"].eq(f)) & (all_gains["baseline"].eq(baseline)) & (all_gains["model"].isin(main_candidates))].copy()
                if tmp.empty:
                    continue
                # Prefer path model if present at the same Y/filter/baseline.
                chosen = []
                for Y in y_values:
                    ty = tmp[tmp["Y"].eq(Y)]
                    if ty.empty:
                        continue
                    if "CHL2_path_excl_cond_eta" in set(ty["model"]):
                        chosen.append(ty[ty["model"].eq("CHL2_path_excl_cond_eta")].iloc[0])
                    else:
                        chosen.append(ty.iloc[0])
                if not chosen:
                    continue
                cdf = pd.DataFrame(chosen)
                vals = cdf["delta_loglik_model_minus_baseline"].astype(float)
                rows.append({
                    "filter": f,
                    "baseline": baseline,
                    "n_Y": int(vals.size),
                    "Y_values": ",".join(map(str, cdf["Y"].astype(int).tolist())),
                    "min_delta_loglik": float(vals.min()),
                    "mean_delta_loglik": float(vals.mean()),
                    "max_delta_loglik": float(vals.max()),
                    "positive_Y_count": int((vals > 0).sum()),
                    "all_positive": bool((vals > 0).all()),
                })
    stability = pd.DataFrame(rows)
    stability.to_csv(outdir / "chl2_y_sweep_stability.csv", index=False)

    text = ["# CHL2 Y-sweep interpretation", "", f"Y values: `{','.join(map(str,y_values))}`.", ""]
    if not stability.empty:
        text.append("## Stability of CHL2 gains")
        for _, r in stability.iterrows():
            text.append(f"- `{r['filter']}` vs `{r['baseline']}`: min Δloglik/event `{r['min_delta_loglik']:.8g}`, mean `{r['mean_delta_loglik']:.8g}`, positive `{int(r['positive_Y_count'])}/{int(r['n_Y'])}` over Y=`{r['Y_values']}`.")
        text.append("")
    text.append("## Reading rule")
    text.append("Positive deltas over several Y horizons indicate that the improvement is not an artifact of a single truncation boundary. Oscillation or sign changes indicate a possible compression shadow and should be treated as a limitation rather than hidden post-hoc success.")
    (outdir / "chl2_y_sweep_interpretacion.md").write_text("\n".join(text), encoding="utf-8")
    print(f"[Y-sweep] wrote aggregate outputs to {outdir}")


if __name__ == "__main__":
    main()
