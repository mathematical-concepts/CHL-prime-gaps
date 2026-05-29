#!/usr/bin/env python3
"""Compute Pearson chi-square diagnostics from CHL2 OS transition matrix CSV.

Usage:
  python chl2_os_chisquare_from_matrix.py \
    --matrix-csv chl2_os_prime_residue_transition_by_mod.csv \
    --summary-csv chl2_os_prime_residue_summary.csv \
    --out chl2_os_prime_residue_chisquare.csv

The script is intentionally standalone so the q=3 anomaly can be audited from
already-generated CHL2 OS outputs without rerunning the full path-exclusion kernel.
"""
from __future__ import annotations
import argparse
import sys
import math
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

# Allow execution as a file from a fresh clone without requiring pip install -e . first.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from chl_kernel.telemetry import telemetry_start, write_telemetry


def safe_chi2_sf(x: float, df: int) -> float:
    if not np.isfinite(x) or df <= 0:
        return float('nan')
    try:
        from scipy.stats import chi2  # type: ignore
        return float(chi2.sf(x, df))
    except Exception:
        return float('nan')


def pearson_for_group(g: pd.DataFrame) -> Dict[str, float]:
    residues_from = sorted(g['from_residue_b_prev_prime'].unique())
    residues_to = sorted(g['to_residue_a_current_prime'].unique())
    row_index = {r:i for i,r in enumerate(residues_from)}
    col_index = {r:j for j,r in enumerate(residues_to)}
    O = np.zeros((len(residues_from), len(residues_to)), dtype=float)
    E = np.zeros_like(O)
    for _, r in g.iterrows():
        i = row_index[r['from_residue_b_prev_prime']]
        j = col_index[r['to_residue_a_current_prime']]
        O[i,j] += float(r['empirical_count'])
        # Prefer raw accumulated expected counts; if absent, use model_prob*row_count.
        if 'model_expected_count' in g.columns and not pd.isna(r.get('model_expected_count')):
            E[i,j] += float(r['model_expected_count'])
        else:
            E[i,j] += float(r['model_prob']) * float(r['row_count'])
    row_counts = O.sum(axis=1)
    # Row-normalize expected counts to the empirical row totals.
    E2 = E.copy()
    for i, rc in enumerate(row_counts):
        es = E2[i].sum()
        if rc > 0 and es > 0:
            E2[i] *= rc / es
    chi2 = 0.0
    df = 0
    cells = 0
    rows = 0
    inf_flag = False
    eps = 1e-15
    for i, rc in enumerate(row_counts):
        if rc <= 0:
            continue
        pos_exp = E2[i] > eps
        pos_obs = O[i] > 0
        if np.any(pos_obs & ~pos_exp):
            inf_flag = True
        if pos_exp.any():
            chi2 += float(np.sum((O[i,pos_exp]-E2[i,pos_exp])**2/E2[i,pos_exp]))
            df += max(int(pos_exp.sum()) - 1, 0)
            cells += int(pos_exp.sum())
            rows += 1
    if inf_flag:
        chi2 = float('inf')
    total = float(row_counts.sum())
    diag_emp = float(np.trace(O) / total) if total > 0 else float('nan')
    diag_model = float(np.trace(E2) / total) if total > 0 else float('nan')
    uniform_diag = 1.0 / float(len(residues_to)) if residues_to else float('nan')
    return {
        'q': int(g['q'].iloc[0]),
        'model': str(g['model'].iloc[0]),
        'n_cells': int(O.size),
        'n_rows_used': int(rows),
        'n_cells_used': int(cells),
        'used_transitions_from_matrix': total,
        'pearson_chi2': float(chi2),
        'pearson_chi2_df': int(df),
        'pearson_chi2_pvalue': safe_chi2_sf(chi2, df),
        'pearson_chi2_per_transition': float(chi2/total) if total > 0 and np.isfinite(chi2) else float('inf'),
        'diagonal_probability_empirical': diag_emp,
        'diagonal_probability_model': diag_model,
        'uniform_diagonal_probability': uniform_diag,
        'diagonal_wrong_sign_vs_uniform': bool((diag_emp-uniform_diag)*(diag_model-uniform_diag) < 0) if all(map(np.isfinite,[diag_emp,diag_model,uniform_diag])) else False,
        'infinite_flag': bool(inf_flag),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--matrix-csv', required=True)
    ap.add_argument('--summary-csv', default=None, help='Optional existing summary CSV to enrich/merge')
    ap.add_argument('--out', default='chl2_os_prime_residue_chisquare.csv')
    ap.add_argument('--summary-out', default=None, help='Optional merged summary output')
    ap.add_argument('--telemetry-json', default=None, help='Optional runtime telemetry JSON path. Default: sibling of --out.')
    args = ap.parse_args()
    telemetry = telemetry_start()
    telemetry["script"] = "os_prime_residue_diagnostic"
    telemetry["args"] = vars(args)
    mat = pd.read_csv(args.matrix_csv)
    rows: List[Dict[str, float]] = []
    for (_, _), grp in mat.groupby(['q','model']):
        rows.append(pearson_for_group(grp))
    chi = pd.DataFrame(rows).sort_values('q')
    chi.to_csv(args.out, index=False)
    print(f'wrote {args.out}')
    if args.summary_csv:
        summ = pd.read_csv(args.summary_csv)
        merged = summ.merge(chi.drop(columns=['model'], errors='ignore'), on='q', how='left', suffixes=('','_chisq'))
        out = args.summary_out or str(Path(args.summary_csv).with_name(Path(args.summary_csv).stem + '_with_chisquare.csv'))
        merged.to_csv(out, index=False)
        print(f'wrote {out}')
    telemetry_path = Path(args.telemetry_json) if args.telemetry_json else Path(args.out).with_name('os_prime_residue_diagnostic_telemetry.json')
    write_telemetry(
        telemetry_path,
        telemetry,
        matrix_csv=str(args.matrix_csv),
        summary_csv=str(args.summary_csv) if args.summary_csv else None,
        output_csv=str(args.out),
        matrix_rows=int(len(mat)),
        chi_square_rows=int(len(chi)),
        modules=sorted([int(x) for x in mat['q'].dropna().unique()]) if 'q' in mat.columns else [],
        models=sorted([str(x) for x in mat['model'].dropna().unique()]) if 'model' in mat.columns else [],
    )

if __name__ == '__main__':
    main()
