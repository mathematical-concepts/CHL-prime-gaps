#!/usr/bin/env python3
"""
HOLT_OMEGA_LENGTH_CONFOUND_AUDIT
================================

Supplemental audit for the CHL-prime-gaps project.

Purpose
-------
The Holt survival-interval compatibility audit observed that the diagonal gap
residue branch r=0 often has larger mean CHL2 path intensity Omega_path than
nonzero branches.  This script checks the main possible confound:

    Omega_path grows with the size of the candidate gap.

Since a fixed exact gap g has a fixed residue g mod q, one cannot compare
residues at identical g.  Instead, this audit uses length-bin matching:

  * raw difference: mean Omega(r=0) - mean Omega(r!=0);
  * length-matched difference: within each gap-length bin, compare r=0 and
    r!=0, then average bin differences using the r=0 event distribution;
  * composition component: the part attributable to different gap-length
    distributions across residue classes.

This is a diagnostic decomposition, not a theorem and not an implementation of
Holt's exact G(p#) population recurrences.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_blocks(spec: str) -> list[int]:
    out: list[int] = []
    for part in str(spec).split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def resolve_block_files(config: dict, root: Path, blocks: Sequence[int]) -> list[tuple[int, Path]]:
    input_dir = root / config.get('input_dir', '')
    blocks_dir = input_dir / config.get('blocks_dir', 'blocks')
    glob_pat = config.get('block_glob', 'parent_wide_B{block:02d}.csv.gz')
    files = [(b, blocks_dir / glob_pat.format(block=b)) for b in blocks]
    missing = [str(p) for _, p in files if not p.exists()]
    if missing:
        raise FileNotFoundError('Missing block files:\n' + '\n'.join(missing[:10]))
    return files


def load_pair_counts(block_files: Sequence[tuple[int, Path]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for block, path in block_files:
        df = pd.read_csv(path)
        required = {'g1', 'g2', 'H'}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f'{path} missing required columns: {sorted(missing)}')
        sub = df[['g1', 'g2', 'H']].copy()
        sub['block'] = int(block)
        sub['g1'] = sub['g1'].astype(int)
        sub['g2'] = sub['g2'].astype(int)
        sub['H'] = sub['H'].astype(float)
        sub = sub[sub['H'] > 0]
        frames.append(sub)
    if not frames:
        raise ValueError('No pair-count blocks loaded')
    out = pd.concat(frames, ignore_index=True)
    out['Gmax'] = np.maximum(out['g1'], out['g2'])
    return out


def load_path_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {'g1', 'g2'}
    if not required.issubset(df.columns):
        raise ValueError(f'{path} must contain g1,g2')

    if 'omega_path' in df.columns:
        omega = df['omega_path'].astype(float)
        source_col = 'omega_path'
    elif 'omega_path_exclusion' in df.columns:
        omega = df['omega_path_exclusion'].astype(float)
        source_col = 'omega_path_exclusion'
    elif 'logE_path_exclusion' in df.columns:
        omega = -df['logE_path_exclusion'].astype(float)
        source_col = 'logE_path_exclusion'
    else:
        raise ValueError(
            f'{path} must contain one of omega_path, omega_path_exclusion, logE_path_exclusion'
        )

    out = df[['g1', 'g2']].copy()
    out['g1'] = out['g1'].astype(int)
    out['g2'] = out['g2'].astype(int)
    out['omega_path'] = omega
    out['omega_source_column'] = source_col
    return out


def weighted_mean(values: pd.Series | np.ndarray, weights: pd.Series | np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    total = float(w.sum())
    if total <= 0:
        return float('nan')
    return float(np.dot(v, w) / total)


def weighted_std(values: pd.Series | np.ndarray, weights: pd.Series | np.ndarray) -> float:
    mu = weighted_mean(values, weights)
    if not math.isfinite(mu):
        return float('nan')
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    total = float(w.sum())
    if total <= 0:
        return float('nan')
    return float(math.sqrt(np.dot(w, (v - mu) ** 2) / total))


def bin_gap_length(g: pd.Series, width: int) -> pd.DataFrame:
    # Even gaps start at 2.  Bins are [2+k*w, 2+(k+1)*w-1].
    idx = np.floor((g.astype(int).to_numpy() - 2) / width).astype(int)
    idx[idx < 0] = 0
    start = 2 + idx * width
    end = 2 + (idx + 1) * width - 1
    return pd.DataFrame({'bin_id': idx, 'bin_start': start, 'bin_end': end})


# ---------------------------------------------------------------------------
# Audit calculations
# ---------------------------------------------------------------------------


def raw_by_residue(df: pd.DataFrame, mods: Sequence[int]) -> pd.DataFrame:
    rows = []
    for q in mods:
        tmp = df.copy()
        tmp['gap_residue_r'] = tmp['g2'] % q
        for r, grp in tmp.groupby('gap_residue_r'):
            H = grp['H']
            rows.append({
                'q': q,
                'gap_residue_r': int(r),
                'count': float(H.sum()),
                'share': float(H.sum() / tmp['H'].sum()),
                'mean_g2': weighted_mean(grp['g2'], H),
                'mean_Gmax': weighted_mean(grp['Gmax'], H),
                'mean_omega_path': weighted_mean(grp['omega_path'], H),
                'std_omega_path': weighted_std(grp['omega_path'], H),
                'mean_exp_neg_omega_path': weighted_mean(np.exp(-grp['omega_path'].to_numpy()), H),
            })
    return pd.DataFrame(rows)


def length_bin_decomposition(df: pd.DataFrame, mods: Sequence[int], widths: Sequence[int], gap_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows = []
    summary_rows = []

    for q in mods:
        base = df.copy()
        base['gap_residue_r'] = base['g2'] % q
        base['is_r0'] = base['gap_residue_r'] == 0
        total_H = float(base['H'].sum())

        raw0 = base[base['is_r0']]
        rawN = base[~base['is_r0']]
        raw_mean0 = weighted_mean(raw0['omega_path'], raw0['H'])
        raw_meanN = weighted_mean(rawN['omega_path'], rawN['H'])
        raw_delta = raw_mean0 - raw_meanN

        for width in widths:
            bins = bin_gap_length(base[gap_col], width)
            tmp = base.reset_index(drop=True).join(bins)

            # Per-bin, per-class means.
            grouped = tmp.groupby(['bin_id', 'bin_start', 'bin_end', 'is_r0'], as_index=False).apply(
                lambda g: pd.Series({
                    'count': float(g['H'].sum()),
                    'mean_omega_path': weighted_mean(g['omega_path'], g['H']),
                    'mean_g2': weighted_mean(g['g2'], g['H']),
                    'mean_Gmax': weighted_mean(g['Gmax'], g['H']),
                })
            ).reset_index(drop=True)

            # Detail rows.
            for _, row in grouped.iterrows():
                detail_rows.append({
                    'q': q,
                    'gap_length_variable': gap_col,
                    'bin_width': width,
                    'bin_id': int(row['bin_id']),
                    'bin_start': int(row['bin_start']),
                    'bin_end': int(row['bin_end']),
                    'class': 'r0' if bool(row['is_r0']) else 'nonzero',
                    'count': row['count'],
                    'mean_omega_path': row['mean_omega_path'],
                    'mean_g2': row['mean_g2'],
                    'mean_Gmax': row['mean_Gmax'],
                })

            # Pivot for matched comparison.
            wide = grouped.pivot_table(
                index=['bin_id', 'bin_start', 'bin_end'],
                columns='is_r0',
                values=['count', 'mean_omega_path'],
                aggfunc='first',
            )
            # Flatten safe column references.
            def get_col(kind, cls):
                try:
                    return wide[(kind, cls)]
                except KeyError:
                    return pd.Series(index=wide.index, dtype=float)

            cN = get_col('count', False).fillna(0.0)
            c0 = get_col('count', True).fillna(0.0)
            mN = get_col('mean_omega_path', False)
            m0 = get_col('mean_omega_path', True)

            both = (c0 > 0) & (cN > 0) & np.isfinite(m0) & np.isfinite(mN)
            matched_r0_count = float(c0[both].sum())
            total_r0_count = float(raw0['H'].sum())
            matched_nonzero_count = float(cN[both].sum())
            total_nonzero_count = float(rawN['H'].sum())

            if matched_r0_count > 0:
                w0 = c0[both] / matched_r0_count
                matched_mean0 = float((w0 * m0[both]).sum())
                length_matched_nonzero = float((w0 * mN[both]).sum())
                within_bin_delta = matched_mean0 - length_matched_nonzero
            else:
                matched_mean0 = float('nan')
                length_matched_nonzero = float('nan')
                within_bin_delta = float('nan')

            # Raw delta decomposition is only approximate if coverage < 1.
            composition_component = raw_delta - within_bin_delta if math.isfinite(within_bin_delta) else float('nan')

            summary_rows.append({
                'q': q,
                'gap_length_variable': gap_col,
                'bin_width': width,
                'raw_mean_omega_r0': raw_mean0,
                'raw_mean_omega_nonzero': raw_meanN,
                'raw_delta_r0_minus_nonzero': raw_delta,
                'matched_mean_omega_r0': matched_mean0,
                'length_matched_mean_omega_nonzero_using_r0_bins': length_matched_nonzero,
                'within_bin_delta_r0_minus_nonzero': within_bin_delta,
                'composition_component_approx': composition_component,
                'matched_r0_coverage': matched_r0_count / total_r0_count if total_r0_count > 0 else float('nan'),
                'matched_nonzero_coverage': matched_nonzero_count / total_nonzero_count if total_nonzero_count > 0 else float('nan'),
                'n_bins_total': int(len(wide)),
                'n_bins_matched': int(both.sum()),
                'total_events': total_H,
            })

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def write_interpretation(outdir: Path, raw: pd.DataFrame, decomp: pd.DataFrame, telemetry: dict) -> None:
    lines = []
    lines.append('# HOLT_OMEGA_LENGTH_CONFOUND_AUDIT')
    lines.append('')
    lines.append('This supplemental audit checks whether the observed larger mean $\\Omega_Y^{path}$ on the diagonal gap-residue branch $r=0$ is merely a gap-length effect.')
    lines.append('')
    lines.append('The audit is a diagnostic decomposition, not an implementation of Holt\'s exact $G(p\#)$ recurrence.')
    lines.append('')
    lines.append('## Raw residue-level means')
    lines.append('')
    # Summarize q, r=0 vs nonzero.
    rows = []
    for q, grp in raw.groupby('q'):
        r0 = grp[grp['gap_residue_r'] == 0]
        nz = grp[grp['gap_residue_r'] != 0]
        if r0.empty or nz.empty:
            continue
        rows.append({
            'q': int(q),
            'mean_omega_r0': float(r0['mean_omega_path'].iloc[0]),
            'mean_omega_nonzero_avg': float(np.average(nz['mean_omega_path'], weights=nz['count'])),
            'mean_g2_r0': float(r0['mean_g2'].iloc[0]),
            'mean_g2_nonzero_avg': float(np.average(nz['mean_g2'], weights=nz['count'])),
        })
    tab = pd.DataFrame(rows)
    if not tab.empty:
        lines.append(tab.to_markdown(index=False, floatfmt='.6f'))
    lines.append('')
    lines.append('## Length-bin control')
    lines.append('')
    lines.append('Because the exact residue $g\\bmod q$ is determined by the exact gap $g$, comparing residues at identical $g$ is impossible. The audit therefore uses coarse length-bin matching. For each bin, it compares $r=0$ versus $r\\ne0$, then averages the bin differences using the $r=0$ event distribution.')
    lines.append('')
    best = decomp[(decomp['gap_length_variable'] == 'g2') & (decomp['bin_width'] == 12)]
    if not best.empty:
        cols = ['q','raw_delta_r0_minus_nonzero','within_bin_delta_r0_minus_nonzero','composition_component_approx','matched_r0_coverage','n_bins_matched']
        lines.append('Using $g_2$ bins of width 12:')
        lines.append('')
        lines.append(best[cols].to_markdown(index=False, floatfmt='.6f'))
    lines.append('')
    lines.append('Interpretation rule: if the within-bin delta is close to zero while the raw delta is positive, the apparent diagonal branch penalty is mostly a gap-length composition effect. If the within-bin delta remains large and positive across bin widths, the effect survives a basic length control.')
    lines.append('')
    lines.append('## Telemetry')
    lines.append('')
    lines.append(f'- merged pair rows: {telemetry.get("merged_rows")}')
    lines.append(f'- missing omega rows: {telemetry.get("missing_omega_rows")}')
    lines.append(f'- total weighted events: {telemetry.get("total_events")}')
    outdir.joinpath('holt_omega_length_confound_interpretation.md').write_text('\n'.join(lines), encoding='utf-8')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Audit whether Omega_path residue effects are confounded by gap length.')
    parser.add_argument('--config', type=Path, default=None, help='Dataset config JSON, used to resolve pair-count block files.')
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--blocks', default='1-10')
    parser.add_argument('--pair-count-glob', default=None, help='Optional glob pattern for pair-count CSVs. Overrides config resolution. Example: data/.../blocks/parent_wide_B*.csv.gz')
    parser.add_argument('--path-cache-file', type=Path, required=True)
    parser.add_argument('--mods', default='3,5,7,11,13')
    parser.add_argument('--bin-widths', default='12,24,60')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)

    t0 = time.time()
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    mods = [int(x.strip()) for x in args.mods.split(',') if x.strip()]
    widths = [int(x.strip()) for x in args.bin_widths.split(',') if x.strip()]

    if args.pair_count_glob:
        files = [(i + 1, Path(p)) for i, p in enumerate(sorted(map(str, Path('.').glob(args.pair_count_glob))))]
        if not files:
            # If glob is absolute, pathlib Path('.').glob will not work.
            import glob
            files = [(i + 1, Path(p)) for i, p in enumerate(sorted(glob.glob(args.pair_count_glob)))]
        if not files:
            raise FileNotFoundError(f'No files matched --pair-count-glob {args.pair_count_glob}')
    else:
        config = load_config(args.config)
        blocks = parse_blocks(args.blocks)
        files = resolve_block_files(config, args.root, blocks)

    pairs = load_pair_counts(files)
    cache = load_path_cache(args.path_cache_file)
    merged = pairs.merge(cache, on=['g1', 'g2'], how='left')
    missing = int(merged['omega_path'].isna().sum())
    merged = merged.dropna(subset=['omega_path']).copy()
    if merged.empty:
        raise ValueError('No pair-count rows matched Omega_path cache')

    raw = raw_by_residue(merged, mods)
    raw.to_csv(outdir / 'holt_omega_raw_by_residue.csv', index=False)

    all_decomp = []
    all_detail = []
    for gap_col in ['g2', 'Gmax']:
        decomp, detail = length_bin_decomposition(merged, mods, widths, gap_col=gap_col)
        all_decomp.append(decomp)
        all_detail.append(detail)
    decomp_df = pd.concat(all_decomp, ignore_index=True)
    detail_df = pd.concat(all_detail, ignore_index=True)
    decomp_df.to_csv(outdir / 'holt_omega_length_confound_decomposition.csv', index=False)
    detail_df.to_csv(outdir / 'holt_omega_length_bin_detail.csv', index=False)

    # Exact gap summary for transparency (not used as matched control because exact g determines residue).
    exact_rows = []
    for q in mods:
        tmp = merged.copy()
        tmp['gap_residue_r'] = tmp['g2'] % q
        for (g2, r), grp in tmp.groupby(['g2', 'gap_residue_r']):
            exact_rows.append({
                'q': q,
                'g2': int(g2),
                'gap_residue_r': int(r),
                'count': float(grp['H'].sum()),
                'mean_omega_path': weighted_mean(grp['omega_path'], grp['H']),
                'mean_g1': weighted_mean(grp['g1'], grp['H']),
                'mean_Gmax': weighted_mean(grp['Gmax'], grp['H']),
            })
    pd.DataFrame(exact_rows).to_csv(outdir / 'holt_omega_by_exact_gap.csv', index=False)

    telemetry = {
        'script': 'holt_omega_length_confound_audit.py',
        'argv': sys.argv,
        'platform': platform.platform(),
        'python': sys.version,
        'cpu_count': os.cpu_count(),
        'blocks': args.blocks,
        'mods': mods,
        'bin_widths': widths,
        'pair_count_files': [str(p) for _, p in files],
        'path_cache_file': str(args.path_cache_file),
        'path_cache_rows': int(len(cache)),
        'pair_rows_loaded': int(len(pairs)),
        'merged_rows': int(len(merged)),
        'missing_omega_rows': int(missing),
        'total_events': float(merged['H'].sum()),
        'elapsed_seconds': time.time() - t0,
        'outputs': [
            'holt_omega_raw_by_residue.csv',
            'holt_omega_length_confound_decomposition.csv',
            'holt_omega_length_bin_detail.csv',
            'holt_omega_by_exact_gap.csv',
            'holt_omega_length_confound_interpretation.md',
            'holt_omega_length_confound_telemetry.json',
        ],
    }
    write_interpretation(outdir, raw, decomp_df, telemetry)
    (outdir / 'holt_omega_length_confound_telemetry.json').write_text(json.dumps(telemetry, indent=2), encoding='utf-8')

    print(f'[OK] Wrote length-confound audit outputs to {outdir}')
    print(f'[OK] missing omega rows: {missing}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
