# Holt / CHL2 Omega length-confound audit

This supplemental audit checks whether the raw observation

$$\Omega_Y^{\rm path}(g_2\equiv0\pmod q) > \Omega_Y^{\rm path}(g_2\not\equiv0\pmod q)$$

is genuinely modular or primarily explained by the fact that different gap-residue classes contain different gap-length populations.

The audit is motivated by the observation that $\Omega_Y^{\rm path}(g_1,g_2;x)$ generally grows with the length of the candidate gap, because longer gaps contain more possible interior prime positions. Therefore, a raw residue comparison can be confounded by gap length.

## Command

```bash
python supplemental_audits/holt_followup/holt_omega_length_confound_audit.py \
  --path-cache-file outputs/chl2_main/chl2_path_exclusion_cache_Y47_logx25328436.csv.gz \
  --mods 3,5,7,11,13 \
  --output-dir supplemental_audits/holt_followup/outputs/holt_omega_length_confound
```

If the path cache does not contain empirical counts, pass a pair-count CSV with `--pair-counts-csv`.

## Outputs

- `holt_omega_length_by_residue.csv`
- `holt_omega_length_confound_summary.csv`
- `holt_omega_by_exact_gap.csv`
- `holt_omega_length_confound_interpretation.md`
- `holt_omega_length_confound_telemetry.json`

## Interpretation

A raw positive contrast for $r=0$ means the diagonal residue branch has higher average $\Omega_Y^{\rm path}$. If the contrast disappears after normalizing by interior slots or after length-control regression, then the effect is primarily a gap-length composition effect rather than a modular penalty.

This audit is diagnostic only. It does not implement Holt's exact $G(p\#)$ recurrence and should be treated as a compatibility check.
