# Holt follow-up audits

This folder contains supplemental audits created after the public `v1.8.1` release of CHL-prime-gaps, in response to Fred Holt's comments on gap-population models, the cycles $G(p\#)$, and survival intervals

$$\Delta H(p)=[p^2,\operatorname{nextprime}(p)^2].$$

These scripts are **not** part of the main v1.8.1 reproduction pipeline. They are intended to make follow-up checks reproducible and to prepare a possible v2 benchmark.

## LOW_PRIME_BOUNDARY_AUDIT

`low_prime_boundary_audit.py` checks whether the old-direct modular diagnostic error decreases with the valid-edge multiplicity asymmetry

$$N_0(q)/N_r(q)=(q-1)/(q-2)$$

for prime moduli $q=3,5,7,11,13$.

It uses the final v1.8.1 outputs:

- `chl4d2_gap_population_by_residue.csv`
- `chl2_os_old_direct_vs_oriented.csv`

and produces:

- `low_prime_boundary_summary.csv`
- `low_prime_boundary_correlations.csv`
- `low_prime_boundary_q3_p0_check.csv`
- `low_prime_boundary_interpretation.md`

This audit is a compatibility check, not a theorem. It does not identify Holt's sieve prime and the diagnostic modulus as the same object. It only tests whether the same low-prime boundary combinatorics appears as valid-edge multiplicity in the reduced-residue transition graph.

## HOLT_GAP_POPULATION_BASELINE_AUDIT

`holt_gap_population_baseline_audit.py` explores the bridge between Holt's survival-interval viewpoint and CHL2's local no-interior survival factor.

It partitions a chronological prime stream by survival intervals

$$\Delta H(s)=[s^2,\operatorname{nextprime}(s)^2]$$

where $s$ is a sieving prime, then measures empirical gap-residue populations inside those intervals. If a CHL2 path-cache is supplied, it also summarizes event-weighted averages of

$$\Omega_Y^{\rm path}$$

and

$$\exp(-\Omega_Y^{\rm path})$$

by survival interval and residue class.

Important limitation: this script does **not** implement Holt's exact cycle-recursion model for $G(p\#)$. It is a compatibility audit that organizes DS1 and CHL2 quantities in Holt survival-interval coordinates.

### Example command

```bash
python supplemental_audits/holt_followup/holt_gap_population_baseline_audit.py \
  --prime-csv data/ds1_1e11_w2e9_g2400/real_primes.csv.gz \
  --path-cache-file outputs/chl2_main/chl2_path_exclusion_cache_Y47_logx25328436.csv.gz \
  --mods 3,5,7,11,13 \
  --gmax 2400 \
  --chunksize 1000000 \
  --output-dir supplemental_audits/holt_followup/outputs/holt_gap_population_baseline
```

### Outputs

- `holt_survival_interval_summary.csv`
- `holt_gap_residue_population_by_interval.csv`
- `holt_gap_residue_population_summary.csv`
- `holt_boundary_gap_population_summary.csv`
- `holt_gap_population_baseline_interpretation.md`
- `holt_gap_population_baseline_telemetry.json`

### Interpretation

The valid-edge reference distribution used by this audit is

$$p_r^{\rm edge}=\frac{N_r(q)}{\sum_s N_s(q)}.$$

This is the edge-count component underlying the orientation lift. It is **not** Holt's full finite-stage population model.

When `--path-cache-file` is provided, the reported CHL2 quantities are local no-interior survival diagnostics. They should be compared conceptually, not identified, with Holt's population survival across $\Delta H(s)$.
