# Paper audits

This directory contains the reproduction scripts for the CHL2 whitepaper tables.
The scripts read an extensible JSON configuration produced by `data_generation/`
or adapted by the user.

## Main conditional likelihood audit

```bash
python paper_audits/chl2_consecutive_exclusion_audit.py \
  --config data/ds1_1e11_w2e9_g2400/config.generated.json \
  --root . \
  --blocks 1-10 \
  --path-exclusion \
  --workers 0 \
  --parallel-mode auto \
  --output-dir outputs/chl2_main
```

This writes `chl2_conditional_summary.csv`, `chl2_pairwise_gains.csv`,
`chl2_memory_irreducibility.csv`, `chl2_runtime_telemetry.json`, and related
diagnostics.

## Truncation-horizon Y-sweep

```bash
python paper_audits/chl2_y_sweep.py \
  --script paper_audits/chl2_consecutive_exclusion_audit.py \
  --config data/ds1_1e11_w2e9_g2400/config.generated.json \
  --root . \
  --blocks 1-10 \
  --y-values 31,47,61,73 \
  --path-exclusion \
  --workers 0 \
  --parallel-mode auto \
  --output-dir outputs/chl2_y_sweep
```

The sweep writes `chl2_y_sweep_telemetry.json` in the sweep output directory.

## Oliver--Soundararajan prime-residue diagnostic

The main audit can run the prime-residue OS diagnostic directly:

```bash
python paper_audits/chl2_consecutive_exclusion_audit.py \
  --config data/ds1_1e11_w2e9_g2400/config.generated.json \
  --root . \
  --blocks 1-10 \
  --path-exclusion \
  --reuse-path-cache \
  --prime-csv AUTO \
  --os-prime-mods 3,5,7 \
  --output-dir outputs/chl2_os
```

To recompute Pearson chi-square values from an existing transition matrix:

```bash
python paper_audits/os_prime_residue_diagnostic.py \
  --matrix-csv outputs/chl2_os/chl2_os_prime_residue_transition_by_mod.csv \
  --summary-csv outputs/chl2_os/chl2_os_prime_residue_summary.csv \
  --out outputs/chl2_os/chl2_os_prime_residue_chisquare.csv
```

The standalone diagnostic writes `os_prime_residue_diagnostic_telemetry.json` by
default next to the output CSV unless `--telemetry-json` is supplied.
