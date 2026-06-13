# CHL-prime-gaps

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20368548.svg)](https://doi.org/10.5281/zenodo.20368548)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**CHL-prime-gaps** contains the code and documentation for the CHL2 finite-window prime-gap model described in the whitepaper:

```text
docs/CHL2_conditional_hardy_littlewood_markov_whitepaper_v1_8.pdf
docs/CHL2_conditional_hardy_littlewood_markov_whitepaper_v1_8.tex
```

CHL2 is a finite conditional Hardy--Littlewood Markov kernel for consecutive prime gaps.  It starts from the conditional singular-series ratio for the triple

$$H_3(g_1,g_2)=\{0,g_1,g_1+g_2\},$$

adds a first-order no-interior-prime survival correction, and normalizes over a finite gap support.  The repository also implements an **orientation-lifted modular diagnostic**: if a gap model produces a gap-residue population

$$p_r=P(g\equiv r\pmod q),$$

then this population is lifted to a reduced-residue transition matrix by distributing $p_r$ over the valid directed edges $b\to b+r$ in the reduced-residue graph.  This diagnostic corrects the older direct modular projection and resolves the apparent $q=3$ anomaly discussed during the development of the project.

This repository is intended to be reproducible: the code can generate DS1-style prime-gap blocks, compute the CHL2 likelihood audits, run the truncation-horizon sweep, and regenerate the orientation-lifted modular diagnostics reported in the whitepaper.

## Repository layout

```text
CHL-prime-gaps/
  chl_kernel/          Pure mathematical kernel code: singular series, CHL models, residue-transfer utilities
  data_generation/     Prime and prime-gap block generation
  paper_audits/        Reproducibility scripts for the whitepaper tables and figures
  research_tools/      CHL2 Hit@K / candidate-prioritization utilities
  docs/                Whitepaper PDF/LaTeX and explanatory notes
  tests/               Smoke tests and mathematical utility tests
  configs/             Example JSON configurations
```

The main mathematical model lives in `chl_kernel/`.  Scripts in `paper_audits/` reproduce the whitepaper outputs; they are not imported by the kernel.

## Installation

A standard local setup is:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Run smoke tests:

```bash
python -m compileall .
python tests/test_kernel_smoke.py
python -m pytest -q
```

## DS1 data used in the whitepaper

The principal audit window, called **DS1**, is the finite prime-gap window

$$[10^{11},\;10^{11}+2\cdot10^9].$$

The DS1 likelihood audits use aggregated pair-count tables of consecutive gaps $(g_1,g_2)$.  The modular diagnostics additionally require the chronological prime stream over the same range.  The data are divided into ten chronological blocks, B01--B10, for block-level stability checks.

A generated DS1-style data directory is expected to look like:

```text
data/ds1_1e11_w2e9_g2400/
  config.generated.json
  real_primes.csv.gz
  blocks/
    parent_wide_B01.csv.gz
    ...
    parent_wide_B10.csv.gz
```

The exact file names above are computational artifacts.  Mathematically, the blocks are chronological pair-count blocks with counts $H(g_1,g_2)$.

To generate a quick test dataset:

```bash
python data_generation/generate_prime_gap_blocks.py \
  --quick-test \
  --output-dir data/quick_test
```

To generate a DS1-style window around $10^{11}$:

```bash
python data_generation/generate_prime_gap_blocks.py \
  --start 100000000000 \
  --end 102000000000 \
  --gmax 2400 \
  --num-blocks 10 \
  --workers 16 \
  --output-dir data/ds1_1e11_w2e9_g2400
```

The whitepaper uses finite support up to $g_{\max}=2400$.

## Mathematical strata and repository names

The whitepaper uses mathematical stratum names.  Some CSV outputs still use legacy variable-style names for compatibility with earlier audits.  The mapping is:

| Whitepaper stratum | Mathematical definition, with $G_{\max}=\max(g_1,g_2)$ | Legacy CSV value |
|---|---|---|
| $S_{\rm all}$ | all observed pairs | `ALL` |
| $S_{\rm dense}$ | $G_{\max}\le 58$ | `LOW_ONLY_LE58` |
| $S_{\rm trans}$ | $59\le G_{\max}\le 120$ | `MID_59_120` |
| $S_{121:240}$ | $121\le G_{\max}\le 240$ | `MID_121_240` |
| $S_{121:400}$ | $121\le G_{\max}\le 400$ | `MID_121_400` |
| $S_{>58}$ | $G_{\max}>58$ | `NO_58` |
| $S_{>120}$ | $G_{\max}>120$ | `NO_120` |
| $S_{>240}$ | $G_{\max}>240$ | `NO_240` |

The principal claims in the paper use all strata except $S_{>240}$, which is retained as a sparse stress diagnostic.

## Reproducing the main CHL2 audits

The commands below assume that DS1 data are in:

```text
data/ds1_1e11_w2e9_g2400/
```

### 1. Main conditional likelihood audit

This reproduces the main CHL2 likelihood tables, CHL2-vs-CHL1 gains, memory-irreducibility results, and the path-exclusion cache.

```bash
python paper_audits/chl2_consecutive_exclusion_audit.py \
  --config data/ds1_1e11_w2e9_g2400/config.generated.json \
  --root . \
  --blocks 1-10 \
  --path-exclusion \
  --workers 0 \
  --parallel-mode auto \
  --path-chunk-size 0 \
  --path-target-tasks-per-worker 6 \
  --output-dir outputs/chl2_main
```

Main outputs:

```text
outputs/chl2_main/chl2_conditional_summary.csv
outputs/chl2_main/chl2_pairwise_gains.csv
outputs/chl2_main/chl2_memory_irreducibility.csv
outputs/chl2_main/chl2_exclusion_diagnostics.csv
outputs/chl2_main/chl2_path_exclusion_cache_Y47_logx25328436.csv.gz
outputs/chl2_main/chl2_runtime_telemetry.json
```

### 2. Truncation-horizon $Y$-sweep

This reproduces the stability sweep over $Y\in\{31,47,61,73\}$.

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

Main outputs:

```text
outputs/chl2_y_sweep/chl2_y_sweep_summary.csv
outputs/chl2_y_sweep/chl2_y_sweep_gains.csv
outputs/chl2_y_sweep/chl2_y_sweep_stability.csv
outputs/chl2_y_sweep/chl2_y_sweep_memory_irreducibility.csv
outputs/chl2_y_sweep/chl2_y_sweep_telemetry.json
```

### 3. Old direct Oliver--Soundararajan-style diagnostic

The old direct diagnostic is retained for provenance and for comparison against the orientation-lifted replacement.  It is not the preferred modular diagnostic for final claims.

```bash
python paper_audits/chl2_consecutive_exclusion_audit.py \
  --config data/ds1_1e11_w2e9_g2400/config.generated.json \
  --root . \
  --blocks 1-10 \
  --path-exclusion \
  --workers 0 \
  --parallel-mode auto \
  --path-chunk-size 0 \
  --prime-csv AUTO \
  --os-prime-mods 3,5,7,11,13 \
  --output-dir outputs/chl2_os
```

Main outputs:

```text
outputs/chl2_os/chl2_os_prime_residue_summary.csv
outputs/chl2_os/chl2_os_prime_residue_transition_by_mod.csv
outputs/chl2_os/chl2_runtime_telemetry.json
```

If only the Pearson chi-square values need to be recomputed from an existing transition matrix:

```bash
python paper_audits/os_prime_residue_diagnostic.py \
  --matrix-csv outputs/chl2_os/chl2_os_prime_residue_transition_by_mod.csv \
  --summary-csv outputs/chl2_os/chl2_os_prime_residue_summary.csv \
  --out outputs/chl2_os/chl2_os_prime_residue_chisquare.csv
```

## Reproducing the orientation-lifted modular diagnostic

The orientation-lifted diagnostic is a multi-step audit.  It first constructs empirical and CHL2 modular transition matrices, then constructs the valid-edge orientation lift, and finally produces the replacement diagnostic table.

### 4. Residual modular transfer matrices

This step generates the empirical matrices and the old direct CHL2 matrices used by later orientation-lift audits.

```bash
python paper_audits/chl4_modular_transfer_residual_audit.py \
  --mode from-prime-csv \
  --config data/ds1_1e11_w2e9_g2400/config.generated.json \
  --root . \
  --blocks 1-10 \
  --block-boundary-mode parent-wide \
  --os-csv outputs/chl2_os/chl2_os_prime_residue_transition_by_mod.csv \
  --prime-csv data/ds1_1e11_w2e9_g2400/real_primes.csv.gz \
  --mods 3,5,7,11,13 \
  --workers 0 \
  --chunksize 1000000 \
  --drop-partial-blocks \
  --output-dir outputs/chl4_residual_blocks_aligned
```

Required outputs for later steps:

```text
outputs/chl4_residual_blocks_aligned/chl4_transfer_empirical_matrices.csv
outputs/chl4_residual_blocks_aligned/chl4_transfer_chl2_matrices.csv
outputs/chl4_residual_blocks_aligned/chl4_character_spectrum.csv
outputs/chl4_residual_blocks_aligned/chl4_q3_diagonal_scalar.csv
```

### 5. Gap-population bias audit

This step verifies that CHL2 reproduces gap-residue populations and prepares population diagnostics used to understand the lift.

```bash
python paper_audits/chl4d2_gap_population_bias_audit.py \
  --config data/ds1_1e11_w2e9_g2400/config.generated.json \
  --root . \
  --blocks 1-10 \
  --Y 47 \
  --path-cache-file outputs/chl2_main/chl2_path_exclusion_cache_Y47_logx25328436.csv.gz \
  --output-dir outputs/chl4d2_gap_population_bias
```

Main outputs:

```text
outputs/chl4d2_gap_population_bias/chl4d2_gap_population_by_residue.csv
outputs/chl4d2_gap_population_bias/chl4d2_scale_wave_summary.csv
outputs/chl4d2_gap_population_by_filter.csv
outputs/chl4d2_gap_population_bias.csv
```

### 6. Orientation-lift generalization

This step constructs the valid-edge orientation lift for $q=3,5,7,11,13$.

```bash
python paper_audits/chl4d4_orientation_lift_generalization_audit.py \
  --config data/ds1_1e11_w2e9_g2400/config.generated.json \
  --root . \
  --blocks 1-10 \
  --mods 3,5,7,11,13 \
  --Y 47 \
  --path-cache-file outputs/chl2_main/chl2_path_exclusion_cache_Y47_logx25328436.csv.gz \
  --empirical-matrix-csv outputs/chl4_residual_blocks_aligned/chl4_transfer_empirical_matrices.csv \
  --model-matrix-csv outputs/chl4_residual_blocks_aligned/chl4_transfer_chl2_matrices.csv \
  --output-dir outputs/chl4d4_orientation_lift_generalization
```

Required output for the final replacement diagnostic:

```text
outputs/chl4d4_orientation_lift_generalization/chl4d4_orientation_lift_matrices.csv
```

### 7. Orientation-lifted OS replacement

This step produces the modular diagnostic used in the final whitepaper.

```bash
python paper_audits/chl4d5_orientation_lift_os_replacement.py \
  --mode from-d4-lift \
  --d4-lift-csv outputs/chl4d4_orientation_lift_generalization/chl4d4_orientation_lift_matrices.csv \
  --empirical-matrix-csv outputs/chl4_residual_blocks_aligned/chl4_transfer_empirical_matrices.csv \
  --old-model-matrix-csv outputs/chl4_residual_blocks_aligned/chl4_transfer_chl2_matrices.csv \
  --mods 3,5,7,11,13 \
  --output-dir outputs/chl4d5_orientation_lift_os_replacement
```

Main outputs:

```text
outputs/chl4d5_orientation_lift_os_replacement/chl2_os_oriented_prime_residue_summary.csv
outputs/chl4d5_orientation_lift_os_replacement/chl2_os_oriented_prime_residue_transition_by_mod.csv
outputs/chl4d5_orientation_lift_os_replacement/chl2_os_oriented_model_matrices.csv
outputs/chl4d5_orientation_lift_os_replacement/chl2_os_old_direct_vs_oriented.csv
```

**Whitepaper table source.**  The orientation-lift diagnostic table in the whitepaper, including the values where the naive direct diagnostic has diagonal probabilities such as `0.100000` for $q=11$ and `0.083333` for $q=13$, is reproduced from:

```text
outputs/chl4d5_orientation_lift_os_replacement/chl2_os_old_direct_vs_oriented.csv
```

Use the `block == ALL` rows and the columns:

```text
q
emp_diag
old_diag_model
oriented_diag_model
old_kl
oriented_kl
```

The `old_diag_model` values are read from the old direct CHL2 matrices generated in step 4:

```text
outputs/chl4_residual_blocks_aligned/chl4_transfer_chl2_matrices.csv
```

The `oriented_diag_model` values are read from the orientation-lift matrices generated in step 6:

```text
outputs/chl4d4_orientation_lift_generalization/chl4d4_orientation_lift_matrices.csv
```

The former proof-of-concept `chl4d3_orientation_lift_audit.py` is not required in the main reproduction chain.  It was an explanatory $q=3$ audit; the paper's final multi-modulus orientation-lift table is generated by step 7.

## Mapping whitepaper items to scripts

| Whitepaper item | Mathematical object | Script | Principal output |
|---|---|---|---|
| Main likelihood table | CHL2 vs CHL1 and baselines | `paper_audits/chl2_consecutive_exclusion_audit.py` | `outputs/chl2_main/chl2_conditional_summary.csv` |
| CHL2-vs-CHL1 gains | Conditional log-likelihood gain | `paper_audits/chl2_consecutive_exclusion_audit.py` | `outputs/chl2_main/chl2_pairwise_gains.csv` |
| Memory irreducibility | CHL2 vs order-zero baselines | `paper_audits/chl2_consecutive_exclusion_audit.py` | `outputs/chl2_main/chl2_memory_irreducibility.csv` |
| $Y$-sweep table and heatmap | Stability over $Y\in\{31,47,61,73\}$ | `paper_audits/chl2_y_sweep.py` | `outputs/chl2_y_sweep/chl2_y_sweep_stability.csv` |
| Old direct modular diagnostic | Naive OS-style transition matrix | `paper_audits/chl2_consecutive_exclusion_audit.py` | `outputs/chl2_os/chl2_os_prime_residue_transition_by_mod.csv` |
| Residual modular transfer audit | Empirical and CHL2 transition matrices | `paper_audits/chl4_modular_transfer_residual_audit.py` | `outputs/chl4_residual_blocks_aligned/chl4_transfer_empirical_matrices.csv` and `chl4_transfer_chl2_matrices.csv` |
| Gap-population audit | $P(g\bmod q)$ by filter/block | `paper_audits/chl4d2_gap_population_bias_audit.py` | `outputs/chl4d2_gap_population_bias/chl4d2_gap_population_by_residue.csv` |
| Orientation-lift generalization | Valid-edge lift for $q=3,5,7,11,13$ | `paper_audits/chl4d4_orientation_lift_generalization_audit.py` | `outputs/chl4d4_orientation_lift_generalization/chl4d4_orientation_lift_matrices.csv` |
| Orientation-lifted OS replacement | Final modular diagnostic | `paper_audits/chl4d5_orientation_lift_os_replacement.py` | `outputs/chl4d5_orientation_lift_os_replacement/chl2_os_old_direct_vs_oriented.csv` |
| Hit@K oracle | Candidate prioritization by CHL2 cost | `research_tools/chl2_hitk_oracle.py` | user-chosen CSV output |

## Runtime telemetry

Most scripts write a `*_runtime_telemetry.json` file in their output directory.  These files are not part of the mathematical model; they are reproducibility records.  They typically include:

```text
elapsed_seconds
command-line arguments
Python version and platform
CPU count / worker count
input paths and output paths
number of rows written
model list / modulus list / block list
cache usage flags
```

Example telemetry files:

```text
outputs/chl2_main/chl2_runtime_telemetry.json
outputs/chl2_y_sweep/chl2_y_sweep_telemetry.json
outputs/chl4_residual_blocks_aligned/chl4_runtime_telemetry.json
outputs/chl4d5_orientation_lift_os_replacement/chl4d5_runtime_telemetry.json
```

Runtime depends on hardware, compression format, and whether caches are reused.  The telemetry exists so that a reader can distinguish mathematical output from hardware-dependent execution time.

## Research tool: CHL2 Hit@K oracle

The CHL2 cost function

$$\mathcal C_Y(g_2\mid g_1;x)=-\log R_Y(g_2\mid g_1)+\Omega_Y^{\rm path}(g_1,g_2;x)$$

can be used as a parameter-free candidate-prioritization score.  It is not a primality test; it ranks candidates before classical tests are applied.

Example:

```bash
python research_tools/chl2_hitk_oracle.py \
  --p-current 1000003 \
  --g-prev 6 \
  --Y 47 \
  --gmax 300 \
  --top-k 20 \
  --output-csv outputs/hitk.csv
```

## Scientific status

The repository supports the following claims made in the v1.8 whitepaper:

1. CHL2 improves the CHL1 ratio-only kernel on DS1.
2. The CHL2-vs-CHL1 gain is stable over the tested truncation horizons $Y=31,47,61,73$.
3. CHL2 outperforms zero-memory Cramér--Granville style baselines in conditional likelihood.
4. The apparent $q=3$ failure of the old direct modular diagnostic is removed by the valid-edge orientation lift.
5. The orientation-lifted modular diagnostic improves the agreement for $q=3,5,7,11,13$.

The repository does not claim an asymptotic theorem for primes; the audits are finite-window computational evaluations of explicitly defined kernels and diagnostics.

## License and citation

The project is released under the MIT license.  Use `CITATION.cff` for citation metadata and `docs/` for the current whitepaper PDF and LaTeX source.
