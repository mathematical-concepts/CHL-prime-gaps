# CHL-prime-gaps

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20368548.svg)](https://doi.org/10.5281/zenodo.20368548)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**CHL-prime-gaps** contains the code and documentation for finite-window conditional Hardy--Littlewood models of consecutive prime gaps. The production kernel is

$$\mathrm{CHL2}=R_Y(g_2\mid g_1)\,\exp\!\left[-\Omega_Y^{\mathrm{path}}(g_1,g_2;x)\right].$$

where

$$R_Y(g_2\mid g_1)=\frac{\mathfrak{S}_Y(\{0,g_1,g_1+g_2\})}{\mathfrak{S}_Y(\{0,g_1\})}.$$

The repository also implements an **orientation-lifted modular diagnostic**. If a gap model produces

$$p_r=P(g\equiv r\pmod q),$$

then the orientation lift distributes $p_r$ over the valid directed edges $b\to b+r$ in the reduced-residue graph before row normalization. This corrects the previous-gap-conditioned direct projection and removes the apparent $q=3$ wrong-sign discrepancy.

The versioned v2.0.0 manuscript source is `docs/CHL2_conditional_hardy_littlewood_markov_whitepaper_v2_0_0_rc1.tex`. The release PDF is generated locally under `outputs/v2_release_paper/` and attached to the published release; it is not committed with scientific outputs. The archived v1.8 source and PDF remain under `docs/` for historical comparison. Scientific CSVs and DS1 are generated locally and are not committed to Git.

## Reproducibility policy

The repository contains the method, not precomputed DS1 or release outputs:

```text
versioned:
  chl_kernel/
  data_generation/
  paper_audits/
  research_tools/
  tests/
  configs/
  docs/ source and asset builders

local and ignored:
  data/
  outputs/
  reference_outputs/
  work/
  tmp/
  .local_artifacts/
```

The release chain is:

```text
scripts
→ locally generated data and audit outputs
→ generated LaTeX tables and figures
→ TeX
→ PDF
```

A clean checkout must be able to generate every reported table and figure from the public scripts. No release claim should depend on a versioned scientific CSV.

## Repository layout

```text
CHL-prime-gaps/
  chl_kernel/          Pure mathematical kernel code
  data_generation/     Prime and prime-gap block generation
  paper_audits/        Public reproducibility scripts
  research_tools/      CHL2 Hit@K / candidate-prioritization utilities
  docs/                Whitepaper sources and paper-asset builders
  tests/               Kernel, audit, and release-builder tests
  configs/             Example JSON configurations
```

The main mathematical model lives in `chl_kernel/`. Scripts in `paper_audits/` reproduce the whitepaper outputs; they are not imported by the kernel.

## Installation

A standard local setup is:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Matplotlib is installed by both `requirements.txt` and `pyproject.toml`. For the release build, use the frozen `uv.lock` under Python 3.11 or later; that environment resolves Matplotlib `3.11.1`, which is required for byte-identical PDF/SVG figure hashes.

Run the tests:

```bash
python -m compileall .
python -m pytest -q
```

## DS1 data used in the whitepaper

The principal audit window, **DS1**, is

$$[10^{11},\;10^{11}+2\cdot10^9].$$

The DS1 likelihood audits use pair-count tables of consecutive gaps $(g_1,g_2)$. The modular audits additionally require the chronological prime stream over the same range. The data are divided into ten chronological blocks, B01--B10.

A generated DS1 directory has the form:

```text
data/ds1_1e11_w2e9_g2400/
  config.generated.json
  real_primes.csv.gz
  blocks/
    parent_wide_B01.csv.gz
    ...
    parent_wide_B10.csv.gz
```

Generate a quick test dataset:

```bash
python data_generation/generate_prime_gap_blocks.py \
  --quick-test \
  --output-dir data/quick_test
```

Generate DS1:

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

The likelihood audits use $G_{\max}=\max(g_1,g_2)$:

| Whitepaper stratum | Definition | CSV value |
|---|---|---|
| $S_{\rm all}$ | all observed pairs | `ALL` |
| $S_{\rm dense}$ | $G_{\max}\le 58$ | `LOW_ONLY_LE58` |
| $S_{\rm trans}$ | $59\le G_{\max}\le 120$ | `MID_59_120` |
| $S_{121:240}$ | $121\le G_{\max}\le 240$ | `MID_121_240` |
| $S_{121:400}$ | $121\le G_{\max}\le 400$ | `MID_121_400` |
| $S_{>58}$ | $G_{\max}>58$ | `NO_58` |
| $S_{>120}$ | $G_{\max}>120$ | `NO_120` |
| $S_{>240}$ | $G_{\max}>240$ | `NO_240` |

The principal claims exclude $S_{>240}$, which is retained as a sparse stress diagnostic.

The A05 scale-wave audit reuses the same legacy filter names but applies them to the **candidate gap $g_2$ alone**. Tables and figures generated by `docs/build_tables.py` and `docs/build_figures.py` label these filters explicitly as $g_2$ filters so that they are not confused with the likelihood strata.

## Canonical v2.0.0 output directories

```text
outputs/v2_release_chl2_main
outputs/v2_release_chl2_y_sweep
outputs/v2_release_chl2_os
outputs/v2_release_chl4d2_gap_population_bias
outputs/v2_release_chl4_residual_blocks_aligned
outputs/v2_release_chl4d5_orientation_lift_os_replacement
outputs/v2_release_chl2_factor_compatibility
outputs/v2_release_paper_assets
```

The commands below assume that DS1 is under `data/ds1_1e11_w2e9_g2400/`.

## A02 — Main conditional likelihood audit

This produces the main likelihood results, CHL2-vs-CHL1 gains, block metrics, memory-irreducibility diagnostics, and the reusable path-exclusion cache.

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
  --output-dir outputs/v2_release_chl2_main
```

Principal outputs:

```text
outputs/v2_release_chl2_main/chl2_metrics_by_block.csv
outputs/v2_release_chl2_main/chl2_conditional_summary.csv
outputs/v2_release_chl2_main/chl2_pairwise_gains.csv
outputs/v2_release_chl2_main/chl2_memory_irreducibility.csv
outputs/v2_release_chl2_main/chl2_exclusion_diagnostics.csv
outputs/v2_release_chl2_main/chl2_path_exclusion_cache_Y47_logx25328436.csv.gz
outputs/v2_release_chl2_main/chl2_runtime_telemetry.json
```

A03 block stability is derived directly from `chl2_metrics_by_block.csv`; no second kernel execution is required.

## A04 — Truncation-horizon $Y$ sweep

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
  --path-chunk-size 0 \
  --output-dir outputs/v2_release_chl2_y_sweep
```

Principal outputs:

```text
outputs/v2_release_chl2_y_sweep/chl2_y_sweep_runs.csv
outputs/v2_release_chl2_y_sweep/chl2_y_sweep_summary.csv
outputs/v2_release_chl2_y_sweep/chl2_y_sweep_gains.csv
outputs/v2_release_chl2_y_sweep/chl2_y_sweep_stability.csv
outputs/v2_release_chl2_y_sweep/chl2_y_sweep_memory_irreducibility.csv
outputs/v2_release_chl2_y_sweep/chl2_y_sweep_telemetry.json
```

## A06a — Absolute-prime previous-gap-conditioned control

This is the reproducible direct modular control used in v2.0.0. It conditions the model prediction on the previous observed gap and compares it with absolute-prime residue transitions.

```bash
python paper_audits/chl2_consecutive_exclusion_audit.py \
  --config data/ds1_1e11_w2e9_g2400/config.generated.json \
  --root . \
  --blocks 1-10 \
  --path-exclusion \
  --workers 0 \
  --parallel-mode auto \
  --path-chunk-size 0 \
  --path-cache-file outputs/v2_release_chl2_main/chl2_path_exclusion_cache_Y47_logx25328436.csv.gz \
  --reuse-path-cache \
  --prime-csv AUTO \
  --os-prime-mods 3,5,7,11,13 \
  --os-model CHL2_path_excl_cond_eta \
  --os-residue-mode reduced \
  --os-max-transitions 0 \
  --os-prime-chunksize 1000000 \
  --output-dir outputs/v2_release_chl2_os
```

Principal outputs:

```text
outputs/v2_release_chl2_os/chl2_os_prime_residue_summary.csv
outputs/v2_release_chl2_os/chl2_os_prime_residue_transition_by_mod.csv
outputs/v2_release_chl2_os/chl2_config.json
outputs/v2_release_chl2_os/chl2_runtime_telemetry.json
```

### Historical Table 4 provenance — A06 closed

The reproducible v2.0.0 control above is named:

```text
absolute_prime_os_previous_gap_conditioned
```

It is distinct from the historical label:

```text
naive_table4_legacy
```

The provenance of the v1.8 Table 4 entries has now been reconstructed. The effective legacy object was a mixed implementation artifact:

```text
q=3,5,7:
  available previous-gap-conditioned direct OS matrices

q=11,13:
  moduli absent from the upstream OS CSV
  -> zero model rows emitted by the residual audit
  -> downstream uniform-row fallback
  -> diagonal 1/(q-1), namely 0.100000 and 0.083333...
```

The reconstruction reproduces both the printed diagonal probabilities and the large legacy KL values for $q=11,13$. The exact historical shell command was not tracked, but the missing-modulus/zero-row fallback and its numerical effect are determined by the v1.8 code path. This object is therefore retained only for historical explanation and is **withdrawn as a scientific baseline**.

Version 2.0.0 uses the fully specified five-modulus control `absolute_prime_os_previous_gap_conditioned`. It does not restore the legacy fallback values. The separate Lab v2 control `naive_row_conditioned` is also a different operator. See [`docs/NAIVE_TABLE4_LEGACY_PROVENANCE.md`](docs/NAIVE_TABLE4_LEGACY_PROVENANCE.md).

## A05 — Gap-population and scale-wave audit

```bash
python paper_audits/chl4d2_gap_population_bias_audit.py \
  --config data/ds1_1e11_w2e9_g2400/config.generated.json \
  --root . \
  --blocks 1-10 \
  --Y 47 \
  --path-cache-file outputs/v2_release_chl2_main/chl2_path_exclusion_cache_Y47_logx25328436.csv.gz \
  --output-dir outputs/v2_release_chl4d2_gap_population_bias
```

Principal outputs:

```text
outputs/v2_release_chl4d2_gap_population_bias/chl4d2_gap_population_by_filter.csv
outputs/v2_release_chl4d2_gap_population_bias/chl4d2_gap_population_by_residue.csv
outputs/v2_release_chl4d2_gap_population_bias/chl4d2_gap_population_block_stability.csv
outputs/v2_release_chl4d2_gap_population_bias/chl4d2_scale_wave_summary.csv
outputs/v2_release_chl4d2_gap_population_bias/chl4d2_offdiag_symmetry.csv
outputs/v2_release_chl4d2_gap_population_bias/chl4d2_runtime_telemetry.json
```

## A07 — Residual modular transfer matrices aligned to DS1 blocks

```bash
python paper_audits/chl4_modular_transfer_residual_audit.py \
  --mode from-prime-csv \
  --os-csv outputs/v2_release_chl2_os/chl2_os_prime_residue_transition_by_mod.csv \
  --model CHL2_path_excl_cond_eta \
  --mods 3,5,7,11,13 \
  --prime-csv data/ds1_1e11_w2e9_g2400/real_primes.csv.gz \
  --config data/ds1_1e11_w2e9_g2400/config.generated.json \
  --root . \
  --blocks 1-10 \
  --block-boundary-mode parent-wide \
  --block-count-col H \
  --drop-partial-blocks \
  --skip-first-transition \
  --chunksize 1000000 \
  --workers 0 \
  --max-pending-factor 2 \
  --hash-inputs \
  --output-dir outputs/v2_release_chl4_residual_blocks_aligned
```

Principal outputs:

```text
outputs/v2_release_chl4_residual_blocks_aligned/chl4_transfer_empirical_matrices.csv
outputs/v2_release_chl4_residual_blocks_aligned/chl4_transfer_chl2_matrices.csv
outputs/v2_release_chl4_residual_blocks_aligned/chl4_transfer_residual_logratio.csv
outputs/v2_release_chl4_residual_blocks_aligned/chl4_q3_diagonal_scalar.csv
outputs/v2_release_chl4_residual_blocks_aligned/chl4_character_spectrum.csv
outputs/v2_release_chl4_residual_blocks_aligned/chl4_block_stability.csv
outputs/v2_release_chl4_residual_blocks_aligned/chl4_config.json
outputs/v2_release_chl4_residual_blocks_aligned/chl4_runtime_telemetry.json
```

The parent-wide alignment gate expects one omitted endpoint in B01 and one in B10, with no shift of the nine internal boundaries.

A07 also enforces a strict upstream model-support gate. Every requested modulus must be present in the selected OS model, every reduced-residue cell must exist exactly once, probabilities must be finite and non-negative, and every source row must have positive mass. The configuration and telemetry record `requested_mods`, `available_mods`, `missing_requested_mods`, `zero_sum_model_rows`, and `model_support_gate_pass`. Missing moduli can no longer be emitted as zero matrices.

## A08 — Orientation-lifted modular replacement

The canonical v2.0.0 route recomputes the gap-residue population from the DS1 blocks and compares the orientation lift with the reproducible A06a control.

```bash
python paper_audits/chl4d5_orientation_lift_os_replacement.py \
  --mode from-blocks \
  --empirical-matrix-csv outputs/v2_release_chl4_residual_blocks_aligned/chl4_transfer_empirical_matrices.csv \
  --old-model-matrix-csv outputs/v2_release_chl4_residual_blocks_aligned/chl4_transfer_chl2_matrices.csv \
  --old-model-label absolute_prime_os_previous_gap_conditioned \
  --oriented-model-label orientation_lift_chl2_gap_population \
  --config data/ds1_1e11_w2e9_g2400/config.generated.json \
  --root . \
  --blocks 1-10 \
  --mods 3,5,7,11,13 \
  --Y 47 \
  --log-x 25.328436022934504 \
  --path-cache-file outputs/v2_release_chl2_main/chl2_path_exclusion_cache_Y47_logx25328436.csv.gz \
  --hash-inputs \
  --output-dir outputs/v2_release_chl4d5_orientation_lift_os_replacement
```

Principal outputs:

```text
outputs/v2_release_chl4d5_orientation_lift_os_replacement/chl2_os_oriented_prime_residue_transition_by_mod.csv
outputs/v2_release_chl4d5_orientation_lift_os_replacement/chl2_os_oriented_prime_residue_summary.csv
outputs/v2_release_chl4d5_orientation_lift_os_replacement/chl2_os_old_direct_vs_oriented.csv
outputs/v2_release_chl4d5_orientation_lift_os_replacement/chl2_os_oriented_gap_residue_populations.csv
outputs/v2_release_chl4d5_orientation_lift_os_replacement/chl2_os_oriented_model_matrices.csv
outputs/v2_release_chl4d5_orientation_lift_os_replacement/chl4d5_config.json
outputs/v2_release_chl4d5_orientation_lift_os_replacement/chl4d5_runtime_telemetry.json
```

The whitepaper modular table is generated from the `block == ALL` rows of:

```text
chl2_os_old_direct_vs_oriented.csv
```

with columns:

```text
q
emp_diag
old_diag_model
oriented_diag_model
old_kl
oriented_kl
```

The label `old_diag_model` refers to the reproducible previous-gap-conditioned control, not to `naive_table4_legacy`.

A08 treats `--old-model-matrix-csv` as a scientific input. Complete reduced-residue support, finite non-negative probabilities, normalized rows, and strictly positive row mass are mandatory. A zero-sum imported row is an error and is never silently converted to a uniform transition row. The generated JSON records the strict support gate and the closed historical-provenance status.

### Optional D4 parity route

`chl4d4_orientation_lift_generalization_audit.py` remains available as an independent multi-modulus construction. Its matrices can be passed to D5 with `--mode from-d4-lift`. This is a parity/cross-check route, not a required step in the canonical v2.0.0 chain.

## A10 — Public H/R/P factor-compatibility audit

The eight vertices share the same conditional support and terminal mean anchor:

| $H$ | $R$ | $P$ | Model |
|---:|---:|---:|---|
| 0 | 0 | 0 | `ROW_MASK_ETA` |
| 1 | 0 | 0 | `H2_COND_ETA` |
| 0 | 1 | 0 | `CHL1` |
| 0 | 0 | 1 | `PATH_ONLY_ETA` |
| 1 | 1 | 0 | `CG_MARKOV_ETA` |
| 1 | 0 | 1 | `CG_PATH_ETA` |
| 0 | 1 | 1 | `CHL2` |
| 1 | 1 | 1 | `CG_MARKOV_PATH_ETA` |

Run:

```bash
python paper_audits/chl2_factor_compatibility_audit.py \
  --config data/ds1_1e11_w2e9_g2400/config.generated.json \
  --root . \
  --blocks 1-10 \
  --filters ALL,LOW_ONLY_LE58,MID_59_120,MID_121_240,MID_121_400,NO_58,NO_120,NO_240 \
  --Y 47 \
  --log-x 25.328436022934504 \
  --path-cache-file outputs/v2_release_chl2_main/chl2_path_exclusion_cache_Y47_logx25328436.csv.gz \
  --workers 0 \
  --anchor-tolerance 1e-10 \
  --hash-inputs \
  --output-dir outputs/v2_release_chl2_factor_compatibility
```

Principal outputs:

```text
outputs/v2_release_chl2_factor_compatibility/chl2_factor_model_metrics_by_block.csv
outputs/v2_release_chl2_factor_compatibility/chl2_factor_model_summary.csv
outputs/v2_release_chl2_factor_compatibility/chl2_factorial_effects_by_block.csv
outputs/v2_release_chl2_factor_compatibility/chl2_factorial_effects_summary.csv
outputs/v2_release_chl2_factor_compatibility/chl2_factor_context_effects_by_block.csv
outputs/v2_release_chl2_factor_compatibility/chl2_factor_context_effects_summary.csv
outputs/v2_release_chl2_factor_compatibility/chl2_factor_support_audit.csv
outputs/v2_release_chl2_factor_compatibility/chl2_factor_missing_cells.csv
outputs/v2_release_chl2_factor_compatibility/chl2_factor_compatibility_config.json
outputs/v2_release_chl2_factor_compatibility/chl2_factor_compatibility_telemetry.json
```

The release gate requires a complete $2^3$ cube, one support hash and one target mean per block/filter cell, strict terminal anchoring, finite metrics, and complete factorial/context effects.

## Generating LaTeX tables

After A02, A04, A05, A07, A08, and A10 are present under the canonical directories:

```bash
python docs/build_tables.py \
  --outputs-root outputs \
  --output-dir outputs/v2_release_paper_assets/tables \
  --strict
```

Generated fragments include:

```text
table_main_likelihood.tex
table_chl2_chl1_block_stability.tex
table_y_sweep.tex
table_memory_irreducibility.tex
table_scale_wave.tex
table_modular_orientation_lift.tex
table_chl4_q3_residual.tex
table_hrp_model_ranking.tex
table_hrp_factorial_effects.tex
table_hrp_context_effects.tex
tables_manifest.json
```

Every fragment records its source path and SHA-256 in comments. The manifest records all source and output hashes.

## Generating paper figures

```bash
python docs/build_figures.py \
  --outputs-root outputs \
  --output-dir outputs/v2_release_paper_assets/figures \
  --formats svg,pdf \
  --strict
```

Generated figures include:

```text
fig_construction_chain.*
fig_tuple_anatomy.*
fig_repro_pipeline.*
fig_chl2_chl1_gain.*
fig_y_sweep_heatmap.*
fig_orientation_lift_diagonal.*
fig_orientation_lift_kl.*
fig_scale_wave_d3.*
fig_scale_wave_residual.*
fig_hrp_model_ranking.*
fig_hrp_factorial_effects.*
figures_manifest.json
```

The first three diagrams are deterministic conceptual figures produced by `docs/figure_scripts/fig_conceptual.py`; the remaining figures are generated from canonical scientific CSVs.

## Compiling the v2.0.0 whitepaper

After the canonical paper assets exist:

```bash
SOURCE_DATE_EPOCH=0 python docs/build_paper.py \
  --tex docs/CHL2_conditional_hardy_littlewood_markov_whitepaper_v2_0_0_rc1.tex \
  --assets-dir outputs/v2_release_paper_assets \
  --output-dir outputs/v2_release_paper \
  --strict-assets
```

The builder validates the table and figure manifests, required asset hashes, publication-safe figure metadata, the release Matplotlib version, and LaTeX references before writing the PDF and `paper_build_manifest.json`. Passing `--build-assets` runs both asset builders first. The complete clean-checkout procedure is in [`docs/REPRODUCE_V2_0_0.md`](docs/REPRODUCE_V2_0_0.md).

## Mapping whitepaper items to scripts

| Whitepaper item | Producer | Principal source |
|---|---|---|
| Main likelihood | `chl2_consecutive_exclusion_audit.py` | `v2_release_chl2_main/chl2_conditional_summary.csv` |
| CHL2-vs-CHL1 block stability | `docs/build_tables.py` / `docs/build_figures.py` | `v2_release_chl2_main/chl2_metrics_by_block.csv` |
| Memory irreducibility | `chl2_consecutive_exclusion_audit.py` | `v2_release_chl2_main/chl2_memory_irreducibility.csv` |
| $Y$ sweep | `chl2_y_sweep.py` | `v2_release_chl2_y_sweep/chl2_y_sweep_gains.csv` |
| Absolute-prime previous-gap control | `chl2_consecutive_exclusion_audit.py` | `v2_release_chl2_os/chl2_os_prime_residue_transition_by_mod.csv` |
| Scale wave | `chl4d2_gap_population_bias_audit.py` | `v2_release_chl4d2_gap_population_bias/chl4d2_scale_wave_summary.csv` |
| Residual transfer audit | `chl4_modular_transfer_residual_audit.py` | `v2_release_chl4_residual_blocks_aligned/chl4_q3_diagonal_scalar.csv` |
| Orientation-lift replacement | `chl4d5_orientation_lift_os_replacement.py` | `v2_release_chl4d5_orientation_lift_os_replacement/chl2_os_old_direct_vs_oriented.csv` |
| H/R/P compatibility | `chl2_factor_compatibility_audit.py` | `v2_release_chl2_factor_compatibility/chl2_factorial_effects_summary.csv` |
| Paper tables | `docs/build_tables.py` | canonical local outputs above |
| Paper figures | `docs/build_figures.py` | canonical local outputs above plus deterministic conceptual diagrams |

## Runtime telemetry and provenance

Audit output directories contain `*_runtime_telemetry.json` and configuration JSON files. A07, A08, and A10 additionally record script SHA-256, Git commit, tracked-worktree status, and optional input SHA-256 values when run with `--hash-inputs`.

Runtime is hardware-dependent and is not part of the mathematical claim. Telemetry distinguishes execution conditions from scientific outputs.

## Research tool: CHL2 Hit@K oracle

The CHL2 cost

$$
\mathcal C_Y(g_2\mid g_1;x)
=-\log R_Y(g_2\mid g_1)+\Omega_Y^{\rm path}(g_1,g_2;x)
$$

can be used as a parameter-free candidate-prioritization score. It is not a primality test.

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

The public audits support these finite-window conclusions:

1. CHL2 improves CHL1 on DS1, with positive gain in all ten blocks of $S_{\rm all}$.
2. The CHL2-vs-CHL1 gain is stable over $Y\in\{31,47,61,73\}$ in the principal strata.
3. CHL2 outperforms the audited order-zero baselines in conditional likelihood.
4. The previous-gap-conditioned direct modular control has the wrong $q=3$ diagonal sign, while the orientation lift removes it.
5. The orientation lift improves weighted KL and $L^1$ for every audited block/modulus comparison.
6. The public H/R/P cube is complete; the $H{:}R$ interaction is negative across the audited block/filter cells, while CHL2 ($R\times P$ without $H$) remains the production kernel.

The repository does not claim an asymptotic theorem for primes. The results are finite-window computational evaluations of explicit kernels and diagnostics.

## License and citation

The project is released under the MIT license. Use `CITATION.cff` for v2.0.0 citation metadata, [`CHANGELOG.md`](CHANGELOG.md) for the version history, [`RELEASE_NOTES_v2.0.0.md`](RELEASE_NOTES_v2.0.0.md) for the release scope, and `docs/` for the manuscript source and build instructions.
