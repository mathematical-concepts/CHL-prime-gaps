# Reproducing CHL2 v2.0.0

This guide reconstructs the v2.0.0 scientific outputs, generated tables and figures, and the release PDF from a clean checkout. The repository versions the producers and paper source; DS1, caches, scientific CSVs, generated paper assets, and the PDF are local artifacts under `data/` and `outputs/`.

## 1. Checkout and environment

```bash
git checkout v2.0.0
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

The paper figures are byte-reproducible under the frozen release environment. For Python 3.11 and later, `uv.lock` resolves Matplotlib `3.11.1`, the version used for the published asset hashes.

```bash
uv sync --frozen
```

A TeX installation with `latexmk` or `pdflatex` is required only for the final PDF.

`latexmk` is a Perl script and most TeX distributions ship no interpreter of their own. Linux and macOS provide one at `/usr/bin/perl`, but on Windows MiKTeX depends on an external Perl, and the usual one (Git for Windows, `Git\usr\bin\perl.exe`) is on `PATH` in Git Bash while absent from PowerShell and `cmd` — so the same machine builds from one shell and fails from another with `The script engine could not be found`. `docs/build_paper.py` resolves an interpreter automatically from Git for Windows or Strawberry Perl and prepends its directory to the subprocess `PATH` only; the system `PATH` is left untouched. If none is found the builder stops with an actionable message. Pass `--engine pdflatex` to avoid Perl entirely.

## 2. Test the public code

```bash
python -m compileall .
python -m pytest -q
```

The smoke suite creates its own small fixtures. It does not depend on DS1 or versioned scientific CSVs.

## 3. Generate DS1

```bash
python data_generation/generate_prime_gap_blocks.py \
  --start 100000000000 \
  --end 102000000000 \
  --gmax 2400 \
  --num-blocks 10 \
  --workers 16 \
  --output-dir data/ds1_1e11_w2e9_g2400
```

Expected structure:

```text
data/ds1_1e11_w2e9_g2400/
  config.generated.json
  real_primes.csv.gz
  blocks/parent_wide_B01.csv.gz
  ...
  blocks/parent_wide_B10.csv.gz
```

The release audit expects ten chronological blocks and a pair-count total of `78,934,825`.

## 4. A02 — main conditional likelihood audit

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

This produces the likelihood tables, per-block metrics, memory-irreducibility results, and the reusable cache:

```text
outputs/v2_release_chl2_main/chl2_path_exclusion_cache_Y47_logx25328436.csv.gz
```

Principal gate:

```text
CHL2 - CHL1 in S_all > 0
positive blocks in S_all = 10/10
```

## 5. A04 — truncation-horizon sweep

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

Principal gate:

```text
CHL2 - CHL1 positive for 4/4 Y values in every principal stratum
S_>240 retained only as a stress diagnostic
```

## 6. A06a — five-modulus previous-gap-conditioned control

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

All five moduli are mandatory. The v2.0.0 downstream audit rejects missing moduli, incomplete reduced-residue support, and zero-sum rows. The historical `naive_table4_legacy` fallback is not part of this chain.

## 7. A05 — gap-population scale wave

```bash
python paper_audits/chl4d2_gap_population_bias_audit.py \
  --config data/ds1_1e11_w2e9_g2400/config.generated.json \
  --root . \
  --blocks 1-10 \
  --Y 47 \
  --path-cache-file outputs/v2_release_chl2_main/chl2_path_exclusion_cache_Y47_logx25328436.csv.gz \
  --output-dir outputs/v2_release_chl4d2_gap_population_bias
```

The A05 filters act on `g2` alone. They are not the likelihood strata based on `max(g1,g2)`.

## 8. A07 — block-aligned direct modular residual

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

Release gates:

```text
model_support_gate_pass = true
missing_requested_mods = []
zero_sum_model_rows = []
block_alignment_matches_endpoint_adjusted_plan = true
counted_transitions = 78,934,823
block_plan_count_deltas = -1,0,0,0,0,0,0,0,0,-1
```

## 9. A08 — orientation-lifted replacement

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

Release gates:

```text
old_model_support_gate_pass = true
replacement_gate_pass = true
KL improvements = 55/55
L1 improvements = 55/55
q=3 old wrong sign = 11/11
q=3 oriented wrong sign = 0/11
legacy Table 4 provenance = closed
```

## 10. A10 — common-support H/R/P factorial audit

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

Release gates:

```text
complete_factorial_cube_gate_pass = true
common_support_gate_pass = true
terminal_anchor_gate_pass = true
factorial_effects_complete_gate_pass = true
context_effects_complete_gate_pass = true
finite_metrics_gate_pass = true
a10_release_gate_pass = true
```

## 11. Generate tables and figures

```bash
python docs/build_tables.py \
  --outputs-root outputs \
  --output-dir outputs/v2_release_paper_assets/tables \
  --strict

SOURCE_DATE_EPOCH=0 python docs/build_figures.py \
  --outputs-root outputs \
  --output-dir outputs/v2_release_paper_assets/figures \
  --formats svg,pdf \
  --strict
```

Expected inventory:

```text
10 LaTeX table fragments
11 PDF figures
11 SVG figures
tables_manifest.json
figures_manifest.json
```

The figure manifest must record:

```text
matplotlib_version = 3.11.1
pdf_fonttype = 42
ps_fonttype = 42
svg_hashsalt = chl2-v2.0.0
source_date_epoch = 0
```

## 12. Compile the paper

```bash
SOURCE_DATE_EPOCH=0 python docs/build_paper.py \
  --tex docs/CHL2_conditional_hardy_littlewood_markov_whitepaper_v2_0_0_rc1.tex \
  --assets-dir outputs/v2_release_paper_assets \
  --output-dir outputs/v2_release_paper \
  --strict-assets
```

Outputs:

```text
outputs/v2_release_paper/CHL2_conditional_hardy_littlewood_markov_whitepaper_v2_0_0_rc1.pdf
outputs/v2_release_paper/CHL2_conditional_hardy_littlewood_markov_whitepaper_v2_0_0_rc1.log
outputs/v2_release_paper/paper_build_manifest.json
```

The builder rejects missing assets, manifest hash mismatches, skipped figure/table builds, Type-3 figure metadata, the wrong Matplotlib version, undefined references, undefined citations, and missing files.

### PDF determinism is per TeX distribution

`SOURCE_DATE_EPOCH=0` removes build timestamps, so repeated builds on one machine with a fixed TeX distribution are byte-identical and the calling shell is irrelevant: PowerShell and Git Bash produce the same PDF. The PDF is **not** expected to be byte-identical across different TeX distributions.

#### Current release-candidate digest

| Build platform | PDF SHA256 | Verification |
|---|---|---|
| Windows, MiKTeX 26.5 (pdfTeX 1.40.29) | `d3099a3318f707507675d091c238e304da82937f1e87041cdbaa70742423c1e9` | Two consecutive clean-asset builds matched on 2026-08-24. |
| Ubuntu 22.04 (WSL), TeX Live 2022 (pdfTeX 1.40.22) | _Not regenerated after the final construction-chain figure correction._ | Do not use the previous TeX Live digest as a v2.0.0 release hash. |

The final correction to `fig_construction_chain` changed a generated figure and therefore invalidated every PDF digest from the preceding RC1 asset set. The MiKTeX digest above is the current canonical release candidate. A TeX Live digest should be published only after rebuilding the corrected final assets under that distribution.

#### Why TeX distributions may differ

A controlled pre-finalization comparison, made from identical assets under MiKTeX 26.5 and TeX Live 2022, confirmed that distribution changes can alter the PDF bytes even when the scientific content and release gates agree. In that comparison, both PDFs agreed on page count (22), page size, PDF version, and embedded Type-1 font subsets; the subset tags were identical and neither PDF contained a Type-3 font.

The difference was not confined to metadata. Rendered at 150 dpi, the two pre-finalization PDFs differed in 0.26% of pixels (worst page 1.52%), spread across the whole text block rather than localized in one figure. The cause was package drift: 39 of the 71 loaded LaTeX packages differed in version, chiefly **microtype 3.2d vs 3.0b**, whose protrusion and font-expansion tables shift glyphs on justified lines, and **amsmath 2.18d vs 2.17l**, which changes formula spacing and makes mathematics-dense pages the worst affected.

pdfTeX 1.40.29 additionally wrote correct `ToUnicode` maps for large operators and horizontal braces. Under TeX Live 2022, the same glyphs were extracted as raw Computer Modern pieces, which degraded copy-paste and screen-reader output:

| Glyph in the PDF | MiKTeX extracts | TeX Live 2022 extracts |
|---|---|---|
| Product | `∏` | `Y` |
| Sum | `∑` | `X` |
| `\underbrace` | `⏟` | `\| {z }` |

Publish release hashes from one pinned platform. The current v2.0.0 release-candidate digest above was produced with MiKTeX on Windows. Reproducing a byte-identical digest on Linux would require a closely matched modern TeX installation and is still not guaranteed. Reviewers rebuilding on another distribution should expect a different hash and should verify the document by content and builder gates rather than by digest alone.

## 13. Final release checks

```bash
python -m pytest -q
git status --short
git diff --check
```

Then inspect the generated PDF visually and run the repository's PDF preflight workflow. The tag and publication package should contain the source repository plus the generated PDF as a release artifact. DS1 and scientific CSVs remain local and are not added to Git.

## Historical Table 4 note

The v1.8 `q=11` and `q=13` values `0.100000` and `0.083333...` arose from missing upstream moduli, zero model rows, and a downstream uniform-row fallback. v2.0.0 uses `absolute_prime_os_previous_gap_conditioned` for all five moduli and treats the legacy object only as historical provenance. See `docs/NAIVE_TABLE4_LEGACY_PROVENANCE.md`.
