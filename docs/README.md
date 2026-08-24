# Documentation

This directory contains the mathematical manuscript, historical notes, and reproducible paper-build tools for **CHL-prime-gaps**.

## Current v2.0.0 manuscript

The versioned manuscript source is:

```text
CHL2_conditional_hardy_littlewood_markov_whitepaper_v2_0_0_rc1.tex
```

The PDF is generated locally rather than committed with scientific outputs:

```text
outputs/v2_release_paper/
  CHL2_conditional_hardy_littlewood_markov_whitepaper_v2_0_0_rc1.pdf
  CHL2_conditional_hardy_littlewood_markov_whitepaper_v2_0_0_rc1.log
  paper_build_manifest.json
```

The archived v1.8 source and PDF remain in this directory for historical comparison:

```text
CHL2_conditional_hardy_littlewood_markov_whitepaper_v1_8.tex
CHL2_conditional_hardy_littlewood_markov_whitepaper_v1_8.pdf
```

The v2.0.0 manuscript studies:

1. CHL1, the conditional singular-series ratio kernel.
2. CHL2, which adds first-order no-interior-prime survival.
3. DS1 conditional likelihood, chronological block stability, and the finite `Y` sweep.
4. The modulo-3 gap-population scale wave, with explicit `g2` filter notation.
5. The block-aligned previous-gap-conditioned modular residual.
6. The valid-edge orientation lift and its replacement of the direct modular control.
7. The common-support `H/R/P` factorial compatibility audit.
8. The reconstructed provenance of the v1.8 Table 4 missing-modulus fallback.

The paper does not claim an asymptotic theorem or a proof of RH, Goldbach, or the twin-prime conjecture.

## Recommended reading order

1. `CHL2_conditional_hardy_littlewood_markov_whitepaper_v2_0_0_rc1.tex`
   Main v2.0.0 mathematical and computational paper source.

2. `REPRODUCE_V2_0_0.md`
   Clean-checkout reconstruction of DS1, A02--A10, paper assets, and the PDF.

3. `NAIVE_TABLE4_LEGACY_PROVENANCE.md`
   Historical reconstruction of the v1.8 `q=11,13` uniform-row fallback.

4. `ORIENTATION_LIFT_NOTE.md`
   Short conceptual note on the valid-edge projection.

5. `CHL4_RESIDUAL_SPECTRAL_TRANSFER_SUMMARY.md`
   Historical audit note on the direct modular residual and character structure.

## Public paper builders

### Tables

```bash
python docs/build_tables.py \
  --outputs-root outputs \
  --output-dir outputs/v2_release_paper_assets/tables \
  --strict
```

This produces ten LaTeX fragments plus `tables_manifest.json`. Each fragment records its source path and SHA-256 in comments.

### Figures

```bash
SOURCE_DATE_EPOCH=0 python docs/build_figures.py \
  --outputs-root outputs \
  --output-dir outputs/v2_release_paper_assets/figures \
  --formats svg,pdf \
  --strict
```

The figure builder produces eleven figures:

```text
fig_construction_chain
fig_tuple_anatomy
fig_repro_pipeline
fig_chl2_chl1_gain
fig_y_sweep_heatmap
fig_orientation_lift_diagonal
fig_orientation_lift_kl
fig_scale_wave_d3
fig_scale_wave_residual
fig_hrp_model_ranking
fig_hrp_factorial_effects
```

The first three are deterministic conceptual diagrams. The other eight are generated from canonical audit CSVs. PDF figures use embedded Type-42 fonts; PDF/SVG timestamps and SVG IDs are stabilized for reproducible hashes.

### Paper

```bash
SOURCE_DATE_EPOCH=0 python docs/build_paper.py \
  --tex docs/CHL2_conditional_hardy_littlewood_markov_whitepaper_v2_0_0_rc1.tex \
  --assets-dir outputs/v2_release_paper_assets \
  --output-dir outputs/v2_release_paper \
  --strict-assets
```

The paper builder validates:

```text
required table and figure inventory
manifest SHA-256 values
no skipped release assets
Matplotlib 3.11.1
pdf.fonttype = 42
SOURCE_DATE_EPOCH = 0
fixed SVG hash salt
no undefined references or citations
```

It writes `paper_build_manifest.json` next to the generated PDF.

## Relationship between documents and code

The root `README.md` maps every audit to its canonical command and output directory. The main public audit families are:

| Purpose | Main producer | Principal outputs |
|---|---|---|
| CHL2 likelihood and baselines | `paper_audits/chl2_consecutive_exclusion_audit.py` | `chl2_conditional_summary.csv`, `chl2_metrics_by_block.csv`, `chl2_memory_irreducibility.csv` |
| Truncation-horizon stability | `paper_audits/chl2_y_sweep.py` | `chl2_y_sweep_gains.csv`, `chl2_y_sweep_stability.csv` |
| Gap-population scale wave | `paper_audits/chl4d2_gap_population_bias_audit.py` | `chl4d2_scale_wave_summary.csv` |
| Block-aligned modular residual | `paper_audits/chl4_modular_transfer_residual_audit.py` | `chl4_q3_diagonal_scalar.csv`, `chl4_character_spectrum.csv` |
| Orientation-lift replacement | `paper_audits/chl4d5_orientation_lift_os_replacement.py` | `chl2_os_old_direct_vs_oriented.csv`, oriented matrices and summaries |
| H/R/P compatibility | `paper_audits/chl2_factor_compatibility_audit.py` | model ranking, factorial effects, context effects, support audit |
| Paper tables | `docs/build_tables.py` | ten generated `.tex` fragments |
| Paper figures | `docs/build_figures.py` | eleven PDF/SVG figures |
| Paper PDF | `docs/build_paper.py` | release PDF and build manifest |

## Mathematical naming versus repository naming

The likelihood strata use `G_max = max(g1,g2)`:

| Paper notation | CSV label | Meaning |
|---|---|---|
| $S_{\rm all}$ | `ALL` | all evaluated gap pairs |
| $S_{\rm dense}$ | `LOW_ONLY_LE58` | $G_{\max}\le58$ |
| $S_{\rm trans}$ | `MID_59_120` | $59\le G_{\max}\le120$ |
| $S_{121:240}$ | `MID_121_240` | $121\le G_{\max}\le240$ |
| $S_{121:400}$ | `MID_121_400` | $121\le G_{\max}\le400$ |
| $S_{>58}$ | `NO_58` | $G_{\max}>58$ |
| $S_{>120}$ | `NO_120` | $G_{\max}>120$ |
| $S_{>240}$ | `NO_240` | sparse stress diagnostic |

The A05 scale-wave filters act on the candidate gap `g2` alone and are displayed as $B^{(g_2)}$ filters. They must not be interpreted as the likelihood strata above.

## Orientation lift in one paragraph

If a gap kernel predicts

$$p_r=P(g\equiv r\pmod q),$$

then this population is not yet a row-wise transition matrix between reduced residues. The valid-edge orientation lift distributes $p_r$ over directed edges

$$b\to b+r\pmod q$$

for which both endpoints are reduced residues. If

$$N_r(q)=\#\{b\in(\mathbb Z/q\mathbb Z)^*:b+r\in(\mathbb Z/q\mathbb Z)^*\},$$

then each valid edge receives mass $p_r/N_r(q)$ before row normalization. This diagnostic layer removes the apparent `q=3` wrong-sign discrepancy of the reproducible previous-gap-conditioned control.

## Historical Table 4 provenance

The v1.8 entries `q=11 -> 0.100000` and `q=13 -> 0.083333...` arose from missing upstream moduli, zero model rows, and a downstream uniform-row fallback. v2.0.0 uses the fully specified five-modulus control:

```text
absolute_prime_os_previous_gap_conditioned
```

The legacy object is documented as `naive_table4_legacy` and withdrawn as a scientific baseline. See `NAIVE_TABLE4_LEGACY_PROVENANCE.md`.

## Telemetry and reproducibility

Audit telemetry records execution conditions separately from scientific outputs. A07, A08, and A10 additionally record script SHA-256, Git commit, tracked-worktree state, and optional input hashes. The paper manifests then connect the canonical scientific CSVs to the generated tables, figures, and PDF.

## Documentation style

Markdown uses GitHub-compatible math delimiters:

```text
Inline math: $...$
Display math: $$...$$
```

Generated data and paper assets belong under `outputs/` and are not committed. The versioned source of truth is the producer code plus the manuscript source.

## License and citation

The repository is released under the MIT License. Citation metadata is maintained in the root `CITATION.cff` and uses the conceptual Zenodo DOI `10.5281/zenodo.20368548` for v2.0.0.
