# Documentation

This directory contains the paper-level documentation for **CHL-prime-gaps**. The main repository code is organized around reproducible audits; the `docs/` directory is the place to read the mathematical motivation, definitions, protocol, and interpretation without following the Python implementation line by line.

## Current whitepaper

The current manuscript is:

```text
CHL2_conditional_hardy_littlewood_markov_whitepaper_v1_8.pdf
CHL2_conditional_hardy_littlewood_markov_whitepaper_v1_8.tex
```

The PDF is the recommended entry point. The LaTeX source is included so that reviewers can inspect the formulas, references, tables, and figure captions used in the paper.

The whitepaper studies a finite conditional Hardy--Littlewood Markov kernel for consecutive prime gaps. In brief:

1. CHL1 converts a Hardy--Littlewood singular-series ratio into a conditional transition kernel.
2. CHL2 adds a first-order no-interior-prime survival factor, turning a triple-constellation model into a consecutive-gap model.
3. The orientation-lifted modular diagnostic converts gap-residue populations into reduced-residue transition matrices using valid directed edges.
4. The apparent old $q=3$ modular anomaly is resolved by this orientation lift, not by changing the CHL2 gap kernel.

## Recommended reading order

A reader new to the project should follow this order:

1. `CHL2_conditional_hardy_littlewood_markov_whitepaper_v1_8.pdf`  
   Main mathematical and computational paper.

2. `ORIENTATION_LIFT_NOTE.md`, if present  
   Short conceptual note on the valid-edge projection from gap-residue populations to residue-transition matrices.

3. `CHL4_RESIDUAL_SPECTRAL_TRANSFER_SUMMARY.md`, if present  
   Historical audit note explaining how the old $q=3$ issue was traced to an orientation-lift defect.

4. Roadmap / audit notes, if present  
   These are development notes for specific audit phases. They are useful for understanding how the final diagnostic was reached, but the whitepaper is the canonical reference.

## Relationship between documents and code

The documentation should be read together with the main repository `README.md`, which maps whitepaper tables and figures to reproducible scripts and output CSVs.

The most important audit families are:

| Purpose | Main script family | Typical outputs |
|---|---|---|
| CHL2 likelihood and baselines | `paper_audits/chl2_consecutive_exclusion_audit.py` | `chl2_conditional_summary.csv`, `chl2_pairwise_gains.csv`, `chl2_memory_irreducibility.csv` |
| Cutoff stability in $Y$ | `paper_audits/chl2_y_sweep.py` | `chl2_y_sweep_stability.csv`, `chl2_y_sweep_summary.csv` |
| Empirical prime-residue matrices | `paper_audits/chl4_modular_transfer_residual_audit.py` | `chl4_transfer_empirical_matrices.csv`, `chl4_transfer_chl2_matrices.csv` |
| Gap-population bias checks | `paper_audits/chl4d2_gap_population_bias_audit.py` | `chl4d2_gap_population_by_residue.csv`, `chl4d2_scale_wave_summary.csv` |
| Orientation lift generalization | `paper_audits/chl4d4_orientation_lift_generalization_audit.py` | `chl4d4_orientation_lift_matrices.csv`, `chl4d4_direct_vs_orientation_lift.csv` |
| Orientation-lifted OS replacement | `paper_audits/chl4d5_orientation_lift_os_replacement.py` | `chl2_os_old_direct_vs_oriented.csv`, `chl2_os_oriented_prime_residue_summary.csv` |

The exact commands are maintained in the root `README.md` so that the documentation here stays mostly mathematical.

## Mathematical naming versus repository naming

The whitepaper uses mathematical stratum names, while some CSVs and scripts retain older legacy labels. The common mapping is:

| Paper notation | Legacy CSV/script label | Meaning |
|---|---|---|
| $S_{\rm all}$ | `ALL` | all evaluated gap pairs |
| $S_{\rm dense}$ | `LOW_ONLY_LE58` | $G_{\max}\le 58$ |
| $S_{\rm trans}$ | `MID_59_120` | $59\le G_{\max}\le 120$ |
| $S_{121:240}$ | `MID_121_240` | $121\le G_{\max}\le 240$ |
| $S_{121:400}$ | `MID_121_400` | $121\le G_{\max}\le 400$ |
| $S_{>58}$ | `NO_58` | $G_{\max}>58$ |
| $S_{>120}$ | `NO_120` | $G_{\max}>120$ |
| $S_{>240}$ | `NO_240` | $G_{\max}>240$; sparse stress diagnostic |

Here

$$G_{\max}(g_1,g_2)=\max(g_1,g_2).$$

## Orientation lift in one paragraph

If a gap kernel predicts a gap-residue population

$$p_r=P(g\equiv r\pmod q),$$

then this population is not yet a row-wise transition matrix between reduced residue classes. The valid-edge orientation lift distributes $p_r$ over the directed edges

$$b\to b+r\pmod q$$

for which both $b$ and $b+r$ are reduced residue classes. If

$$N_r(q)=\#\{b\in(\mathbb Z/q\mathbb Z)^*: b+r\in(\mathbb Z/q\mathbb Z)^*\},$$

then each valid edge receives mass proportional to $p_r/N_r(q)$ before row normalization. This is the diagnostic layer that resolves the old apparent $q=3$ modular anomaly.

## Telemetry and reproducibility

Most audit scripts write a runtime telemetry file:

```text
*_runtime_telemetry.json
```

These files record information such as:

```text
elapsed_seconds
command-line arguments
Python version
platform information
CPU count
block list
row counts
output file paths
```

Telemetry is included to make runtime claims and reproducibility checks auditable. It is not part of the mathematical model.

## Documentation style rules

Markdown files in this repository use GitHub-compatible math delimiters:

```text
Inline math: $...$
Display math: $$...$$
```

Avoid backslash-parenthesis and backslash-bracket math delimiters in Markdown files. Also avoid compact expressions such as `{u<v}` inside math; write `{u < v}` instead.

## Versioning notes

The current documentation release is **v1.8**. Earlier development names such as `CHL3` and `CHL4-D` refer to audit phases and script families, not to separate mathematical kernels replacing CHL2. The main mathematical object remains the CHL2 conditional Hardy--Littlewood gap kernel; the orientation lift is a modular diagnostic layer.

## License and citation

The repository is released under the MIT License. Citation information is maintained in the root `CITATION.cff` file and in the root `README.md`.
