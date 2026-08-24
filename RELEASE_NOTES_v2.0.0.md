# CHL-prime-gaps v2.0.0

Version 2.0.0 is the minimal academic release of the CHL2 finite-window conditional Hardy--Littlewood Markov kernel. It starts from the public v1.8.1 code line, preserves the production kernel, and closes the principal reproducibility, modular-diagnostic, and factorial-compatibility audits needed for a new preprint.

## Production kernel

The release keeps:

```text
CHL1 = R
CHL2 = R * P
```

with

```text
R = conditional singular-series ratio
P = exp(-Omega_path), the first-order no-interior survival factor
```

Likelihood is an evaluation metric, not a factor of the kernel.

## Main finite-window result

The DS1 window is:

```text
[10^11, 10^11 + 2*10^9]
10 chronological blocks
78,934,825 pair-count transitions
finite support g <= 2400
```

In `S_all`, CHL2 improves CHL1 by approximately `8.376322e-4` nats per event, positive in 10/10 chronological blocks. The sign remains positive in all principal strata for `Y = 31,47,61,73`. The sparse `S_>240` cell is retained only as a stress diagnostic.

## Modular diagnostic replacement

Version 2.0.0 uses:

```text
absolute_prime_os_previous_gap_conditioned
```

as the reproducible five-modulus direct control and compares it with:

```text
orientation_lift_chl2_gap_population
```

The orientation lift improves weighted KL and L1 in all 55 block--modulus comparisons and changes the `q=3` wrong-sign flag from 11/11 to 0/11.

## Historical Table 4 correction

The v1.8 Table 4 values:

```text
q=11 -> 0.100000
q=13 -> 0.083333...
```

were produced by a mixed implementation artifact:

```text
upstream OS CSV omitted q=11,13
-> residual audit emitted zero rows
-> downstream normalizer used a uniform-row fallback
-> diagonal = 1/(q-1)
```

The provenance is now closed and documented in `docs/NAIVE_TABLE4_LEGACY_PROVENANCE.md`. The legacy object is not used as a scientific baseline in v2.0.0. Strict gates now reject absent moduli, incomplete support, non-finite values, non-normalized rows, and zero-sum rows.

## H/R/P compatibility

The public A10 audit evaluates all eight vertices of the common-support `2^3` cube:

```text
H = marginal two-point singular-series factor
R = conditional singular-series ratio
P = path survival
```

The full-window ranking retains CHL2 as the best vertex. The `H:R` interaction is stably negative, and adding `H` when `R` is already present reduces likelihood both with and without `P`. Adding `P` to CHL1 produces the positive CHL2 gain.

## Paper and reproducibility chain

The repository versions the producers and manuscript source, not DS1 or generated scientific CSVs:

```text
scripts
-> local data and audit outputs
-> generated LaTeX tables and vector figures
-> TeX
-> PDF
```

The release includes:

```text
docs/build_tables.py
docs/build_figures.py
docs/build_paper.py
docs/REPRODUCE_V2_0_0.md
docs/CHL2_conditional_hardy_littlewood_markov_whitepaper_v2_0_0_rc1.tex
```

The paper assets are deterministic under the locked environment:

```text
Matplotlib 3.11.1
PDF Type-42 fonts
SOURCE_DATE_EPOCH=0
fixed SVG hash salt
SHA-256 manifests
```

The generated PDF is a GitHub/Zenodo/arXiv release artifact and is not committed with scientific outputs.

## Scope and exclusions

Version 2.0.0 does not include:

```text
chl_lab/
full Lab v2 DAGs
DS1 or caches
scientific output directories
CHL-Z
Lockstep/IE experiments
the detailed Holt follow-up supplement
quantitative low-wheel or local LOS claims without public producers
```

The release also withdraws the interpretation of `Omega_path` as an autonomous modular mechanism. It remains the first-order consecutive-gap survival factor and contributes to likelihood; residue-conditioned raw averages are length-confounded.

## Citation

Use the conceptual Zenodo DOI:

```text
10.5281/zenodo.20368548
```

The version-specific v2.0.0 DOI will be assigned by Zenodo after publication and does not require retagging the repository.
