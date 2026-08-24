# Changelog

All notable public changes to **CHL-prime-gaps** are documented here.

## [2.0.0] - 2026-08-24

### Added

- A public, self-contained `H/R/P` common-support factorial audit.
- Block-aligned modular residual and character-spectrum diagnostics.
- Strict scientific-input gates for modular model support, reduced-residue cells, row normalization, and positive row mass.
- Deterministic builders for LaTeX tables, data-driven figures, conceptual figures, and the release PDF.
- SHA-256 manifests connecting scientific CSVs to tables, figures, and the paper.
- A clean-checkout reproduction guide in `docs/REPRODUCE_V2_0_0.md`.
- A documented forensic reconstruction of the v1.8 Table 4 legacy artifact.

### Changed

- The canonical modular comparison now uses the fully specified five-modulus control `absolute_prime_os_previous_gap_conditioned`.
- The orientation lift is evaluated from aggregated `ALL` matrices and chronological B01--B10 blocks with explicit provenance.
- The paper distinguishes likelihood strata based on `max(g1,g2)` from scale-wave filters applied to `g2` alone.
- Figure generation is pinned to Matplotlib `3.11.1`, uses embedded Type-42 PDF fonts, and stabilizes PDF/SVG metadata.
- The package version and citation metadata are updated to `2.0.0` and the conceptual Zenodo DOI.
- `docs/build_paper.py` resolves a Perl interpreter for `latexmk` on every platform: if `perl` is absent from `PATH` it is recovered from Git for Windows (`Git\usr\bin`), Strawberry Perl, or the standard POSIX locations, and only the subprocess `PATH` is extended. When no interpreter exists the builder raises a platform-specific, actionable error instead of MiKTeX's `The script engine could not be found`; `--engine pdflatex` bypasses Perl entirely.
- `docs/build_paper.py` now echoes the last 40 lines of a failed toolchain command to stderr before re-raising, so a LaTeX or asset-builder failure reports its own cause rather than an opaque `CalledProcessError` traceback.
- `docs/REPRODUCE_V2_0_0.md` documents the Perl requirement, states that PDF determinism holds per TeX distribution, records the verified MiKTeX release-candidate digest, and separates the earlier cross-distribution comparison from the current release-candidate hash.

### Corrected

- Reconstructed the origin of the v1.8 Table 4 entries for `q=11` and `q=13`: the upstream OS CSV omitted those moduli, the residual audit emitted zero model rows, and downstream normalization replaced them with uniform rows. This produced `1/(q-1)`, namely `0.100000` and `0.083333...`.
- Prevented missing-modulus or zero-row scientific inputs from being silently converted into uniform transition matrices.
- Corrected parent-wide block alignment so the two unavoidable endpoint omissions occur in B01 and B10 without shifting internal block boundaries.
- Corrected the block-stability summary of the dominant non-principal character energy.
- Replaced the unsupported interpretation of `Omega_path` as an autonomous modular mechanism with the narrower consecutive-gap survival interpretation.
- Fixed a shell-dependent paper build on Windows: `latexmk` is a Perl script, MiKTeX bundles no interpreter, and the only one present (Git for Windows) is on `PATH` in Git Bash but not in PowerShell or `cmd`. The same checkout therefore built from one shell and failed from another, with no `.log` written and the real cause hidden inside the captured subprocess output.
- Corrected the construction-chain diagram so the four arrow labels occupy a separate band and no longer overlap the following process boxes.

### Removed from release claims

- Quantitative low-wheel and local truncated LOS results whose exact public producers were not retained in the minimal release chain.
- The mixed `naive_table4_legacy` object as a scientific baseline.

## [1.8.1] - 2026-05-24

- Published the self-contained CHL2 finite-window protocol and the initial orientation-lifted modular diagnostic.
- Added DS1 likelihood, finite-`Y` stability, and reproducibility scripts.
