# CHL-prime-gaps

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20368549.svg)](https://doi.org/10.5281/zenodo.20368549)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Reproducible code for the **Conditional Hardy--Littlewood Markov kernel** for
consecutive prime gaps.

The repository is organized around one mathematical object:

$$
P_Y^{\mathrm{CHL2}}(g_2\mid g_1;x)
\propto
\mathbf 1_{\mathrm{adm}}^{(3)}(g_1,g_2;Y)
\frac{\mathfrak S_Y(\{0,g_1,g_1+g_2\})}{\mathfrak S_Y(\{0,g_1\})}
\exp[-\Omega_Y^{\mathrm{path}}(g_1,g_2;x)]
\exp(\eta_Y g_2).
$$

Here `Y` is an auditable **truncation horizon** for the singular series, not a
classical asymptotic sieve level.  The no-interior-prime survival term is the
first-order Poisson correction

$$
\Omega_Y^{\mathrm{path}}(g_1,g_2;x)
=
\sum_{2\le u\le g_2-2,\ u\ \mathrm{even}}
\frac{\mathfrak S_Y(\{0,g_1,g_1+u,g_1+g_2\})}
     {\mathfrak S_Y(\{0,g_1,g_1+g_2\})}\frac{1}{\log x}.
$$

## Repository layout

```text
CHL-prime-gaps/
  data_generation/   segmented-sieve generation of empirical prime-gap blocks
  chl_kernel/        pure Python mathematical kernel, importable as a package
  paper_audits/      scripts that reproduce the whitepaper tables
  research_tools/    CHL2 Hit@K candidate-ranking oracle
  configs/           example JSON configurations
  docs/              whitepaper and appendix material
  tests/             smoke tests
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Running the Smoke Test

Run the kernel smoke test:

```bash
python tests/test_kernel_smoke.py
```

Then generate a small empirical dataset:

```bash
python data_generation/generate_prime_gap_blocks.py --quick-test --output-dir data/quick_test
```

Run the CHL2 audit on it:

```bash
python paper_audits/chl2_consecutive_exclusion_audit.py \
  --config data/quick_test/config.generated.json \
  --root . \
  --blocks 1-2 \
  --path-exclusion \
  --workers 2 \
  --parallel-mode auto \
  --output-dir outputs/quick_chl2
```

Run the CHL2 Hit@K oracle:

```bash
python research_tools/chl2_hitk_oracle.py \
  --p-current 1000003 \
  --g-prev 6 \
  --Y 47 \
  --gmax 300 \
  --top-k 20
```

## DS1-style reproduction

Generate empirical data around \(10^{11}\):

```bash
python data_generation/generate_prime_gap_blocks.py \
  --start 100000000000 \
  --end 102000000000 \
  --gmax 2400 \
  --num-blocks 10 \
  --workers 16 \
  --output-dir data/ds1_1e11_w2e9_g2400
```

Main CHL2 audit:

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

Y-sweep:

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

Oliver--Soundararajan absolute prime-residue diagnostic:

```bash
python paper_audits/chl2_consecutive_exclusion_audit.py \
  --config data/ds1_1e11_w2e9_g2400/config.generated.json \
  --root . \
  --path-exclusion \
  --reuse-path-cache \
  --prime-csv AUTO \
  --os-prime-mods 3,5,7 \
  --require-os \
  --os-only \
  --output-dir outputs/chl2_os
```

## Importable kernel

```python
from chl_kernel import CHLKernel

kernel = CHLKernel(Y=47, log_x=25.328436)
print(kernel.log_R(g1=6, g2=10))
print(kernel.omega_path(g1=6, g2=10))
print(kernel.hit_cost(g1=6, g2=10))
```

## Scientific scope

This repository does not claim a proof of the Riemann hypothesis, Goldbach's
conjecture, the twin-prime conjecture, or new asymptotic prime-gap bounds. It
provides a reproducible parameter-free conditional Markov kernel derived from
truncated Hardy--Littlewood singular-series ratios plus a first-order Poisson
no-interior-prime survival correction, together with empirical audits and
research tools.

The first-order Poisson no-interior correction is deliberately limited. In the
absolute prime-residue Oliver--Soundararajan diagnostic, CHL2 captures the
modular repulsion structure well for modulus 7 and moderately for modulus 5,
but it has a documented wrong-sign failure in modulus 3. This low-wheel anomaly
is treated as a boundary of the first-order approximation, not as a positive
validation.

## Citation

Zenodo archive: https://doi.org/10.5281/zenodo.20368549

```bibtex
@software{cano_gregorio_chl_prime_gaps_2026,
  author    = {Cano Gregorio, Jose Antonio},
  title     = {CHL-prime-gaps: Conditional Hardy--Littlewood Markov kernels for consecutive prime gaps},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v0.0},
  doi       = {10.5281/zenodo.20368549},
  url       = {https://doi.org/10.5281/zenodo.20368549},
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
