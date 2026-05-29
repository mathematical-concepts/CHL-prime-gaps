# Data generation

This directory contains the theory-free segmented-sieve generator used to create
empirical consecutive-prime-gap blocks.

The generator exports:

- `real_primes.csv.gz` — chronological primes inside the requested window.
- `blocks/parent_wide_BXX.csv.gz` — observed counts `H` for consecutive gap pairs `(g1,g2)`.
- `config.generated.json` — an audit configuration compatible with `paper_audits/`.

Quick test:

```bash
python data_generation/generate_prime_gap_blocks.py --quick-test --output-dir data/quick_test
```

DS1-style window:

```bash
python data_generation/generate_prime_gap_blocks.py \
  --start 100000000000 \
  --end 102000000000 \
  --gmax 2400 \
  --num-blocks 10 \
  --workers 16 \
  --output-dir data/ds1_1e11_w2e9_g2400
```

## Runtime telemetry

Every successful run writes `data_generation_telemetry.json` in the output directory.  The file records wall-clock runtime, command-line arguments, CPU count, segment count, prime counts, block counts, and output paths.  It is a reproducibility aid only; it is not used by any CHL kernel or audit metric.
