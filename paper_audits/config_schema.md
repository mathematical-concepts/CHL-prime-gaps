# Configuration schema

A minimal configuration JSON contains:

```json
{
  "input_dir": "data/ds1_1e11_w2e9_g2400",
  "blocks_dir": "blocks",
  "block_glob": "parent_wide_B{block:02d}.csv.gz",
  "blocks": [1,2,3,4,5,6,7,8,9,10],
  "real_prime_sequence": "real_primes.csv.gz",
  "gmax": 2400,
  "log_x_floor": 25
}
```

The audit scripts are deliberately permissive: additional fields are preserved
and ignored unless explicitly required by a script.
