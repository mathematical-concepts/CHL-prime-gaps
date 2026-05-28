# CHL3 second-order cluster audit package

Copy `paper_audits/chl3_second_order_cluster_audit.py` into the repository on the
experimental branch `feature/chl3-second-order-cluster`.

The script expects the stable CHL2 audit helper to remain available as:

```text
paper_audits/chl2_consecutive_exclusion_audit.py
```

and expects the CHL3 kernel patch to have been applied to `chl_kernel/`.

## Quick smoke run

```bash
python paper_audits/chl3_second_order_cluster_audit.py \
  --config data/quick_test/config.generated.json \
  --root . \
  --blocks 1-2 \
  --workers 1 \
  --parallel-mode auto \
  --output-dir outputs/quick_chl3
```

## DS1 block-1 diagnostic

```bash
python paper_audits/chl3_second_order_cluster_audit.py \
  --config data/ds1_1e11_w2e9_g2400/config.generated.json \
  --root . \
  --blocks 1 \
  --pmax 47 \
  --lowwheel-sets "3;3,5;3,5,7" \
  --workers 0 \
  --parallel-mode auto \
  --prime-csv AUTO \
  --os-prime-mods 3,5,7 \
  --output-dir outputs/chl3_B01
```

## Full DS1 run

```bash
python paper_audits/chl3_second_order_cluster_audit.py \
  --config data/ds1_1e11_w2e9_g2400/config.generated.json \
  --root . \
  --blocks 1-10 \
  --pmax 47 \
  --lowwheel-sets "3;3,5;3,5,7" \
  --workers 0 \
  --parallel-mode auto \
  --prime-csv AUTO \
  --os-prime-mods 3,5,7 \
  --output-dir outputs/chl3_full
```

## Output files

- `chl3_conditional_summary.csv`
- `chl3_pairwise_gains.csv`
- `chl3_memory_irreducibility.csv`
- `chl3_cluster_cache_*.csv.gz`
- `chl3_cluster_telemetry.json`
- `chl3_os_prime_residue_summary.csv` if `--prime-csv` is supplied
- `chl3_q3_diagnostic.csv` if OS runs
- `chl3_interpretacion.md`

## Reading rule

The main comparison is each CHL3 candidate against:

```text
CHL2_path_excl_cond_eta
```

A candidate is useful only if it improves the `q=3` absolute prime-residue
failure without materially damaging conditional log-likelihood or the `q=5,7`
OS diagnostics.
