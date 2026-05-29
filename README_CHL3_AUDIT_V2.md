# CHL3 Audit v2: multimodel OS, hard-zero telemetry, self+cluster variants

This experimental patch applies the next CHL3 auditing layer on top of the
`feature/chl3-second-order-cluster` branch.

## What changed

1. **Hard-zero telemetry is now explicit and auditable.**
   The cluster cache includes, for each low-wheel set:

   - `hard_zero_h5_pairs_<tag>`
   - `hard_zero_h5_weight_<tag>`
   - `hard_zero_h5_kappa_mass_<tag>`
   - `nonzero_h5_pairs_<tag>`
   - `kappa_negative_mass_<tag>`
   - `kappa_positive_mass_<tag>`
   - `skipped_zero_h4_pairs_<tag>`

   This tells us whether the low-wheel Möbius hard-zero is actually firing and
   how much first-order mass it carries.

2. **`expected_count_model` is written consistently.**
   In the Oliver--Soundararajan transition matrix CSV, the expected count is now
   always:

   ```python
   expected_count_model = row_count * model_probability
   ```

3. **The absolute OS diagnostic supports multiple models in one run.**
   Use:

   ```bash
   --os-models all
   ```

   to evaluate CHL2, Bernoulli, all CHL3 low-wheel models and all CHL3
   self+cluster models in the same OS CSV.

4. **Self+cluster variants were added.**
   The new parameter-free variant is:

   \[
   \Omega_{\mathrm{self+cluster}}^{(2)}
   =
   \sum_u p_u
   +\frac12\sum_u p_u^2
   -\sum_{u<v}\kappa_{uv}.
   \]

   Models:

   - `CHL3_self_lowwheel2_p3_cond_eta`
   - `CHL3_self_lowwheel2_p3p5_cond_eta`
   - `CHL3_self_lowwheel2_p3p5p7_cond_eta`

## Apply

From the repository root:

```bash
cp patch/paper_audits/chl3_second_order_cluster_audit.py paper_audits/chl3_second_order_cluster_audit.py
cp patch/chl_kernel/cluster_expansion.py chl_kernel/cluster_expansion.py
```

Then run:

```bash
python -m compileall .
python tests/test_kernel_smoke.py
python -m pytest -q tests/test_cluster_expansion.py
```

## Recommended B01 rerun

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
  --os-models all \
  --output-dir outputs/chl3_B01_v2_multimodel_self
```

## First things to inspect

- `chl3_cluster_cache_*.csv.gz`: hard-zero and self-term columns.
- `chl3_cluster_telemetry.json`: aggregate means/sums for hard-zero mass.
- `chl3_os_prime_residue_summary.csv`: compare `q=3` for all CHL3 variants.
- `chl3_q3_diagnostic.csv`: check diagonal sign and `pearson_chi2_per_transition`.

A CHL3 candidate only passes if it reduces the q=3 wrong-sign/chi-square without
materially degrading conditional log-likelihood or the q=5/q=7 diagnostics.
