# CHL3 Möbius strict low-wheel patch

This patch is intended for the experimental CHL3 branch. It makes the low-wheel
second-order correction auditable against the q=3 anomaly checklist.

## Main fixes

1. Adds explicit hard-zero predicates:
   - `lowwheel_covers_all_classes(residues, p)`
   - `lowwheel_is_inadmissible(residues, low_primes)`

2. Preserves the Möbius hard zero for H5:

   If `H5={0,g1,g1+u,g1+v,g1+g2}` occupies every class modulo any selected low
   prime, then `S_low(H5)=0`, the coupling is forced to zero, and the cumulant
   contribution is exactly `-p_u p_v` on the aggregated residue class.

3. Keeps unordered-pair convention:

   The implementation sums over residue-class pairs `r <= s`, corresponding to
   unordered interior pairs `u < v`. Therefore no extra factor `1/2` is applied.

4. Fixes the OS transition CSV reporting bug:

   `expected_count_model = row_count * model_probability`.

## What this patch does not claim

This patch does not guarantee that CHL3 will fix the q=3 sign. It only ensures
that the implementation does not dilute hard low-wheel zeros or misreport the
OS expected counts.
