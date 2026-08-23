# Provenance of the v1.8 Table 4 legacy control

## Status

```text
A06: CLOSED — historical implementation path reconstructed.
A06a: PASS — reproducible five-modulus control retained for v2.0.0.
naive_table4_legacy: historical implementation artifact; not a scientific baseline.
```

The control used by the v2.0.0 modular audit is:

```text
absolute_prime_os_previous_gap_conditioned
```

It must not be renamed to `naive_table4_legacy`.

## Evidence base

The reconstruction used the complete reachable Git history supplied in:

```text
labv2_naive_table4_history_all.bundle
SHA-256: 7ceee50cd955a258c690aaf62b3e346c0ed88a764fcad7b49cb21a91438bd50c
```

Relevant revisions include:

```text
v1.8 release commit:
  cb1daf0f7bcb15c6e7e7f355d067edccac78cad3

initial Lab v2 orientation-lift CLI:
  c8e033734554218939244448293cd777b016872d

Lab v2 refactor requiring an upstream old matrix:
  c13e04b981d272796ab5ea998f0680b5cf338989
```

The literal name `naive_table4_legacy` was introduced later as a naming distinction. It was not the identifier of an original projector in the reachable v1.8/Lab v2 history.

## Effective v1.8 computation

### 1. The OS producer defaulted to three moduli

The v1.8 CLI declared:

```python
--os-prime-mods 3,5,7
```

The exact shell command used for the historical run was not versioned. Omitting an explicit five-modulus `--os-prime-mods` argument, or reusing a three-modulus OS CSV, produces the same downstream state.

### 2. The residual audit did not reject missing requested moduli

The residual producer was asked to construct matrices for:

```text
3,5,7,11,13
```

For each requested modulus it initialized a zero matrix and filled cells from the OS CSV. When q=11 and q=13 were absent, the script still wrote complete zero model rows to:

```text
chl4_transfer_chl2_matrices.csv
```

The effective legacy object was therefore:

```text
q=3,5,7:
  previous-gap-conditioned direct matrices

q=11,13:
  zero model rows
```

### 3. D5 converted zero rows to uniform reduced-residue rows

The v1.8 matrix reader applied this fallback:

```python
if row_sum > 0:
    normalize_row()
else:
    row[:] = 1.0 / number_of_reduced_residues
```

For prime q there are q-1 reduced residues. Consequently:

```text
q=11: diagonal = 1/(11-1) = 0.100000
q=13: diagonal = 1/(13-1) = 0.083333333...
```

This missing-modulus/zero-row fallback also reproduces the large legacy KL values printed in the v1.8 table. The match is therefore a reconstruction of the effective historical calculation, not merely a coincidence in two diagonal entries.

## What the legacy object was not

`naive_table4_legacy` was not:

- the current reproducible five-modulus previous-gap-conditioned control;
- the Lab v2 `naive_row_conditioned` projector;
- the D5 `naive_invalid` gap-population projection;
- one coherent five-modulus mathematical baseline.

The initial Lab v2 command built `OrientationLiftProjector` and a separate row-conditioned negative control from a gap-residue population. A later Lab v2 refactor stopped generating an old control and required an upstream `old_model_matrix_csv` instead.

## Historical aggregation note

The v1.8 documentation referred to `block == ALL`, but the historical D5 path produced ten block rows per modulus and no count-exact empirical `ALL` row. The printed values correspond to the unweighted arithmetic mean across B01--B10. In v2.0.0, the paper table instead uses the explicitly constructed `block=ALL` aggregate produced by the hardened A08 pipeline.

## v2.0.0 treatment

The v2.0.0 paper and generated table use:

```text
absolute_prime_os_previous_gap_conditioned
```

for all five moduli. In particular, the reproducible control diagonals are approximately:

```text
q=11: 0.056914
q=13: 0.039698
```

The legacy values `0.100000` and `0.083333` may be mentioned only in the historical provenance note.

## Regression guards

The public producers now fail when:

- a requested modulus is absent from the selected OS CSV/model;
- reduced-residue cell support is incomplete or duplicated;
- probabilities are non-finite or negative;
- a model row has zero total mass;
- an imported model matrix is not row-normalized.

A uniform fallback is permitted only when explicitly requested for a named synthetic control. Imported scientific matrices are never repaired silently.
