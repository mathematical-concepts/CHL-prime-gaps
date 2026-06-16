# Revised review of Holt survival-interval and Omega follow-up audits

This note supersedes the earlier interpretation that treated the raw excess of $\Omega_Y^{path}$ on the $r=0$ branch as a modular survival signal.

## Files reviewed

- `holt_gap_population_baseline_telemetry.json`
- `holt_survival_interval_summary.csv`
- `holt_gap_residue_population_summary.csv`
- `holt_gap_residue_population_by_interval.csv`
- `holt_omega_raw_by_residue.csv`
- `holt_omega_length_bin_detail.csv`
- `holt_omega_length_confound_decomposition.csv`
- `holt_omega_by_exact_gap.csv`

## Solid conclusions

The DS1 stream is successfully partitioned into Holt-style survival intervals $\Delta H(s)=[s^2,\operatorname{nextprime}(s)^2]$.

- nonempty intervals: 262
- triples processed: 78,934,823
- mean gaps in intervals: approximately 25.34, consistent with $\log(10^{11})$

The $q=3$ gap-residue population is stable across survival intervals:

- $P(g\equiv0\pmod3)=0.455612$
- interval-level standard deviation: approximately 0.00123

## Withdrawn interpretation

The raw comparison shows larger mean $\Omega_Y^{path}$ on $r=0$ for all audited modules. This is real, but it is not evidence of a residue-specific survival penalty.

The reason is a length confound: $r=g\bmod q$ is a deterministic function of the gap length. Branch $r=0$ is enriched in longer gaps, and $\Omega_Y^{path}$ increases with gap length.

Coarse bin controls are insufficient in this setting. For example, in the first $q=3$ bin, $g_2\in[2,13]$, the nonzero branch has mean $g_2\approx6.00$ while the $r=0$ branch has mean $g_2\approx8.65$. Thus the confound remains inside the bin.

## Exact-neighbor control

A more appropriate diagnostic compares each $r=0$ gap with neighboring nonzero gaps of nearly the same length, typically $g\pm2$.

The count-weighted excess of $\Omega_Y^{path}$ after this exact-neighbor control is tiny compared with the raw difference:

| q | raw delta | neighbor-controlled excess | share of raw |
|---:|---:|---:|---:|
| 3 | 0.117918 | 0.002660 | 2.26% |
| 5 | 0.234865 | 0.001653 | 0.70% |
| 7 | 0.323400 | -0.012283 | -3.80% |
| 11 | 0.468872 | -0.036132 | -7.71% |
| 13 | 0.614640 | 0.022133 | 3.60% |

For $q=3$, the raw excess is about 0.118, but the exact-neighbor excess is about 0.0027, roughly 2.3% of the raw effect. The signs of exact-gap excesses oscillate rather than remaining positive.

## Corrected interpretation

The survival-interval partition is useful and stable. The $\Omega$-by-residue comparison is not evidence of modular diagonal penalization. It is primarily a gap-length composition effect.

The correct statement is:

> DS1 gap-residue populations are stable in Holt-style survival intervals. However, the raw excess $\Omega_Y^{path}(r=0)>\Omega_Y^{path}(r\ne0)$ is explained by the fact that $r=0$ selects longer gaps. Exact-neighbor controls remove the apparent modular signal.

## Implication for communication with Holt

It is safe to mention the survival-interval partition and the stable $p_0(q=3)$ population. It is not safe to use the $\Omega$ comparison as evidence of a modular survival bridge.
