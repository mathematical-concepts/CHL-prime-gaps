# Research tools

## CHL2 Hit@K oracle

`chl2_hitk_oracle.py` ranks possible next gaps using the parameter-free cost

$$
\mathcal C_Y(g_2\mid g_1,x)=-\log R_Y(g_2\mid g_1)+\Omega_Y^{path}(g_1,g_2;x).
$$

Example:

```bash
python research_tools/chl2_hitk_oracle.py \
  --p-current 100000000003 \
  --g-prev 6 \
  --Y 47 \
  --gmax 2400 \
  --top-k 25
```

This is a prioritization tool, not a primality test.  It is intended to reduce
the number of classical primality checks needed in searches for rare
constellations or large gaps.

## Runtime telemetry

The Hit@K oracle can write telemetry with `--telemetry-json`. If `--output-csv` is provided and `--telemetry-json` is omitted, a sidecar file named `<output>.telemetry.json` is written automatically. Telemetry records runtime, candidate count, top-K count, wheel-mask usage, and the command line used.
