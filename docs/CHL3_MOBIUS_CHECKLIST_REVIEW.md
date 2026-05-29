# CHL3 Möbius checklist review

## Exact cumulant

For unordered interior candidates $u < v$, the intended second-order cumulant is

$$
\kappa_{uv} = p_{uv} - p_u p_v .
$$


where

$$
p_u = \frac{1}{\log x}\,\frac{\mathfrak{S}_{Y}(H_4(u))}{\mathfrak{S}_{Y}(H_3)},\qquad 
p_{uv} = \frac{1}{(\log x)^2}\,\frac{\mathfrak{S}_{Y}(H_5(u,v))}{\mathfrak{S}_{Y}(H_3)} .
$$

The quintet must be

$$
H_5(u,v) = \{0,\; g_1,\; g_1+u,\; g_1+v,\; g_1+g_2\}.
$$

The second-order intensity is

$$
\Omega_Y^{(2)} = \Omega_Y^{(1)} - \sum_{u < v} \kappa_{uv}.
$$

Because the sum is over unordered pairs, no additional factor $1/2$ is used.

## Low-wheel hard zero

If $H_5(u,v)$ covers all residue classes modulo any selected low prime, then

$$
\mathfrak{S}_{\mathrm{low}}(H_5) = 0.
$$

Therefore the low-wheel coupling is zero and

$$
\kappa_{uv} = -p_u p_v .
$$

This negative cumulant increases $\Omega^{(2)}$ and decreases the survival probability.

## Hybrid design

The intended CHL3 low-wheel design is hybrid:

- one-point intensities $p_u$ are still full CHL2 ratios of type $H_4 / H_3$;
- pairwise covariance is corrected only through selected low primes, for example $\{3\}$, $\{3,5\}$, or $\{3,5,7\}$;
- the hard-zero condition for these low primes is checked explicitly.

This avoids an exponential Möbius expansion while preventing the $q=3$ hard zeros from being averaged away.