# CHL3: Second-order cluster corrections for low-wheel prime-residue bias

CHL2 uses a first-order Poisson no-interior-prime correction,

$$
E_{\mathrm{path}}^{(1)} = \exp\!\left(-\Omega_Y^{(1)}\right),
$$

with

$$
\Omega_Y^{(1)}(g_1,g_2;x) = \sum_u \frac{1}{\log x}\,\frac{\mathfrak{S}_{Y}(H_4(u))}{\mathfrak{S}_{Y}(H_3)}.
$$

This assumes weak dependence among possible interior-prime events. The $q=3$ prime-residue diagnostic suggests that this approximation is too crude in binary low-wheel systems.

For

$$
H_3 = \{0,\; g_1,\; g_1+g_2\},
$$

we define

$$
H_4(u) = \{0,\; g_1,\; g_1+u,\; g_1+g_2\},
$$

and

$$
H_5(u,v) = \{0,\; g_1,\; g_1+u,\; g_1+v,\; g_1+g_2\}.
$$

The conditional one-point and two-point interior intensities are

$$
p_u = \frac{1}{\log x}\,\frac{\mathfrak{S}_{Y}(H_4(u))}{\mathfrak{S}_{Y}(H_3)},
$$

and

$$
p_{uv} = \frac{1}{(\log x)^2}\,\frac{\mathfrak{S}_{Y}(H_5(u,v))}{\mathfrak{S}_{Y}(H_3)}.
$$

The pair cumulant is

$$
\kappa_{uv} = p_{uv} - p_u p_v.
$$

A second-order cluster approximation gives

$$
\log E_{\text{no interior}}^{(2)} \approx -\sum_u p_u + \sum_{u < v} \kappa_{uv}.
$$

Equivalently,

$$
\Omega_Y^{(2)} = \sum_u p_u - \sum_{u < v} \kappa_{uv}.
$$

The CHL3 kernel is

$$
P_Y^{\mathrm{CHL3}}(g_2 \mid g_1; x) \propto \mathbf{1}_{\mathrm{adm}}^{(3)}(g_1,g_2;Y)\; R_Y(g_2 \mid g_1)\; \exp\!\left[-\Omega_Y^{(2)}(g_1,g_2;x)\right] e^{\eta_Y g_2}.
$$

This branch tests whether this correction improves the $q=3$ absolute prime-residue diagnostic while preserving CHL2 likelihood and the behavior in the $q=5$ and $q=7$ diagnostics.