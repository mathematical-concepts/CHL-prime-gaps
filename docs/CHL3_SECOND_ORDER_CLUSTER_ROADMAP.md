# CHL3: Second-Order Cluster Corrections for Low-Wheel Prime-Residue Bias

CHL2 uses a first-order Poisson no-interior-prime correction,

\[
E^{(1)}_{\mathrm{path}}=\exp(-\Omega_Y^{(1)}),
\]

with

\[
\Omega_Y^{(1)}(g_1,g_2;x)
=
\sum_{u}
\frac{1}{\log x}
\frac{\mathfrak S_Y(H_4(u))}{\mathfrak S_Y(H_3)}.
\]

This assumes weak dependence among possible interior-prime events. The \(q=3\) prime-residue diagnostic suggests that this approximation is too crude in binary low-wheel systems.

For \(H_3=\{0,g_1,g_1+g_2\}\),

\[
H_4(u)=\{0,g_1,g_1+u,g_1+g_2\},
\]

\[
H_5(u,v)=\{0,g_1,g_1+u,g_1+v,g_1+g_2\}.
\]

Define

\[
p_u =
\frac{1}{\log x}
\frac{\mathfrak S_Y(H_4(u))}{\mathfrak S_Y(H_3)},
\]

\[
p_{uv} =
\frac{1}{(\log x)^2}
\frac{\mathfrak S_Y(H_5(u,v))}{\mathfrak S_Y(H_3)},
\]

and

\[
\kappa_{uv}=p_{uv}-p_up_v.
\]

Then a second-order cluster approximation gives

\[
\log E_{\mathrm{no\ interior}}^{(2)}
\approx
-\sum_u p_u+\sum_{u<v}\kappa_{uv}.
\]

Equivalently,

\[
\Omega_Y^{(2)}
=
\sum_u p_u-\sum_{u<v}\kappa_{uv}.
\]

The CHL3 kernel is

\[
P_Y^{CHL3}(g_2\mid g_1;x)
\propto
\mathbf 1_{\mathrm{adm}}^{(3)}
R_Y(g_2\mid g_1)
\exp[-\Omega_Y^{(2)}(g_1,g_2;x)]
e^{\eta_Y g_2}.
\]

This branch tests whether this correction improves the \(q=3\) absolute prime-residue diagnostic while preserving CHL2 likelihood and \(q=5,7\) behavior.