# CHL3: Second-Order Cluster Corrections for Low-Wheel Prime-Residue Bias

CHL3 is an experimental continuation of CHL2. It is not part of the stable
CHL2 v1.4 release. The goal is narrow: test whether second-order low-wheel
cluster corrections reduce the absolute prime-residue anomaly at `q=3` without
breaking the CHL2 conditional likelihood gains or the `q=5,7` diagnostics.

## CHL2 baseline

For

\[
H_3=\{0,g_1,g_1+g_2\},
\]

CHL2 uses the first-order path-sensitive no-interior correction

\[
\Omega_Y^{(1)}(g_1,g_2;x)=
\sum_{2\le u\le g_2-2\atop u\ \mathrm{even}}
\frac{1}{\log x}
\frac{\mathfrak S_Y(\{0,g_1,g_1+u,g_1+g_2\})}{\mathfrak S_Y(H_3)}.
\]

The corresponding survival factor is

\[
E_Y^{(1)}=\exp[-\Omega_Y^{(1)}].
\]

This treats possible interior-prime events as weakly dependent. The `q=3`
prime-residue diagnostic suggests that this approximation is too coarse in the
binary reduced-residue system.

## CHL3 second-order cluster correction

Define

\[
H_4(u)=\{0,g_1,g_1+u,g_1+g_2\},
\]

\[
H_5(u,v)=\{0,g_1,g_1+u,g_1+v,g_1+g_2\}.
\]

Let

\[
p_u=rac{1}{\log x}\frac{\mathfrak S_Y(H_4(u))}{\mathfrak S_Y(H_3)},
\]

\[
p_{uv}=\frac{1}{(\log x)^2}\frac{\mathfrak S_Y(H_5(u,v))}{\mathfrak S_Y(H_3)},
\]

and

\[
\kappa_{uv}=p_{uv}-p_up_v.
\]

A second-order cluster approximation gives

\[
\log E_Y^{(2)}\approx -\sum_u p_u + \sum_{u<v}\kappa_{uv},
\]

or equivalently

\[
\Omega_Y^{(2)}=\sum_u p_u-\sum_{u<v}\kappa_{uv}.
\]

CHL3 tests

\[
P_Y^{\mathrm{CHL3}}(g_2\mid g_1;x)\propto
\mathbf 1_{\rm adm}^{(3)}(g_1,g_2;Y)
R_Y(g_2\mid g_1)
\exp[-\Omega_Y^{(2)}(g_1,g_2;x)]e^{\eta_Yg_2}.
\]

## Low-wheel practical approximation

The production candidate does not evaluate all \(H_5(u,v)\) pairs. Instead,
it keeps the full CHL2 single-event intensities \(p_u\), but approximates
cross-dependence by low-prime local cumulants over preregistered low-wheel
sets:

- `{3}`
- `{3,5}`
- `{3,5,7}`

This produces the models:

- `CHL3_lowwheel2_p3_cond_eta`
- `CHL3_lowwheel2_p3p5_cond_eta`
- `CHL3_lowwheel2_p3p5p7_cond_eta`

## PASS rule

A CHL3 candidate is not considered a CHL2 replacement unless it:

1. reduces the `q=3` wrong-sign / Pearson chi-square diagnostic;
2. preserves the qualitative `q=5` and `q=7` repulsion directions;
3. does not materially degrade conditional log-likelihood relative to
   `CHL2_path_excl_cond_eta`;
4. remains stable under the same truncation-horizon sensitivity checks used for
   CHL2.

CHL3 is therefore a preregistered experimental branch, not a silent revision of
CHL2.
