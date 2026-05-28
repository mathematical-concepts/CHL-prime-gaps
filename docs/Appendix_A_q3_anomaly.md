# Appendix A: The $q=3$ Anomaly and Limits of the First-Order Poisson Approximation

This appendix isolates a small but important failure mode of the CHL2 kernel. The main likelihood audits show that the path-sensitive no-interior correction improves the conditional Hardy--Littlewood Markov kernel over CHL1 and over zero-memory baselines. However, the absolute prime-residue Oliver--Soundararajan diagnostic reveals that the reduced residue system modulo $3$ is not correctly reproduced by the present first-order model.

## A.1. Why row-cosine is insufficient in a two-state reduced residue space

For a modulus $q$, let

$$T_q^{\mathrm{emp}}(b,a)=\mathbb{P}_{\mathrm{emp}}(p_i\equiv a\pmod q\mid p_{i-1}\equiv b\pmod q)$$

be the empirical transition matrix on reduced prime residues, and let $T_q^{\mathrm{CHL2}}(b,a)$ be the matrix induced by the CHL2 path-exclusion kernel. In the reduced system modulo $3$, the state space contains only two classes, $\{1,2\}$. In such a two-state space, row-cosine similarity is a visually forgiving metric: two probability vectors close to $(1/2,1/2)$ can have cosine similarity exceeding $0.99$ even when the diagonal/off-diagonal bias has the wrong sign.

For this reason, the diagnostic table reports not only row-cosine and KL divergence, but also the diagonal probability and a Pearson chi-square statistic. The diagonal probability is

$$D_q(T)=\sum_b \pi_b T_q(b,b),$$

where $\pi_b$ is the empirical row mass. In the OS run, the empirical and CHL2 diagonal masses were

$$D_3(T^{\mathrm{emp}})=0.455612,\qquad D_3(T^{\mathrm{CHL2}})=0.521890,\qquad D_3(T^{\mathrm{uniform}})=0.500000.$$

Thus the empirical transition prefers changing reduced residue class modulo $3$, whereas CHL2 predicts excess persistence. This is a qualitative wrong-sign failure. The high row-cosine value in this case must therefore not be interpreted as a successful reproduction of the modulo-$3$ transition bias.

## A.2. Pearson chi-square diagnostic

Let $O_{ba}$ denote the observed transition count from $b$ to $a$, and let

$$E_{ba}=N_b\,T_q^{\mathrm{CHL2}}(b,a),\qquad N_b=\sum_a O_{ba},$$

be the expected count under the CHL2 transition matrix, row-normalized to the empirical number of transitions leaving $b$. The Pearson statistic is

$$\chi_q^2=\sum_b\sum_a\frac{(O_{ba}-E_{ba})^2}{E_{ba}},$$

with rows restricted to positive empirical mass. For rows with $m_q$ positive expected cells, the nominal degrees of freedom are summed as

$$\mathrm{df}_q=\sum_{b:N_b>0}(m_q(b)-1).$$

The normalized statistic $\chi_q^2/N$ is especially useful for comparing moduli with different state-space sizes.

For the absolute prime-residue OS diagnostic on DS1, the Pearson summaries were:

| $q$ | $N$ | $\chi^2$ | df | $\chi^2/N$ | $D_q(T^{emp})$ | $D_q(T^{CHL2})$ |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 78,934,823 | 1,389,703.35 | 2 | 0.017606 | 0.455612 | 0.521890 |
| 5 | 78,934,823 | 897,882.34 | 12 | 0.011375 | 0.192411 | 0.236270 |
| 7 | 78,934,823 | 315,442.52 | 30 | 0.003996 | 0.111342 | 0.130987 |

The modulo-$7$ case is the strongest agreement: both empirical and modeled diagonal probabilities lie below the uniform diagonal probability $1/6$, and the chi-square per transition is smallest. Modulo $5$ is intermediate: CHL2 captures the direction of repulsion but smooths its intensity. Modulo $3$ fails qualitatively because the diagonal bias crosses the uniform baseline in the wrong direction.

## A.3. Interpretation: compression of the reduced wheel modulo $3$

We interpret the modulo-$3$ anomaly as a limit of the first-order Poisson no-interior-prime approximation in a maximally compressed reduced residue system. The CHL2 path-exclusion factor is

$$E_Y^{\mathrm{path}}(g_1,g_2;x)=\exp\{-\Omega_Y^{\mathrm{path}}(g_1,g_2;x)\},$$

where

$$\Omega_Y^{\mathrm{path}}(g_1,g_2;x)=\sum_{\substack{2\le u\le g_2-2\\ u\ \mathrm{even}}}\frac{\mathfrak S_Y(\{0,g_1,g_1+u,g_1+g_2\})}{\mathfrak S_Y(\{0,g_1,g_1+g_2\})}\frac{1}{\log x}.$$

This is a first-order survival approximation: it sums the conditional Hardy--Littlewood intensity of possible interior primes and exponentiates the negative expected count. In doing so, it treats the interior-prime events as weakly dependent. That approximation is adequate as a parameter-free correction for the main conditional gap likelihood, but it is not designed to capture the strongest micro-scale modular repulsion in a two-state reduced space.

Modulo $3$, the reduced residue space has only two viable states. The local geometry is therefore highly compressed: any excess persistence or excess switching immediately dominates the entire transition matrix. In this regime, the dependence between possible interior-prime candidates is strongest. If one interior candidate is excluded, the conditional probability of the next nearby candidate is not merely adjusted by an independent Poisson survival factor; it is altered by local inclusion-exclusion structure among pairs of interior candidates.

A second-order cluster correction would introduce terms of the schematic form

$$\Omega_Y^{(2)}=\Omega_Y^{(1)}-\frac12\sum_{u\ne v}\Delta_Y(u,v\mid H_3)+\cdots,$$

where

$$\Delta_Y(u,v\mid H_3)=\frac{\mathfrak S_Y(H_5(u,v))}{\mathfrak S_Y(H_3)}-\frac{\mathfrak S_Y(H_4(u))}{\mathfrak S_Y(H_3)}\frac{\mathfrak S_Y(H_4(v))}{\mathfrak S_Y(H_3)},$$

with

$$H_3=\{0,g_1,g_1+g_2\},\quad H_4(u)=\{0,g_1,g_1+u,g_1+g_2\},$$

and

$$H_5(u,v)=\{0,g_1,g_1+u,g_1+v,g_1+g_2\}.$$

Such terms are omitted deliberately in CHL2. The present model keeps only the first-order Poisson survival factor in order to remain parameter-free, computationally feasible, and directly interpretable. The modulo-$3$ failure therefore marks a boundary of the approximation, not a hidden success.

## A.4. Consequence for the claims of the paper

The $q=3$ anomaly is not fatal to the main CHL2 claim. The main claim concerns conditional gap prediction and the path-sensitive improvement of CHL2 over CHL1 and zero-memory baselines. Those likelihood gains are stable across filters, blocks, and the evaluated truncation horizons. The anomaly does, however, show that CHL2 in its present form is structurally insufficient for micro-biases in binary reduced residue systems. Consequently, the paper treats modulo $3$ as a failure mode of the first-order model and not as positive validation.

This distinction is essential. CHL2 is a strong first-order conditional kernel for consecutive prime gaps; it is not a complete model of every low-modulus residue bias. A future CHL3-style refinement would need to test whether second-order inclusion-exclusion or cluster corrections repair the $q=3$ wrong-sign effect without degrading the parameter-free likelihood gains that make CHL2 useful.
