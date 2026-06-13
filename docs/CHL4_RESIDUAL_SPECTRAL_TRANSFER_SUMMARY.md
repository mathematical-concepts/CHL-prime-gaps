# CHL4 Residual Spectral Transfer — Anchor Note

**Status:** CHL4-A and CHL4-C validated; CHL4-B local LOS projection falsified in the audited truncation family.  
**Purpose:** This note records the current state of the direct modular transfer branch so that the project can be resumed without losing the logic of the experiments.

> CHL2 is an interior-survival kernel for consecutive prime gaps.  
> CHL4 studies the direct modular transfer residual left by CHL2 in the transition matrix of prime residue classes.

The key object is not a gap probability, but the transition matrix

$$T_q(b,a;x)=P(p_n\equiv a \pmod q \mid p_{n-1}\equiv b \pmod q),$$

where $a,b$ range over reduced residue classes modulo $q$.

---

## 1. Context: why CHL4 exists

CHL2 successfully models the conditional survival of a candidate next gap by combining:

$$\text{conditional Hardy--Littlewood endpoint ratio} + \text{no-interior-prime survival}.$$

However, the absolute Oliver--Soundararajan residue diagnostic exposed a persistent failure in the compressed binary system modulo $3$.

For $q=3$, define the diagonal log-odds statistic

$$D_3(T)=\log\left(\frac{T(1,1)T(2,2)}{T(1,2)T(2,1)}\right).$$

Interpretation:

- $D_3<0$: diagonal repulsion; consecutive primes tend to change reduced residue class.
- $D_3>0$: diagonal persistence; consecutive primes tend to repeat reduced residue class.

Empirically, DS1 shows $D_3^{emp}<0$, but CHL2 predicts $D_3^{CHL2}>0$. Therefore the residual

$$D_3^{res}=D_3^{emp}-D_3^{CHL2}$$

is stable and negative.

---

## 2. CHL4-A: direct modular residual audit

CHL4-A measured the row-centered residual transfer matrix

$$\mathcal R_q(b,a;x)=\log\frac{T_q^{emp}(b,a;x)+\varepsilon}{T_q^{CHL2}(b,a;x)+\varepsilon},$$

and then removed row offsets:

$$\widetilde{\mathcal R}_q(b,a;x)=\mathcal R_q(b,a;x)-\frac{1}{\varphi(q)}\sum_c \mathcal R_q(b,c;x).$$

### 2.1 Clean block alignment

The final aligned CHL4-A run used the real `parent_wide_B01` through `parent_wide_B10` block boundaries, not artificial fixed-size prime-stream chunks. This removed the earlier artificial `B11` tail artifact.

Expected block status:

```text
B01 ... B10 only
No B11
No TAIL_PARTIAL included in stability summaries
```

### 2.2 Main q=3 result

Across the ten real DS1 blocks:

| quantity | value |
|---|---:|
| mean $D_3^{emp}$ | $-0.356043$ |
| mean $D_3^{CHL2}$ | $+0.175274$ |
| mean $D_3^{res}$ | $-0.531318$ |
| std of $D_3^{res}$ | $0.001312$ |
| mean empirical diagonal probability | $0.455612$ |
| mean CHL2 diagonal probability | $0.521895$ |
| CHL2 wrong sign | $10/10$ blocks |

Thus:

$$D_3^{emp}<0, \qquad D_3^{CHL2}>0, \qquad D_3^{res}<0,$$

stably across B01--B10.

**Interpretation:** CHL2 explains interior survival of gaps, but it leaves a direct modular-transfer residual in the binary modulo $3$ system.

---

## 3. Character-spectrum diagnosis

CHL4-A decomposed the row-centered residual using Dirichlet characters over the reduced residue group:

$$\widehat{\mathcal R}_q(\chi,\psi;x)=\frac{1}{\varphi(q)^2}\sum_{a,b}\widetilde{\mathcal R}_q(b,a;x)\overline{\chi(a)}\psi(b).$$

For $q=3$, there is only one non-principal character. The diagonal/off-diagonal residual collapses almost entirely onto that mode.

### 3.1 Control spectrum by modulus

| $q$ | dominant non-principal energy fraction | effective rank |
|---:|---:|---:|
| $3$ | $0.999964$ | $1.000398$ |
| $5$ | $0.326995$ | $3.793258$ |
| $7$ | $0.191432$ | $7.051608$ |
| $11$ | $0.058106$ | $36.558789$ |
| $13$ | $0.041346$ | $48.124459$ |

**Conclusion:** the $q=3$ residual is essentially rank one. For $q\ge 5$, the residual becomes progressively higher-dimensional and dispersed.

This is the main structural discovery of CHL4-A/C.

---

## 4. CHL4-B: local Oliver--Soundararajan projection audit

CHL4-B tested whether a truncated local Oliver--Soundararajan / Hardy--Littlewood finite-cluster projection could explain the $q=3$ residual.

The audited projection family was denoted

$$\Theta_{q,2,H,Y}^{LOS,\Pi},$$

with explicit cutoffs:

- truncation horizon $Y\in\{31,47,61,73\}$;
- gap cutoff $H_{\max}\in\{120,240,400,800,1200,2400\}$;
- low-wheel sets $\{3\}$, $\{3,5\}$, and $\{3,5,7\}$;
- base weight `cg_excl`;
- projection mode `lowwheel2`.

This gave $72$ audited configurations.

### 4.1 Result

All audited projections retained the wrong qualitative sign:

```text
negative_D3_count = 0 / 72
los_sign_matches_empirical = False in 72 / 72 cases
```

The empirical target was

$$D_3^{emp}=-0.356043,$$

while the projected LOS values remained positive, approximately in the range

$$0.804369 \le D_3^{LOS,\Pi} \le 0.806590.$$

The projected term moved in the correct direction,

$$D_{\theta}=D_{LOS}-D_{base}<0,$$

but only by about

$$\Delta D_{LOS}\approx -0.005,$$

whereas the CHL2 residual requires a correction of approximately

$$-0.531.$$

### 4.2 Strategic conclusion

CHL4-B is a clean negative result:

> The truncated local LOS projection detects the correct character mode but with insufficient amplitude; it does not explain the $q=3$ diagonal-repulsion anomaly.

This does **not** refute Hardy--Littlewood or Oliver--Soundararajan. It falsifies this specific local truncated projection as an explanation of the binary transfer residual.

---

## 5. CHL4-C: residual spectral transfer

After CHL4-B failed, CHL4-C treated the $q=3$ residual as a direct rank-one transfer mode.

Define

$$K_3(b,a)=\begin{cases}1,&a=b,\\-1,&a\ne b.\end{cases}$$

The corrected diagnostic kernel is

$$T_3^{corr}(b,a)\propto T_3^{CHL2}(b,a)\exp(\theta_3 K_3(b,a)).$$

The minimal residual coefficient is

$$\theta_3=\frac{D_3^{emp}-D_3^{CHL2}}{4}.$$

### 5.1 Measured rigidity of $\theta_3$

| statistic | value |
|---|---:|
| mean $\theta_3$ | $-0.132829$ |
| weighted mean $\theta_3$ | $-0.132829$ |
| std | $0.000328$ |
| SEM | $0.000104$ |
| min | $-0.133286$ |
| max | $-0.132242$ |
| negative count | $10/10$ blocks |
| positive count | $0/10$ blocks |

Thus:

$$\theta_3<0$$

in every real DS1 block.

### 5.2 Holdout validation

The holdout protocol fit $\theta_3$ on B01--B05 and tested it on B06--B10.

| holdout quantity | value |
|---|---:|
| fitted $\theta_3$ from B01--B05 | $-0.132599$ |
| corrected sign matches empirical | $5/5$ test blocks |
| mean corrected diagonal probability | $0.455726$ |
| mean empirical diagonal probability in test half | approximately $0.455497$ |

The fitted residual transfer coefficient learned on the first half corrects the sign and nearly the intensity on the second half.

### 5.3 Interpretation

CHL4-C identifies a stable rank-one residual transfer mode in the modulo $3$ prime-residue transition matrix. It is not yet an arithmetic derivation of $\theta_3$; it is a measurement of a rigid residual mode.

---

## 6. Current scientific status

### Established

1. CHL2 is a conditional interior-survival kernel for consecutive prime gaps.
2. CHL2 leaves a wrong-sign residual in the binary prime-residue transfer matrix modulo $3$.
3. The $q=3$ residual is stable across B01--B10.
4. The residual is spectrally rank one, concentrated in the non-principal character mode.
5. A truncated local LOS projection moves in the right direction but is two orders of magnitude too small.
6. A direct residual spectral coefficient $\theta_3\approx -0.132829$ corrects the transfer sign and passes holdout.

### Not established

1. No arithmetic first-principles derivation of $\theta_3$ has been found yet.
2. CHL4-C is not yet a replacement for CHL2.
3. The measured residual coefficient is not yet connected to Dirichlet $L$-functions, prime races, or a closed-form sieve term.
4. The LOS projection tested so far is not the full LOS theory; it is a finite local projection.

---

## 7. Recommended next phase

The next phase should be named internally:

```text
CHL4-D_ARITHMETIC_ORIGIN_OF_THETA3
```

External title suggestion:

```text
Arithmetic Origin of a Rank-One Modular Transfer Residual in Consecutive Prime Residues
```

The objective is not to remeasure $\theta_3$, but to explain it.

Candidate directions:

1. character-theoretic derivation of the diagonal/off-diagonal mode;
2. comparison with Oliver--Soundararajan secondary terms under alternative projections;
3. low-modulus transfer laws independent of interior-survival exclusion;
4. relation to Dirichlet characters and, later, possible $L(s,\chi)$ terms;
5. finite-scale prime-residue transfer as a separate object from gap survival.

---

## 8. Decision log

### CHL4-A

```text
PASS.
Residual modular transfer exists and is stable.
```

### CHL4-B

```text
FAIL as explanation.
Local truncated LOS projection detects the correct mode but has insufficient amplitude.
```

### CHL4-C

```text
PASS.
Rank-one residual coefficient theta_3 is stable, negative, and holdout-valid.
```

### Next step

```text
Do not force higher local cluster order.
Do not fold theta_3 back into CHL2.
Study the arithmetic origin of direct modular transfer as a separate problem.
```

---

## 9. Markdown math rendering rules

For repository Markdown files, use:

```text
Display math: $$...$$
Inline math: $...$
Do not use bracket-style LaTeX math delimiters.
Do not use parenthesis-style LaTeX math delimiters.
Write inequalities with spaces around the relation symbol, for example `$u < v$`.
Avoid hyphens directly adjacent to $...$ math spans.
```

