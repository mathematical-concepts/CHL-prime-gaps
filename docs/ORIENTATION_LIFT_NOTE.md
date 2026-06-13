# Orientation-lifted modular diagnostics

The earlier direct Oliver--Soundararajan-style diagnostic suggested a wrong-sign anomaly in modulus $3$. Later audits showed that this was not a failure of the CHL2 gap kernel. CHL2 already reproduces the aggregate gap-residue population $P(g \bmod q)$ accurately.

The missing step was the orientation lift from gap residues to prime-residue transition matrices.

For a modulus $q$, let

$$p_r=P(g\equiv r\pmod q).$$

A gap residue $r$ induces directed edges

$$b\to b+r\pmod q$$

only when both $b$ and $b+r$ are reduced residue classes. Define

$$N_r(q)=\#\{b\in(\mathbb Z/q\mathbb Z)^*:b+r\in(\mathbb Z/q\mathbb Z)^*\}.$$

The orientation lift puts mass $p_r/N_r(q)$ on each valid edge and then normalizes row-wise.

For $q=3$ this gives

$$T(1,1)\propto p_0/2,\qquad T(1,2)\propto p_1,$$

$$T(2,2)\propto p_0/2,\qquad T(2,1)\propto p_2.$$

This factor $p_0/2$ removes the false diagonal-persistence signal produced by the old direct lift.

The orientation lift is a diagnostic correction. It does not alter the CHL2 conditional gap-survival kernel.
