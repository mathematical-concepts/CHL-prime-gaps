"""CHL prime-gap kernel package.

Import examples
---------------
>>> from chl_kernel import CHLKernel
>>> kernel = CHLKernel(Y=47, log_x=25.328436)
>>> kernel.log_R(6, 10)
>>> kernel.hit_cost(6, 10)
"""
from .primes import primes_upto, reduced_residues, is_prime_trial_division
from .singular_series import SingularSeriesCache, is_admissible, nu_mod_q
from .models import CHLKernel, even_candidates, survives_actual_wheel
from .residue_transfer import (
    row_centered_log_residual,
    diagonal_log_odds_q3,
    character_spectrum,
    dirichlet_character_table_prime_modulus,
)

__all__ = [
    "CHLKernel",
    "SingularSeriesCache",
    "even_candidates",
    "is_admissible",
    "nu_mod_q",
    "primes_upto",
    "is_prime_trial_division",
    "reduced_residues",
    "survives_actual_wheel",
    "row_centered_log_residual",
    "diagonal_log_odds_q3",
    "character_spectrum",
    "dirichlet_character_table_prime_modulus",
]
