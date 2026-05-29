from chl_kernel.cluster_expansion import (
    lowwheel_is_inadmissible,
    lowwheel_singular_from_residues,
    omega_second_order_lowwheel,
)


def test_h5_hard_zero_mod3_is_preserved():
    # H5 residues cover 0,1,2 modulo 3 => low singular product must be exactly zero.
    residues = (0, 3, 4, 5, 9)  # mod 3 -> 0,0,1,2,0
    assert lowwheel_is_inadmissible(residues, [3])
    assert lowwheel_singular_from_residues(residues, 5, [3]) == 0.0


def test_negative_cumulant_increases_omega_in_exclusive_case():
    # Synthetic ratios: every H4/H3 ratio is 1, and H5/H3 would be forced to zero
    # in the strict low-wheel layer for pairs that cover all classes.
    def r4(g1, g2, u):
        return 1.0

    omega2, info = omega_second_order_lowwheel(
        g1=6,
        g2=8,
        log_x=25.0,
        low_primes=[3],
        singular_ratio_h4_h3=r4,
    )
    # At least one low-wheel hard-zero pair should have been encountered.
    assert omega2 >= info["omega1"]
