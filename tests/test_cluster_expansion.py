from __future__ import annotations

from chl_kernel import CHLKernel
from chl_kernel.cluster_expansion import (
    even_interior_offsets,
    omega_path_bernoulli,
    omega_second_order_lowwheel,
)


def test_even_interior_offsets():
    assert even_interior_offsets(2) == []
    assert even_interior_offsets(4) == [2]
    assert even_interior_offsets(10) == [2, 4, 6, 8]


def test_empty_lowwheel_reproduces_first_order():
    kernel = CHLKernel(Y=47, log_x=25.328436)
    g1, g2 = 6, 30
    omega_low, info = omega_second_order_lowwheel(
        g1, g2, kernel.log_x, [], kernel.singular_ratio_h4_h3
    )
    assert abs(omega_low - kernel.omega_path(g1, g2)) < 1e-12
    assert info["kappa_sum"] == 0.0


def test_lowwheel_omega_is_finite():
    kernel = CHLKernel(Y=47, log_x=25.328436)
    omega, info = kernel.omega_second_order_lowwheel(6, 30, [3])
    assert omega == omega  # not NaN
    assert "kappa_sum" in info


def test_bernoulli_dominates_poisson_for_positive_pu():
    kernel = CHLKernel(Y=47, log_x=25.328436)
    g1, g2 = 6, 30
    omega_b, _ = omega_path_bernoulli(g1, g2, kernel.log_x, kernel.singular_ratio_h4_h3)
    assert omega_b >= kernel.omega_path(g1, g2) - 1e-12
