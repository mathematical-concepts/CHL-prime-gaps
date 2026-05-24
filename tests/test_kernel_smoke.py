from chl_kernel import CHLKernel


def test_chl_kernel_smoke():
    k = CHLKernel(Y=47, log_x=25.0)
    assert k.log_R(6, 10) == k.log_R(6, 10)
    assert k.omega_path(6, 10) >= 0
    dist = k.normalized_distribution(6, range(2, 40, 2), model="chl2_path")
    assert abs(sum(dist.values()) - 1.0) < 1e-10
