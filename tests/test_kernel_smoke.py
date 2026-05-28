"""Smoke tests for the importable CHL kernel.

This file can be run directly:

    python tests/test_kernel_smoke.py

or collected by pytest.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chl_kernel import CHLKernel, is_prime_trial_division


def test_chl_kernel_smoke() -> None:
    """Verify basic CHL2 kernel operations and normalization."""
    k = CHLKernel(Y=47, log_x=25.0)
    assert k.log_R(6, 10) == k.log_R(6, 10)
    assert k.omega_path(6, 10) >= 0
    dist = k.normalized_distribution(6, range(2, 40, 2), model="chl2_path")
    assert abs(sum(dist.values()) - 1.0) < 1e-10
    assert is_prime_trial_division(101)
    assert not is_prime_trial_division(102)


if __name__ == "__main__":
    test_chl_kernel_smoke()
    print("CHL kernel smoke test passed.")
