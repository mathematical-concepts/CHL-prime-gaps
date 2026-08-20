from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_a07_interpretation_reports_observed_result(tmp_path: Path) -> None:
    module = load_script(
        "test_chl4_modular_transfer_residual_audit",
        "paper_audits/chl4_modular_transfer_residual_audit.py",
    )
    q3 = pd.DataFrame(
        {
            "block": ["B01", "B02"],
            "D_empirical": [-0.3, -0.31],
            "D_chl2": [0.2, 0.2],
            "D_residual_emp_minus_chl2": [-0.5, -0.51],
            "diagonal_probability_empirical": [0.45, 0.451],
            "diagonal_probability_chl2": [0.52, 0.52],
            "wrong_sign_chl2": [True, True],
        }
    )
    spectrum = pd.DataFrame(
        {
            "block": ["B01", "B01", "B02", "B02"],
            "q": [3, 3, 3, 3],
            "chi_index": [0, 1, 0, 1],
            "psi_index": [0, 1, 0, 1],
            "energy_fraction": [0.0, 0.999, 0.0, 0.998],
            "effective_rank": [1.001, 1.001, 1.002, 1.002],
        }
    )

    report = module.write_interpretation(tmp_path, q3, spectrum)
    text = (tmp_path / "chl4_interpretacion.md").read_text(encoding="utf-8")

    assert report["q3_full_blocks"] == 2
    assert report["q3_wrong_sign_count"] == 2
    assert report["q3_dominant_mode"] == {"chi_index": 1, "psi_index": 1}
    assert "## Next step" not in text
    assert "separately generated CHL4-D5" in text


def test_d5_interpretation_is_result_driven(tmp_path: Path) -> None:
    module = load_script(
        "test_chl4d5_orientation_lift_os_replacement",
        "paper_audits/chl4d5_orientation_lift_os_replacement.py",
    )
    summary = pd.DataFrame(
        {
            "block": ["ALL"],
            "q": [3],
            "diagonal_probability_empirical": [0.4556],
            "diagonal_probability_model": [0.4559],
            "uniform_diagonal_probability": [0.5],
            "diagonal_wrong_sign_vs_uniform": [False],
            "D3_empirical": [-0.356],
            "D3_model": [-0.354],
        }
    )
    old_vs_new = pd.DataFrame(
        {
            "block": ["B01", "ALL", "B01", "ALL"],
            "q": [3, 3, 5, 5],
            "old_kl": [0.01, 0.01, 0.005, 0.005],
            "oriented_kl": [1e-6, 1e-6, 0.001, 0.001],
            "old_l1": [0.1, 0.1, 0.05, 0.05],
            "oriented_l1": [0.001, 0.001, 0.01, 0.01],
            "old_wrong_sign": [True, True, False, False],
            "oriented_wrong_sign": [False, False, False, False],
            "delta_kl_old_minus_oriented": [0.009999, 0.009999, 0.004, 0.004],
            "emp_diag": [0.4556, 0.4556, 0.19, 0.19],
            "old_diag_model": [0.52, 0.52, 0.23, 0.23],
            "oriented_diag_model": [0.4559, 0.4559, 0.192, 0.192],
        }
    )

    outcome = module.write_interpretation(tmp_path, summary, old_vs_new)
    text = (tmp_path / "chl4d5_interpretacion.md").read_text(encoding="utf-8")

    assert outcome["replacement_gate_pass"] is True
    assert outcome["kl_improved_count"] == 4
    assert outcome["l1_improved_count"] == 4
    assert outcome["q3_old_wrong_sign_count"] == 2
    assert outcome["q3_oriented_wrong_sign_count"] == 0
    assert "If the oriented matrix" not in text
    assert "The replacement gate passes" in text


def test_provenance_hashes_script_and_declared_inputs(tmp_path: Path) -> None:
    module = load_script(
        "test_chl4d5_provenance",
        "paper_audits/chl4d5_orientation_lift_os_replacement.py",
    )
    source = tmp_path / "input.txt"
    source.write_bytes(b"abc")

    provenance = module.build_provenance(
        repo_root=tmp_path,
        input_paths={"sample": source},
        hash_inputs=True,
    )

    assert provenance["schema"] == "chl-release-provenance@1"
    assert provenance["script"]["sha256"]
    assert provenance["inputs"]["sample"]["sha256"] == hashlib.sha256(b"abc").hexdigest()
