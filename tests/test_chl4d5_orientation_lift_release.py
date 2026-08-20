from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "paper_audits" / "chl4d5_orientation_lift_os_replacement.py"


def load_script():
    spec = importlib.util.spec_from_file_location("chl4d5_orientation_lift_release_tested", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def q3_empirical_rows(block: str, counts: list[list[float]]) -> list[dict]:
    residues = [1, 2]
    rows = []
    arr = np.asarray(counts, dtype=float)
    for i, b in enumerate(residues):
        row_count = float(arr[i].sum())
        for j, a in enumerate(residues):
            rows.append({
                "block": block,
                "q": 3,
                "from_residue": b,
                "to_residue": a,
                "empirical_count": float(arr[i, j]),
                "row_count": row_count,
                "empirical_probability": float(arr[i, j] / row_count),
            })
    return rows


def q3_model_rows(block: str, diagonal: float) -> list[dict]:
    return [
        {"block": block, "q": 3, "from_residue": 1, "to_residue": 1, "model_probability": diagonal},
        {"block": block, "q": 3, "from_residue": 1, "to_residue": 2, "model_probability": 1.0 - diagonal},
        {"block": block, "q": 3, "from_residue": 2, "to_residue": 1, "model_probability": 1.0 - diagonal},
        {"block": block, "q": 3, "from_residue": 2, "to_residue": 2, "model_probability": diagonal},
    ]


def test_q3_orientation_lift_uses_valid_edge_normalization() -> None:
    mod = load_script()
    p0 = 0.455834
    table = mod.transition_from_gap_residue_population({0: p0, 1: 0.272083, 2: 0.272083}, 3)
    _, matrix = mod.matrix_from_rows(table, 3, "probability")
    assert np.allclose(np.diag(matrix), p0)
    assert np.allclose(matrix.sum(axis=1), 1.0)
    assert mod.d3_log_odds(matrix) < 0.0


def test_all_aggregate_and_provenance_labels(tmp_path: Path) -> None:
    mod = load_script()
    empirical = pd.DataFrame(
        q3_empirical_rows("B01", [[45, 55], [54, 46]])
        + q3_empirical_rows("B02", [[46, 54], [55, 45]])
    )
    old = pd.DataFrame(q3_model_rows("B01", 0.52) + q3_model_rows("B02", 0.52))
    oriented = pd.DataFrame(
        q3_model_rows("B01", 0.455)
        + q3_model_rows("B02", 0.456)
        + q3_model_rows("ALL", 0.4555)
    )
    old_path = tmp_path / "old.csv"
    old.to_csv(old_path, index=False)

    empirical = mod.ensure_all_empirical(empirical)
    assert set(empirical["block"]) == {"B01", "B02", "ALL"}
    result = mod.compare_with_old(
        str(old_path),
        empirical,
        oriented,
        [3],
        old_model_label="absolute_prime_os_previous_gap_conditioned",
        oriented_model_label="orientation_lift_chl2_gap_population",
    )
    assert set(result["block"]) == {"B01", "B02", "ALL"}
    assert set(result["old_model_label"]) == {"absolute_prime_os_previous_gap_conditioned"}
    assert set(result["oriented_model_label"]) == {"orientation_lift_chl2_gap_population"}
    all_row = result[result["block"] == "ALL"].iloc[0]
    assert bool(all_row["old_wrong_sign"]) is True
    assert bool(all_row["oriented_wrong_sign"]) is False
