from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "paper_audits" / "chl4_modular_transfer_residual_audit.py"


def load_script():
    spec = importlib.util.spec_from_file_location("chl4_modular_transfer_residual_audit_tested", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parentwide_skip_preserves_original_event_coordinates(tmp_path: Path) -> None:
    mod = load_script()
    prime_csv = tmp_path / "primes.csv"
    # Six planned event positions. Five adjacent-prime transitions exist; after
    # removing the first transition, four triple-compatible transitions remain.
    pd.DataFrame({"prime": [5, 7, 11, 13, 17, 19]}).to_csv(prime_csv, index=False)
    plan = mod.BlockPlan(
        labels=["B01", "B02"],
        counts=[3, 3],
        mode="parent-wide",
        source_files=[],
        total_expected_transitions=6,
    )

    outputs = []
    for chunksize in (2, 100):
        counts, meta = mod.empirical_counts_from_prime_csv(
            prime_csv,
            [3],
            chunksize=chunksize,
            workers=1,
            block_plan=plan,
            drop_partial_blocks=True,
            skip_first_transition=True,
        )
        outputs.append({key: int(matrix.sum()) for key, matrix in counts.items()})
        assert meta["raw_adjacent_transitions_seen"] == 5
        assert meta["candidate_transitions_after_skip"] == 4
        assert meta["skipped_initial_transition"] == 1
        assert meta["block_plan_observed_counts"] == [2, 2]
        assert meta["block_plan_count_deltas"] == [-1, -1]
        assert meta["block_plan_endpoint_adjusted_expected_counts"] == [2, 2]
        assert meta["block_alignment_matches_endpoint_adjusted_plan"] is True

    assert outputs[0] == outputs[1] == {("B01", 3): 2, ("B02", 3): 2}


def _q3_os_rows(*, row2_zero: bool = False) -> list[dict]:
    rows: list[dict] = []
    for b in [1, 2]:
        for a in [1, 2]:
            probability = 0.0 if row2_zero and b == 2 else 0.5
            rows.append({
                "q": 3,
                "model": "CHL2_path_excl_cond_eta",
                "from_residue": b,
                "to_residue": a,
                "model_probability": probability,
                "row_count": 100,
            })
    return rows


def test_os_model_support_rejects_missing_requested_modulus(tmp_path: Path) -> None:
    mod = load_script()
    os_csv = tmp_path / "os_missing_q5.csv"
    pd.DataFrame(_q3_os_rows()).to_csv(os_csv, index=False)

    with pytest.raises(ValueError, match=r"missing requested moduli \[5\]"):
        mod.build_model_matrices_from_os_csv(
            os_csv,
            [3, 5],
            model="CHL2_path_excl_cond_eta",
        )


def test_os_model_support_rejects_zero_sum_rows(tmp_path: Path) -> None:
    mod = load_script()
    os_csv = tmp_path / "os_zero_row.csv"
    pd.DataFrame(_q3_os_rows(row2_zero=True)).to_csv(os_csv, index=False)

    with pytest.raises(ValueError, match=r"zero-sum model rows.*q=3"):
        mod.validate_os_model_support(
            os_csv,
            [3],
            model="CHL2_path_excl_cond_eta",
        )


def test_os_model_support_audit_records_complete_requested_moduli(tmp_path: Path) -> None:
    mod = load_script()
    os_csv = tmp_path / "os_complete.csv"
    pd.DataFrame(_q3_os_rows()).to_csv(os_csv, index=False)

    audit = mod.validate_os_model_support(
        os_csv,
        [3],
        model="CHL2_path_excl_cond_eta",
    )

    assert audit["requested_mods"] == [3]
    assert audit["available_mods"] == [3]
    assert audit["missing_requested_mods"] == []
    assert audit["zero_sum_model_rows"] == []
    assert audit["model_support_gate_pass"] is True
    assert audit["model_support_expected_cells"] == 4
    assert audit["model_support_observed_cells"] == 4
