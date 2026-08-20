from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


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
