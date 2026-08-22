from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "paper_audits" / "chl2_factor_compatibility_audit.py"


def load_script():
    name = "chl2_factor_compatibility_audit_tested"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_factorial_catalog_is_complete_and_canonical() -> None:
    mod = load_script()
    observed = {spec.signature: spec.name for spec in mod.MODEL_SPECS}
    assert observed == {
        (0, 0, 0): "ROW_MASK_ETA",
        (1, 0, 0): "H2_COND_ETA",
        (0, 1, 0): "CHL1",
        (0, 0, 1): "PATH_ONLY_ETA",
        (1, 1, 0): "CG_MARKOV_ETA",
        (1, 0, 1): "CG_PATH_ETA",
        (0, 1, 1): "CHL2",
        (1, 1, 1): "CG_MARKOV_PATH_ETA",
    }


def test_factorial_effect_scaling_matches_contrast_over_four() -> None:
    mod = load_script()
    desired = {
        "H": 0.12,
        "R": -0.08,
        "P": 0.03,
        "H:R": -0.04,
        "H:P": 0.02,
        "R:P": 0.01,
        "H:R:P": -0.006,
    }
    values = {}
    for h, r, p in mod.MODEL_BY_SIGNATURE:
        xh, xr, xp = 2 * h - 1, 2 * r - 1, 2 * p - 1
        response = 1.7
        response += 0.5 * desired["H"] * xh
        response += 0.5 * desired["R"] * xr
        response += 0.5 * desired["P"] * xp
        response += 0.5 * desired["H:R"] * xh * xr
        response += 0.5 * desired["H:P"] * xh * xp
        response += 0.5 * desired["R:P"] * xr * xp
        response += 0.5 * desired["H:R:P"] * xh * xr * xp
        values[(h, r, p)] = response

    actual = mod.factorial_effects_from_values(values)
    assert actual == pytest.approx(desired, abs=1e-14)


def _synthetic_block_and_cache(mod):
    rows = []
    cache_rows = []
    g1_values = [2, 4, 6, 8, 10, 12]
    g2_values = [2, 4, 6, 8, 10, 12]
    for g1 in g1_values:
        for g2 in g2_values:
            _, _, _, adm = mod.singular_logs_for_pairs(
                np.array([g1]), np.array([g2]), [2, 3]
            )
            admissible = bool(adm[0])
            rows.append(
                {
                    "g1": g1,
                    "g2": g2,
                    "H": float(2 + ((3 * g1 + 5 * g2) % 11)) if admissible else 0.0,
                }
            )
            if admissible:
                cache_rows.append(
                    {
                        "g1": g1,
                        "g2": g2,
                        "logE_path_exclusion": -0.01 * g2 - 0.0005 * g1 * g2,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(cache_rows)


def test_end_to_end_uses_one_support_and_one_terminal_target(tmp_path: Path) -> None:
    mod = load_script()
    dataset = tmp_path / "dataset"
    blocks = dataset / "blocks"
    blocks.mkdir(parents=True)
    block_df, cache_df = _synthetic_block_and_cache(mod)
    block_path = blocks / "parent_wide_B01.csv"
    cache_path = tmp_path / "path_cache.csv"
    block_df.to_csv(block_path, index=False)
    cache_df.to_csv(cache_path, index=False)

    config_path = tmp_path / "config.generated.json"
    config_path.write_text(
        json.dumps(
            {
                "input_dir": "dataset",
                "blocks_dir": "blocks",
                "block_glob": "parent_wide_B{block:02d}.csv",
                "blocks": [1],
                "pmax": 3,
                "start_x": 1000,
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "outputs"
    rc = mod.main(
        [
            "--config",
            str(config_path),
            "--root",
            str(tmp_path),
            "--blocks",
            "1",
            "--filters",
            "ALL",
            "--Y",
            "3",
            "--path-cache-file",
            str(cache_path),
            "--workers",
            "1",
            "--hash-inputs",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert rc == 0

    metrics = pd.read_csv(output_dir / "chl2_factor_model_metrics_by_block.csv")
    support = pd.read_csv(output_dir / "chl2_factor_support_audit.csv")
    effects = pd.read_csv(output_dir / "chl2_factorial_effects_by_block.csv")
    contexts = pd.read_csv(output_dir / "chl2_factor_context_effects_by_block.csv")
    missing = pd.read_csv(output_dir / "chl2_factor_missing_cells.csv")
    cfg = json.loads(
        (output_dir / "chl2_factor_compatibility_config.json").read_text(encoding="utf-8")
    )

    assert len(metrics) == 8
    assert set(metrics["model"]) == set(mod.MODEL_BY_NAME)
    assert metrics["support_sha256"].nunique() == 1
    assert metrics["target_mean_g2"].max() - metrics["target_mean_g2"].min() < 1e-12
    assert metrics["anchor_abs_error"].max() <= 1e-9
    assert len(support) == 1
    assert bool(support.iloc[0]["common_support_gate_pass"])
    assert bool(support.iloc[0]["anchor_gate_pass"])
    assert len(effects) == 7
    assert set(effects["effect"]) == {name for name, _ in mod.FACTORIAL_EFFECTS}
    assert len(contexts) == 12
    assert missing.empty
    assert cfg["gates"]["a10_release_gate_pass"] is True
    assert cfg["provenance"]["schema"] == "chl-release-provenance@1"
    assert cfg["provenance"]["input_hashes_enabled"] is True
    assert cfg["provenance"]["script"]["sha256"]
    assert cfg["provenance"]["inputs"]["path_cache"]["sha256"]


def test_observed_triple_inadmissible_row_is_rejected() -> None:
    mod = load_script()
    block_df = pd.DataFrame({"g1": [2], "g2": [2], "H": [1.0], "max_g": [2]})
    cache = pd.DataFrame(columns=["g1", "g2", "log_P"])
    with pytest.raises(ValueError, match="observed triple-inadmissible"):
        mod.prepare_block_factors(block_df, [2, 3], cache, block=1)


def test_release_gate_rejects_missing_requested_block_filter_cell(tmp_path: Path) -> None:
    mod = load_script()
    dataset = tmp_path / "dataset"
    blocks = dataset / "blocks"
    blocks.mkdir(parents=True)
    block_df, cache_df = _synthetic_block_and_cache(mod)
    block_path = blocks / "parent_wide_B01.csv"
    cache_path = tmp_path / "path_cache.csv"
    block_df.to_csv(block_path, index=False)
    cache_df.to_csv(cache_path, index=False)
    config_path = tmp_path / "config.generated.json"
    config_path.write_text(
        json.dumps(
            {
                "input_dir": "dataset",
                "blocks_dir": "blocks",
                "block_glob": "parent_wide_B{block:02d}.csv",
                "blocks": [1],
                "pmax": 3,
                "start_x": 1000,
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "outputs"
    rc = mod.main(
        [
            "--config",
            str(config_path),
            "--root",
            str(tmp_path),
            "--blocks",
            "1",
            "--filters",
            "ALL,NO_240",
            "--Y",
            "3",
            "--path-cache-file",
            str(cache_path),
            "--workers",
            "1",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert rc == 2
    missing = pd.read_csv(output_dir / "chl2_factor_missing_cells.csv")
    cfg = json.loads(
        (output_dir / "chl2_factor_compatibility_config.json").read_text(encoding="utf-8")
    )
    assert len(missing) == 8
    assert set(missing["filter"]) == {"NO_240"}
    assert set(missing["reason"]) == {"missing_block_filter_cell"}
    assert cfg["gates"]["expected_block_filter_cells"] == 2
    assert cfg["gates"]["actual_block_filter_cells"] == 1
    assert cfg["gates"]["a10_release_gate_pass"] is False
