from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _make_outputs(root: Path) -> None:
    # A02
    a02 = root / "v2_release_chl2_main"
    summary = []
    metrics = []
    for filt, n, gain in [("ALL", 1000, 0.001), ("LOW_ONLY_LE58", 700, 0.0008), ("NO_240", 20, -0.0002)]:
        for model, ll in [
            ("CHL1_ratio_only_cond_eta", -3.2),
            ("CHL2_path_excl_cond_eta", -3.2 + gain),
        ]:
            summary.append(
                {
                    "filter": filt,
                    "model": model,
                    "n_events": n,
                    "loglik_per_event": ll,
                    "conditional_kl": 0.001,
                }
            )
            for block in [1, 2]:
                metrics.append(
                    {
                        "filter_name": filt,
                        "model": model,
                        "block": block,
                        "n_events": n / 2,
                        "loglik_sum": (n / 2) * ll,
                    }
                )
    _write_csv(a02 / "chl2_conditional_summary.csv", summary)
    _write_csv(a02 / "chl2_metrics_by_block.csv", metrics)
    _write_csv(
        a02 / "chl2_memory_irreducibility.csv",
        [
            {
                "filter": "ALL",
                "order1_model": "CHL2_path_excl_cond_eta",
                "baseline": "Cramer_order0_exp",
                "n_events": 1000,
                "delta_loglik_per_event_total": 0.25,
                "block_z": 12.0,
                "n_blocks": 2,
            }
        ],
    )

    # A04
    a04 = root / "v2_release_chl2_y_sweep"
    _write_csv(
        a04 / "chl2_y_sweep_gains.csv",
        [
            {
                "Y": y,
                "filter": filt,
                "model": "CHL2_path_excl_cond_eta",
                "baseline": "CHL1_ratio_only_cond_eta",
                "delta_loglik_model_minus_baseline": value,
            }
            for filt, values in [("ALL", [0.0009, 0.0010]), ("LOW_ONLY_LE58", [0.0007, 0.0008]), ("NO_240", [-0.0002, -0.00015])]
            for y, value in zip([31, 47], values)
        ],
    )

    # A05
    a05 = root / "v2_release_chl4d2_gap_population_bias"
    _write_csv(
        a05 / "chl4d2_scale_wave_summary.csv",
        [
            {
                "filter": filt,
                "empirical_events": events,
                "event_share_vs_ALL": share,
                "empirical_diag_probability": diag,
                "D3_empirical_gap_population": d_emp,
                "D3_chl2_gap_population": d_model,
                "D3_residual_gap_population": d_emp - d_model,
            }
            for filt, events, share, diag, d_emp, d_model in [
                ("ALL", 1000, 1.0, 0.45, -0.35, -0.34),
                ("LOW_ONLY_LE58", 700, 0.7, 0.44, -0.45, -0.44),
                ("NO_240", 1, 0.000001, 0.51, 0.04, 0.03),
            ]
        ],
    )

    # A07
    a07 = root / "v2_release_chl4_residual_blocks_aligned"
    _write_csv(
        a07 / "chl4_q3_diagonal_scalar.csv",
        [
            {
                "block": f"B{block:02d}",
                "q": 3,
                "D_empirical": -0.35,
                "D_chl2": 0.17,
                "D_residual_emp_minus_chl2": -0.52,
                "diagonal_probability_empirical": 0.455,
                "diagonal_probability_chl2": 0.522,
                "wrong_sign_chl2": True,
            }
            for block in [1, 2]
        ],
    )
    _write_csv(
        a07 / "chl4_block_stability.csv",
        [{"diagnostic": "q3_D_residual", "n_blocks": 2, "mean": -0.52, "min": -0.53, "max": -0.51}],
    )

    # A08
    a08 = root / "v2_release_chl4d5_orientation_lift_os_replacement"
    _write_csv(
        a08 / "chl2_os_old_direct_vs_oriented.csv",
        [
            {
                "block": "ALL",
                "q": q,
                "old_model_label": "absolute_prime_os_previous_gap_conditioned",
                "oriented_model_label": "orientation_lift_chl2_gap_population",
                "old_kl": old_kl,
                "oriented_kl": new_kl,
                "delta_kl_old_minus_oriented": old_kl - new_kl,
                "old_l1": 0.1,
                "oriented_l1": 0.001,
                "old_diag_model": old_diag,
                "oriented_diag_model": oriented_diag,
                "emp_diag": emp_diag,
                "old_wrong_sign": q == 3,
                "oriented_wrong_sign": False,
            }
            for q, old_kl, new_kl, old_diag, oriented_diag, emp_diag in [
                (3, 0.0088, 0.0000004, 0.522, 0.456, 0.4556),
                (5, 0.0059, 0.0003, 0.236, 0.192, 0.1924),
            ]
        ],
    )

    # A10
    a10 = root / "v2_release_chl2_factor_compatibility"
    models = [
        ("CHL2", 0, 1, 1, -3.2281),
        ("CHL1", 0, 1, 0, -3.2290),
        ("PATH_ONLY_ETA", 0, 0, 1, -3.2639),
        ("ROW_MASK_ETA", 0, 0, 0, -3.2663),
        ("CG_PATH_ETA", 1, 0, 1, -3.2678),
        ("H2_COND_ETA", 1, 0, 0, -3.2685),
        ("CG_MARKOV_ETA", 1, 1, 0, -3.2824),
        ("CG_MARKOV_PATH_ETA", 1, 1, 1, -3.2829),
    ]
    _write_csv(
        a10 / "chl2_factor_model_summary.csv",
        [
            {
                "filter": "ALL",
                "model": model,
                "H": h,
                "R": r,
                "P": p,
                "n_events": 1000,
                "loglik_per_event": ll,
                "conditional_kl": ll + 3.3,
                "rank_by_loglik": rank,
            }
            for rank, (model, h, r, p, ll) in enumerate(models, start=1)
        ],
    )
    effects = ["H", "R", "P", "H:R", "H:P", "R:P", "H:R:P"]
    _write_csv(
        a10 / "chl2_factorial_effects_summary.csv",
        [
            {
                "filter": "ALL",
                "effect": effect,
                "pooled_effect_loglik_per_event": value,
                "n_blocks": 2,
                "positive_blocks": 2 if value > 0 else 0,
                "negative_blocks": 2 if value < 0 else 0,
                "all_blocks_same_nonzero_sign": True,
            }
            for effect, value in zip(effects, [-0.03, 0.01, 0.001, -0.025, -0.001, -0.001, 0.0001])
        ],
    )
    context_rows = [
        ("P", "H=0;R=1", "CHL1", "CHL2", 0.001),
        ("R", "H=0;P=1", "PATH_ONLY_ETA", "CHL2", 0.035),
        ("H", "R=1;P=0", "CHL1", "CG_MARKOV_ETA", -0.053),
        ("H", "R=1;P=1", "CHL2", "CG_MARKOV_PATH_ETA", -0.055),
    ]
    _write_csv(
        a10 / "chl2_factor_context_effects_summary.csv",
        [
            {
                "filter": "ALL",
                "factor": factor,
                "context": context,
                "model_low": low,
                "model_high": high,
                "pooled_delta_loglik_per_event": value,
                "n_blocks": 2,
                "positive_blocks": 2 if value > 0 else 0,
                "negative_blocks": 2 if value < 0 else 0,
            }
            for factor, context, low, high, value in context_rows
        ],
    )


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(DOCS) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def test_table_builder_generates_fragments_and_hash_manifest(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    target = tmp_path / "paper" / "tables"
    _make_outputs(outputs)
    subprocess.run(
        [
            sys.executable,
            str(DOCS / "build_tables.py"),
            "--outputs-root",
            str(outputs),
            "--output-dir",
            str(target),
            "--strict",
        ],
        check=True,
        env=_env(),
    )
    expected = {
        "table_main_likelihood.tex",
        "table_chl2_chl1_block_stability.tex",
        "table_y_sweep.tex",
        "table_memory_irreducibility.tex",
        "table_scale_wave.tex",
        "table_modular_orientation_lift.tex",
        "table_chl4_q3_residual.tex",
        "table_hrp_model_ranking.tex",
        "table_hrp_factorial_effects.tex",
        "table_hrp_context_effects.tex",
        "tables_manifest.json",
    }
    assert expected == {p.name for p in target.iterdir()}
    main = (target / "table_main_likelihood.tex").read_text(encoding="utf-8")
    assert "Source-SHA256" in main
    assert r"$\Delta\ell$" in main
    stability = (target / "table_chl2_chl1_block_stability.tex").read_text(encoding="utf-8")
    scale_wave = (target / "table_scale_wave.tex").read_text(encoding="utf-8")
    factorial = (target / "table_hrp_factorial_effects.tex").read_text(encoding="utf-8")
    assert r"Unweighted block mean $\Delta\ell$" in stability
    assert "0.000001" in scale_wave
    assert "Factorial effect (nats/event)" in factorial

    manifest = json.loads((target / "tables_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "chl2-paper-tables@1"
    assert manifest["producer"] == "docs/build_tables.py"
    assert len(manifest["outputs"]) == 10
    assert not manifest["skipped"]


def test_figure_builder_generates_separate_svg_files(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    outputs = tmp_path / "outputs"
    target = tmp_path / "paper" / "figures"
    _make_outputs(outputs)
    subprocess.run(
        [
            sys.executable,
            str(DOCS / "build_figures.py"),
            "--outputs-root",
            str(outputs),
            "--output-dir",
            str(target),
            "--formats",
            "svg",
            "--strict",
        ],
        check=True,
        env=_env(),
    )
    svgs = sorted(target.glob("*.svg"))
    assert len(svgs) == 8
    assert all(p.stat().st_size > 1000 for p in svgs)
    heatmap = (target / "fig_y_sweep_heatmap.svg").read_text(encoding="utf-8")
    assert "#ffffff" in heatmap.lower()

    manifest = json.loads((target / "figures_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "chl2-paper-figures@1"
    assert manifest["producer"] == "docs/build_figures.py"
    assert manifest["pdf_fonttype"] == 42
    assert manifest["ps_fonttype"] == 42
    assert manifest["svg_hashsalt"] == "chl2-v2.0.0"
    assert len(manifest["outputs"]) == 8
    assert not manifest["skipped"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_vector_figures_are_deterministic_and_avoid_type3_fonts(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    outputs = tmp_path / "outputs"
    first = tmp_path / "first"
    second = tmp_path / "second"
    _make_outputs(outputs)

    command = [
        sys.executable,
        str(DOCS / "build_figures.py"),
        "--outputs-root",
        str(outputs),
        "--formats",
        "svg,pdf",
        "--only",
        "y_sweep_heatmap,hrp_factorial",
        "--strict",
    ]
    subprocess.run([*command, "--output-dir", str(first)], check=True, env=_env())
    subprocess.run([*command, "--output-dir", str(second)], check=True, env=_env())

    first_files = sorted(p for p in first.iterdir() if p.suffix in {".svg", ".pdf"})
    second_files = sorted(p for p in second.iterdir() if p.suffix in {".svg", ".pdf"})
    assert [p.name for p in first_files] == [p.name for p in second_files]
    assert len(first_files) == 6

    for left, right in zip(first_files, second_files):
        assert _sha256(left) == _sha256(right), left.name
        if left.suffix == ".pdf":
            assert b"/Subtype /Type3" not in left.read_bytes()


def test_readme_uses_canonical_v2_paths_and_reconstructed_table4_language() -> None:
    text = (REPO / "README.md").read_text(encoding="utf-8")
    assert "outputs/v2_release_chl2_main" in text
    assert "outputs/v2_release_chl4_residual_blocks_aligned" in text
    assert "outputs/v2_release_chl2_factor_compatibility" in text
    assert "absolute_prime_os_previous_gap_conditioned" in text
    assert "naive_table4_legacy" in text
    assert "missing-modulus/zero-row fallback" in text
    assert "1/(q-1)" in text
    assert "withdrawn as a scientific baseline" in text
    assert "historical provenance remains unresolved" not in text
    assert "reference_outputs/v2.0" not in text

    provenance = REPO / "docs" / "NAIVE_TABLE4_LEGACY_PROVENANCE.md"
    assert provenance.is_file()
    provenance_text = provenance.read_text(encoding="utf-8")
    assert "A06: CLOSED" in provenance_text
    assert "--os-prime-mods" in provenance_text
    assert "zero model rows" in provenance_text
    assert "uniform reduced-residue rows" in provenance_text


def test_non_strict_table_build_skips_missing_audits(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    target = tmp_path / "tables"
    _make_outputs(outputs)
    # Remove A05 to exercise the explicit non-strict path.
    for path in (outputs / "v2_release_chl4d2_gap_population_bias").iterdir():
        path.unlink()
    (outputs / "v2_release_chl4d2_gap_population_bias").rmdir()
    subprocess.run(
        [
            sys.executable,
            str(DOCS / "build_tables.py"),
            "--outputs-root",
            str(outputs),
            "--output-dir",
            str(target),
            "--no-strict",
        ],
        check=True,
        env=_env(),
    )
    manifest = json.loads((target / "tables_manifest.json").read_text(encoding="utf-8"))
    assert any(item["table"] == "scale_wave" for item in manifest["skipped"])
