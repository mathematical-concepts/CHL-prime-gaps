from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
SOURCE = DOCS / "CHL2_conditional_hardy_littlewood_markov_whitepaper_v2_0_0_rc1.tex"
BUILD_PAPER_PATH = DOCS / "build_paper.py"


def _load_build_paper():
    spec = importlib.util.spec_from_file_location("chl2_build_paper", BUILD_PAPER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_asset_manifests(module, assets: Path) -> None:
    tables = assets / "tables"
    figures = assets / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    for name in module.REQUIRED_TABLES:
        (tables / name).write_text(r"\begin{tabular}{l}x\\\end{tabular}" + "\n", encoding="utf-8")

    # Validation only requires a non-Type-3 PDF marker.  The integration test
    # below creates a real PDF for LaTeX compilation.
    for name in module.REQUIRED_FIGURES:
        (figures / name).write_bytes(b"%PDF-1.4\n% deterministic fixture\n")

    table_outputs = [
        {"path": f"outputs/v2_release_paper_assets/tables/{name}", "sha256": _sha(tables / name)}
        for name in module.REQUIRED_TABLES
    ]
    figure_outputs = [
        {"path": f"outputs/v2_release_paper_assets/figures/{name}", "sha256": _sha(figures / name)}
        for name in module.REQUIRED_FIGURES
    ]
    (tables / "tables_manifest.json").write_text(
        json.dumps({"schema": "chl2-paper-tables@1", "producer": "docs/build_tables.py", "producer_sha256": _sha(DOCS / "build_tables.py"), "strict": True, "outputs": table_outputs, "skipped": []}), encoding="utf-8"
    )
    (figures / "figures_manifest.json").write_text(
        json.dumps(
            {
                "schema": "chl2-paper-figures@1",
                "producer": "docs/build_figures.py",
                "producer_sha256": _sha(DOCS / "build_figures.py"),
                "strict": True,
                "outputs": figure_outputs,
                "skipped": [],
                "matplotlib_version": "3.11.1",
                "pdf_fonttype": 42,
                "ps_fonttype": 42,
                "svg_hashsalt": "chl2-v2.0.0",
                "source_date_epoch": "0",
            }
        ),
        encoding="utf-8",
    )


def test_whitepaper_uses_generated_assets_and_safe_release_language() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    assert "Whitepaper v2.0.0 RC1" in text
    assert "10.5281/zenodo.20368548" in text
    for name in (
        "table_main_likelihood.tex",
        "table_modular_orientation_lift.tex",
        "table_hrp_factorial_effects.tex",
        "fig_construction_chain.pdf",
        "fig_tuple_anatomy.pdf",
        "fig_orientation_lift_kl.pdf",
        "fig_hrp_model_ranking.pdf",
        "fig_repro_pipeline.pdf",
    ):
        assert name in text

    assert "absolute\\_prime\\_os\\_previous\\_gap\\_conditioned" in text
    assert "naive\\_table4\\_legacy" in text
    assert "missing upstream moduli" in text
    assert "withdrawn as a scientific baseline" in text
    assert "not identify $\\Omega_Y^{\\mathrm{path}}$ as an autonomous modular mechanism" in text
    assert "Quantitative low-wheel and local LOS" in text

    assert "fig07_negative_controls" not in text
    assert "Truncated LOS projections" not in text
    assert "The inadequacy of higher-order low-wheel clusters" not in text
    assert "reference_outputs" not in text


def test_release_metadata_and_reproduction_documents_are_v2() -> None:
    citation = (REPO / "CITATION.cff").read_text(encoding="utf-8")
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    reproduce = (DOCS / "REPRODUCE_V2_0_0.md").read_text(encoding="utf-8")
    release_notes = (REPO / "RELEASE_NOTES_v2.0.0.md").read_text(encoding="utf-8")

    assert 'version: "2.0.0"' in citation
    assert "10.5281/zenodo.20368548" in citation
    assert 'version = "2.0.0"' in pyproject
    assert "docs/build_paper.py" in readme
    assert "--assets-dir outputs/v2_release_paper_assets" in reproduce
    assert "--strict-assets" in reproduce
    assert "upstream OS CSV omitted" in release_notes
    assert "generated PDF is a GitHub/Zenodo/arXiv release artifact" in release_notes


def test_asset_manifest_validation_detects_tampering(tmp_path: Path) -> None:
    module = _load_build_paper()
    assets = tmp_path / "assets"
    _write_asset_manifests(module, assets)
    report = module.validate_assets(assets, strict_manifest=True)
    assert report["manifest_validation"] == "pass"
    assert len(report["tables"]) == 10
    assert len(report["figures"]) == 11

    (assets / "tables" / module.REQUIRED_TABLES[0]).write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        module.validate_assets(assets, strict_manifest=True)


def test_wrapper_uses_overridable_asset_directories(tmp_path: Path) -> None:
    module = _load_build_paper()
    source = REPO / "docs" / "paper.tex"
    tables = REPO / "outputs" / "v2_release_paper_assets" / "tables"
    figures = REPO / "outputs" / "v2_release_paper_assets" / "figures"
    wrapper = module.write_wrapper(
        tex_source=source,
        tables_dir=tables,
        figures_dir=figures,
        output_dir=tmp_path,
    )
    text = wrapper.read_text(encoding="utf-8")
    assert r"\def\CHLTableDir{outputs/v2_release_paper_assets/tables}" in text
    assert r"\def\CHLFigureDir{outputs/v2_release_paper_assets/figures}" in text
    assert r"\input{docs/paper.tex}" in text


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex is not installed")
def test_builder_compiles_a_minimal_fixture(tmp_path: Path) -> None:
    module = _load_build_paper()
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assets = tmp_path / "assets"
    _write_asset_manifests(module, assets)

    fig, ax = plt.subplots(figsize=(2, 1))
    ax.plot([0, 1], [0, 1])
    ax.set_axis_off()
    real_pdf = tmp_path / "fixture.pdf"
    fig.savefig(real_pdf, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    for name in module.REQUIRED_FIGURES:
        shutil.copyfile(real_pdf, assets / "figures" / name)

    tex = tmp_path / "paper.tex"
    tex.write_text(
        r"""\documentclass{article}
\usepackage{graphicx,booktabs}
\providecommand{\CHLTableDir}{.}
\providecommand{\CHLFigureDir}{.}
\begin{document}
\input{\CHLTableDir/table_main_likelihood.tex}
\includegraphics[width=0.4\textwidth]{\CHLFigureDir/fig_construction_chain.pdf}
\end{document}
""",
        encoding="utf-8",
    )
    out = tmp_path / "paper-output"
    subprocess.run(
        [
            sys.executable,
            str(BUILD_PAPER_PATH),
            "--tex",
            str(tex),
            "--assets-dir",
            str(assets),
            "--output-dir",
            str(out),
            "--engine",
            "pdflatex",
            "--no-strict-assets",
            "--no-fail-on-overfull",
        ],
        cwd=REPO,
        check=True,
    )
    pdf = out / f"{module.PDF_BASENAME}.pdf"
    manifest = out / "paper_build_manifest.json"
    assert pdf.is_file() and pdf.stat().st_size > 1000
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["pdf"]["sha256"] == _sha(pdf)
    assert payload["log"]["undefined_reference_patterns"] == []


def test_perl_is_resolved_for_latexmk_outside_git_bash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: latexmk is a Perl script and MiKTeX ships no interpreter.

    Perl lives in Git for Windows' ``usr/bin``, which is on PATH in Git Bash but
    not in PowerShell or cmd, so the build failed with MiKTeX's opaque
    ``The script engine could not be found`` depending only on the calling
    shell.  ``ensure_perl_on_path`` must recover the interpreter and prepend its
    directory to the subprocess PATH.
    """
    module = _load_build_paper()

    perl_dir = tmp_path / "usr" / "bin"
    perl_dir.mkdir(parents=True)
    suffix = ".exe" if sys.platform == "win32" else ""
    perl = perl_dir / f"perl{suffix}"
    perl.write_text("#!/bin/sh" + chr(10), encoding="utf-8")
    perl.chmod(0o755)
    empty = tmp_path / "empty"
    empty.mkdir()

    # No candidate directory holds an interpreter: the helper must report the
    # failure so the caller can raise an actionable error.
    monkeypatch.setattr(module, "_perl_search_dirs", lambda: (empty,))
    env = {"PATH": str(empty)}
    assert module.ensure_perl_on_path(env) is None
    assert env["PATH"] == str(empty)

    # The interpreter is off PATH but present in a known location: recover it
    # and prepend its directory, leaving the rest of PATH intact.
    monkeypatch.setattr(module, "_perl_search_dirs", lambda: (perl_dir,))
    env = {"PATH": str(empty)}
    assert module.ensure_perl_on_path(env) is not None
    assert env["PATH"] == str(perl_dir) + os.pathsep + str(empty)

    # Already on PATH: leave the environment untouched.
    env = {"PATH": str(perl_dir)}
    assert module.ensure_perl_on_path(env) is not None
    assert env["PATH"] == str(perl_dir)


def test_run_reports_command_output_on_failure(capsys: pytest.CaptureFixture[str]) -> None:
    """Regression: ``check=True`` hid the toolchain error behind a traceback."""
    module = _load_build_paper()
    script = "import sys; print('MARKER-real-cause'); sys.exit(3)"
    with pytest.raises(subprocess.CalledProcessError):
        module._run([sys.executable, "-c", script], cwd=REPO, env=dict(os.environ))
    captured = capsys.readouterr()
    assert "MARKER-real-cause" in captured.err
