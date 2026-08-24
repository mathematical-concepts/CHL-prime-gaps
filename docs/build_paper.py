#!/usr/bin/env python3
"""Validate CHL2 v2.0.0 paper assets and compile the release whitepaper.

The repository versions scientific producers and TeX sources, while DS1, audit
CSVs, generated tables/figures, and PDFs remain local under ``outputs/``.  This
builder closes the public chain:

    audits -> generated paper assets -> TeX -> deterministic PDF + manifest.

Toolchain notes
---------------
``latexmk`` is a Perl script and no TeX distribution bundles an interpreter.
POSIX hosts supply one at ``/usr/bin/perl``; on Windows MiKTeX depends on an
external Perl, and the common one (Git for Windows) sits in ``Git\\usr\\bin``,
which Git Bash puts on ``PATH`` and PowerShell and ``cmd`` do not.  Without the
handling in :func:`ensure_perl_on_path` the same checkout builds from one shell
and dies from another with MiKTeX's ``The script engine could not be found``.
Only the subprocess environment is modified; the system ``PATH`` is untouched.

Failed subprocesses echo their captured output through :func:`_run` before the
exception propagates, because ``subprocess.run(check=True)`` otherwise hides a
plain toolchain error behind a Python traceback.

The PDF is byte-reproducible for a fixed TeX distribution, not across
distributions; see ``docs/REPRODUCE_V2_0_0.md`` for the measured differences.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEX = Path("docs/CHL2_conditional_hardy_littlewood_markov_whitepaper_v2_0_0_rc1.tex")
DEFAULT_ASSETS = Path("outputs/v2_release_paper_assets")
DEFAULT_OUTPUT = Path("outputs/v2_release_paper")
PDF_BASENAME = "CHL2_conditional_hardy_littlewood_markov_whitepaper_v2_0_0_rc1"
EXPECTED_MATPLOTLIB = "3.11.1"
EXPECTED_SVG_SALT = "chl2-v2.0.0"
_FAILURE_OUTPUT_LINES = 40

REQUIRED_TABLES = (
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
)

REQUIRED_FIGURES = (
    "fig_construction_chain.pdf",
    "fig_tuple_anatomy.pdf",
    "fig_chl2_chl1_gain.pdf",
    "fig_y_sweep_heatmap.pdf",
    "fig_scale_wave_d3.pdf",
    "fig_scale_wave_residual.pdf",
    "fig_orientation_lift_diagonal.pdf",
    "fig_orientation_lift_kl.pdf",
    "fig_hrp_model_ranking.pdf",
    "fig_hrp_factorial_effects.pdf",
    "fig_repro_pipeline.pdf",
)

UNDEFINED_PATTERNS = (
    re.compile(r"LaTeX Warning: Reference .* undefined"),
    re.compile(r"LaTeX Warning: Citation .* undefined"),
    re.compile(r"There were undefined references"),
    re.compile(r"There were undefined citations"),
)
OVERFULL_PATTERN = re.compile(r"Overfull \\hbox")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_path(path: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return repo_root / p


def portable_path(path: str | Path, repo_root: Path = REPO_ROOT) -> str:
    p = Path(path).resolve(strict=False)
    try:
        return p.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def _records_by_basename(manifest: dict) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for item in manifest.get("outputs", []):
        name = Path(str(item.get("path", ""))).name
        if name:
            records[name] = item
    return records


def _load_manifest(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required paper-asset manifest is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("outputs"), list):
        raise ValueError(f"Malformed paper-asset manifest: {path}")
    return payload


def validate_assets(
    assets_dir: Path,
    *,
    strict_manifest: bool = True,
    expected_matplotlib_version: str = EXPECTED_MATPLOTLIB,
) -> dict[str, object]:
    tables_dir = assets_dir / "tables"
    figures_dir = assets_dir / "figures"
    table_manifest_path = tables_dir / "tables_manifest.json"
    figure_manifest_path = figures_dir / "figures_manifest.json"

    missing: list[str] = []
    for name in REQUIRED_TABLES:
        if not (tables_dir / name).is_file():
            missing.append(f"tables/{name}")
    for name in REQUIRED_FIGURES:
        if not (figures_dir / name).is_file():
            missing.append(f"figures/{name}")
    if missing:
        raise FileNotFoundError("Missing required paper assets:\n  " + "\n  ".join(missing))

    result: dict[str, object] = {
        "tables": [],
        "figures": [],
        "manifests": [],
        "manifest_validation": "skipped",
    }
    for name in REQUIRED_TABLES:
        p = tables_dir / name
        result["tables"].append({"path": portable_path(p), "sha256": sha256_file(p), "size_bytes": p.stat().st_size})
    for name in REQUIRED_FIGURES:
        p = figures_dir / name
        result["figures"].append({"path": portable_path(p), "sha256": sha256_file(p), "size_bytes": p.stat().st_size})

    if strict_manifest:
        table_manifest = _load_manifest(table_manifest_path)
        figure_manifest = _load_manifest(figure_manifest_path)
        failures: list[str] = []
        if table_manifest.get("schema") != "chl2-paper-tables@1":
            failures.append("unexpected table-manifest schema")
        if figure_manifest.get("schema") != "chl2-paper-figures@1":
            failures.append("unexpected figure-manifest schema")
        expected_producers = [
            (table_manifest, REPO_ROOT / "docs" / "build_tables.py", "table"),
            (figure_manifest, REPO_ROOT / "docs" / "build_figures.py", "figure"),
        ]
        for manifest, producer_path, label in expected_producers:
            if Path(str(manifest.get("producer", ""))).as_posix() != portable_path(producer_path):
                failures.append(f"unexpected {label}-manifest producer path")
            if str(manifest.get("producer_sha256", "")).lower() != sha256_file(producer_path):
                failures.append(f"{label}-manifest producer SHA-256 does not match current source")
        if table_manifest.get("skipped"):
            failures.append("table manifest contains skipped assets")
        if figure_manifest.get("skipped"):
            failures.append("figure manifest contains skipped assets")
        if int(figure_manifest.get("pdf_fonttype", -1)) != 42:
            failures.append("figure manifest does not declare pdf_fonttype=42")
        if int(figure_manifest.get("ps_fonttype", -1)) != 42:
            failures.append("figure manifest does not declare ps_fonttype=42")
        if str(figure_manifest.get("source_date_epoch")) != "0":
            failures.append("figure manifest does not declare source_date_epoch=0")
        if str(figure_manifest.get("svg_hashsalt")) != "chl2-v2.0.0":
            failures.append("figure manifest has an unexpected svg_hashsalt")
        if expected_matplotlib_version and str(figure_manifest.get("matplotlib_version")) != expected_matplotlib_version:
            failures.append(
                "figure manifest Matplotlib version "
                f"{figure_manifest.get('matplotlib_version')!r} != {expected_matplotlib_version!r}"
            )
        table_records = _records_by_basename(table_manifest)
        figure_records = _records_by_basename(figure_manifest)
        for name in REQUIRED_TABLES:
            record = table_records.get(name)
            if record is None:
                failures.append(f"table manifest lacks {name}")
                continue
            actual = sha256_file(tables_dir / name)
            if str(record.get("sha256", "")).lower() != actual:
                failures.append(f"table hash mismatch for {name}")
        for name in REQUIRED_FIGURES:
            record = figure_records.get(name)
            if record is None:
                failures.append(f"figure manifest lacks {name}")
                continue
            actual = sha256_file(figures_dir / name)
            if str(record.get("sha256", "")).lower() != actual:
                failures.append(f"figure hash mismatch for {name}")
        if table_manifest.get("strict") is not True:
            failures.append("table manifest was not generated in strict mode")
        if figure_manifest.get("strict") is not True:
            failures.append("figure manifest was not generated in strict mode")
        for name in REQUIRED_FIGURES:
            if b"/Subtype /Type3" in (figures_dir / name).read_bytes():
                failures.append(f"Type-3 font marker found in {name}")
        if failures:
            raise ValueError("Paper-asset manifest validation failed:\n  " + "\n  ".join(failures))
        result["manifests"] = [
            {"path": portable_path(table_manifest_path), "sha256": sha256_file(table_manifest_path)},
            {"path": portable_path(figure_manifest_path), "sha256": sha256_file(figure_manifest_path)},
        ]
        result["manifest_validation"] = "pass"
    return result


_PERL_INSTALL_HINT = (
    "Install Strawberry Perl (winget install StrawberryPerl.StrawberryPerl) "
    "or add Git for Windows' usr" + chr(92) * 2 + "bin to PATH."
    if os.name == "nt"
    else "Install Perl (Debian/Ubuntu: apt-get install perl; macOS ships one at /usr/bin/perl)."
)


def _perl_search_dirs() -> tuple[Path, ...]:
    """Directories that may hold a Perl interpreter on Windows.

    MiKTeX ships ``latexmk`` as a Perl script but bundles no interpreter.  The
    usual host has one inside Git for Windows or Strawberry Perl, neither of
    which puts its ``perl.exe`` on the default ``PATH``.  Git Bash does, so the
    build succeeds there and fails from PowerShell or cmd on the same machine.
    """
    candidates: list[Path] = []
    for root_var in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"):
        root = os.environ.get(root_var)
        if root:
            candidates.append(Path(root) / "Git" / "usr" / "bin")
    for drive in ("C:/", "D:/"):
        candidates.append(Path(drive) / "Strawberry" / "perl" / "bin")
    if os.name != "nt":
        # A truncated PATH (minimal containers, cron) can hide the system Perl.
        candidates.extend((Path("/usr/bin"), Path("/usr/local/bin"), Path("/opt/homebrew/bin")))
    git = shutil.which("git")
    if git:
        candidates.append(Path(git).resolve().parents[1] / "usr" / "bin")
    seen: dict[str, Path] = {}
    for candidate in candidates:
        key = str(candidate).lower()
        if key not in seen:
            seen[key] = candidate
    return tuple(seen.values())


def ensure_perl_on_path(env: dict[str, str]) -> str | None:
    """Return the Perl interpreter latexmk will use, extending ``env`` if needed.

    Returns ``None`` when no interpreter can be located, so callers can fail
    with an actionable message instead of MiKTeX's opaque
    ``The script engine could not be found``.
    """
    existing = shutil.which("perl", path=env.get("PATH", ""))
    if existing:
        return existing
    for directory in _perl_search_dirs():
        candidate = shutil.which("perl", path=str(directory))
        if candidate:
            env["PATH"] = str(directory) + os.pathsep + env.get("PATH", "")
            print(f"[paper] using Perl for latexmk: {candidate}")
            return candidate
    return None


def _run(command: Sequence[str], *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    print("[paper] " + " ".join(command))
    try:
        return subprocess.run(
            list(command),
            cwd=cwd,
            env=env,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as error:
        # ``check=True`` swallows the captured stream inside the exception, which
        # turns a plain toolchain failure into an opaque traceback.  Surface the
        # tail of the real output so the operator sees the actual cause.
        output = (error.output or "").strip()
        if output:
            tail = output.splitlines()[-_FAILURE_OUTPUT_LINES:]
            print(f"[paper] command failed with exit status {error.returncode}; last output:", file=sys.stderr)
            for line in tail:
                print(f"[paper]   {line}", file=sys.stderr)
        raise


def build_assets(*, outputs_root: Path, assets_dir: Path, env: dict[str, str]) -> None:
    tables = assets_dir / "tables"
    figures = assets_dir / "figures"
    _run(
        [
            sys.executable,
            "docs/build_tables.py",
            "--outputs-root",
            portable_path(outputs_root),
            "--output-dir",
            portable_path(tables),
            "--strict",
        ],
        cwd=REPO_ROOT,
        env=env,
    )
    _run(
        [
            sys.executable,
            "docs/build_figures.py",
            "--outputs-root",
            portable_path(outputs_root),
            "--output-dir",
            portable_path(figures),
            "--formats",
            "svg,pdf",
            "--strict",
        ],
        cwd=REPO_ROOT,
        env=env,
    )


def _git_metadata(repo_root: Path) -> dict[str, object]:
    def call(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=repo_root, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    status = call("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": call("rev-parse", "HEAD"),
        "describe": call("describe", "--always", "--dirty", "--tags"),
        "branch": call("rev-parse", "--abbrev-ref", "HEAD"),
        "tracked_worktree_clean": not bool(status),
        "tracked_changes": [line for line in status.splitlines() if line],
    }


def _engine_choice(requested: str) -> tuple[str, str]:
    if requested == "latexmk":
        executable = shutil.which("latexmk")
        if not executable:
            raise FileNotFoundError("latexmk was requested but is not installed")
        return executable, "latexmk"
    if requested == "pdflatex":
        executable = shutil.which("pdflatex")
        if not executable:
            raise FileNotFoundError("pdflatex was requested but is not installed")
        return executable, "pdflatex"
    executable = shutil.which("latexmk")
    if executable:
        return executable, "latexmk"
    executable = shutil.which("pdflatex")
    if executable:
        return executable, "pdflatex"
    raise FileNotFoundError("No LaTeX engine found; install latexmk or pdflatex")


def _engine_version(executable: str) -> str:
    try:
        line = subprocess.check_output([executable, "--version"], text=True, stderr=subprocess.STDOUT).splitlines()[0]
        return line.strip()
    except Exception:
        return "unknown"


def write_wrapper(*, tex_source: Path, tables_dir: Path, figures_dir: Path, output_dir: Path) -> Path:
    wrapper = output_dir / f"{PDF_BASENAME}.tex"
    wrapper.write_text(
        "\\def\\CHLTableDir{" + portable_path(tables_dir) + "}\n"
        "\\def\\CHLFigureDir{" + portable_path(figures_dir) + "}\n"
        "\\input{" + portable_path(tex_source) + "}\n",
        encoding="utf-8",
        newline="\n",
    )
    return wrapper


def _compile(*, wrapper: Path, output_dir: Path, engine: str, env: dict[str, str]) -> tuple[Path, Path, str, str]:
    executable, kind = _engine_choice(engine)
    if kind == "latexmk":
        if ensure_perl_on_path(env) is None:
            raise FileNotFoundError(
                "latexmk is a Perl script and no Perl interpreter was found on PATH. "
                + _PERL_INSTALL_HINT
                + " Alternatively rerun with --engine pdflatex, which needs no Perl."
            )
        command = [
            executable,
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            "-outdir=" + portable_path(output_dir),
            portable_path(wrapper),
        ]
        completed = _run(command, cwd=REPO_ROOT, env=env)
        combined_output = completed.stdout
    else:
        command = [
            executable,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            "-output-directory=" + portable_path(output_dir),
            portable_path(wrapper),
        ]
        passes = [_run(command, cwd=REPO_ROOT, env=env) for _ in range(3)]
        combined_output = "\n".join(item.stdout for item in passes)

    pdf = output_dir / f"{PDF_BASENAME}.pdf"
    log = output_dir / f"{PDF_BASENAME}.log"
    if not pdf.is_file() or not log.is_file():
        raise FileNotFoundError("LaTeX completed without the expected PDF/log output")
    return pdf, log, kind, combined_output


def _check_log(log_path: Path, *, fail_on_overfull: bool) -> dict[str, object]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    undefined = [pattern.pattern for pattern in UNDEFINED_PATTERNS if pattern.search(text)]
    overfull = len(OVERFULL_PATTERN.findall(text))
    if undefined:
        raise RuntimeError("The paper contains unresolved references or citations: " + "; ".join(undefined))
    if fail_on_overfull and overfull:
        raise RuntimeError(f"The paper contains {overfull} overfull hbox warning(s)")
    return {"undefined_reference_patterns": undefined, "overfull_hbox_count": overfull}


def _cleanup_aux(output_dir: Path, *, keep_aux: bool) -> None:
    if keep_aux:
        return
    extensions = {".aux", ".fdb_latexmk", ".fls", ".lof", ".lot", ".out", ".toc"}
    for path in output_dir.iterdir():
        if path.suffix in extensions or path.name == f"{PDF_BASENAME}.tex":
            path.unlink(missing_ok=True)


def write_manifest(
    *,
    output_dir: Path,
    tex_source: Path,
    assets_dir: Path,
    asset_report: dict[str, object],
    pdf: Path,
    log: Path,
    engine_kind: str,
    engine_executable: str,
    log_report: dict[str, object],
    source_date_epoch: str,
) -> Path:
    payload = {
        "schema": "chl2-paper-build@1",
        "status": "compiled",
        "paper_version": "2.0.0-rc1",
        "producer": portable_path(Path(__file__)),
        "producer_sha256": sha256_file(Path(__file__)),
        "tex_source": {
            "path": portable_path(tex_source),
            "sha256": sha256_file(tex_source),
            "size_bytes": tex_source.stat().st_size,
        },
        "assets_dir": portable_path(assets_dir),
        "assets": asset_report,
        "source_date_epoch": source_date_epoch,
        "engine": {
            "kind": engine_kind,
            "executable": engine_executable,
            "version": _engine_version(engine_executable),
        },
        "git": _git_metadata(REPO_ROOT),
        "log": {
            "path": portable_path(log),
            "sha256": sha256_file(log),
            **log_report,
        },
        "pdf": {
            "path": portable_path(pdf),
            "sha256": sha256_file(pdf),
            "size_bytes": pdf.stat().st_size,
        },
    }
    path = output_dir / "paper_build_manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_check_manifest(
    *,
    output_dir: Path,
    tex_source: Path,
    assets_dir: Path,
    asset_report: dict[str, object],
    source_date_epoch: str,
) -> Path:
    payload = {
        "schema": "chl2-paper-build@1",
        "status": "validated",
        "paper_version": "2.0.0-rc1",
        "producer": portable_path(Path(__file__)),
        "producer_sha256": sha256_file(Path(__file__)),
        "tex_source": {
            "path": portable_path(tex_source),
            "sha256": sha256_file(tex_source),
            "size_bytes": tex_source.stat().st_size,
        },
        "assets_dir": portable_path(assets_dir),
        "assets": asset_report,
        "source_date_epoch": source_date_epoch,
        "git": _git_metadata(REPO_ROOT),
    }
    path = output_dir / "paper_build_manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", default=str(DEFAULT_TEX), help="TeX source relative to repository root")
    parser.add_argument("--outputs-root", default="outputs", help="Canonical audit outputs root")
    parser.add_argument("--assets-dir", "--assets-root", dest="assets_dir", default=str(DEFAULT_ASSETS), help="Generated paper-assets directory")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="PDF build directory")
    parser.add_argument("--engine", choices=["auto", "latexmk", "pdflatex"], default="auto")
    parser.add_argument("--build-assets", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--strict-assets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-on-overfull", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-aux", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--source-date-epoch", default=os.environ.get("SOURCE_DATE_EPOCH", "0"))
    parser.add_argument("--expected-matplotlib-version", default=EXPECTED_MATPLOTLIB)
    parser.add_argument("--check-only", action="store_true", help="Validate source/assets without compiling")
    parser.add_argument("--strict", action="store_true", help="Alias for the default strict release gates")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    tex_source = repo_path(args.tex)
    outputs_root = repo_path(args.outputs_root)
    assets_dir = repo_path(args.assets_dir)
    output_dir = repo_path(args.output_dir)
    if not tex_source.is_file():
        raise FileNotFoundError(f"Whitepaper TeX source is missing: {tex_source}")
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["SOURCE_DATE_EPOCH"] = str(args.source_date_epoch)
    env.setdefault("FORCE_SOURCE_DATE", "1")
    env.setdefault("TZ", "UTC")

    if args.build_assets:
        build_assets(outputs_root=outputs_root, assets_dir=assets_dir, env=env)
    asset_report = validate_assets(
        assets_dir,
        strict_manifest=bool(args.strict_assets),
        expected_matplotlib_version=str(args.expected_matplotlib_version).strip(),
    )
    if args.check_only:
        manifest = write_check_manifest(
            output_dir=output_dir,
            tex_source=tex_source,
            assets_dir=assets_dir,
            asset_report=asset_report,
            source_date_epoch=str(args.source_date_epoch),
        )
        print(f"[paper] PASS source/assets validation: {portable_path(manifest)}")
        return 0
    wrapper = write_wrapper(
        tex_source=tex_source,
        tables_dir=assets_dir / "tables",
        figures_dir=assets_dir / "figures",
        output_dir=output_dir,
    )
    pdf, log, engine_kind, _ = _compile(
        wrapper=wrapper,
        output_dir=output_dir,
        engine=args.engine,
        env=env,
    )
    log_report = _check_log(log, fail_on_overfull=bool(args.fail_on_overfull))
    executable, _ = _engine_choice(args.engine)
    manifest = write_manifest(
        output_dir=output_dir,
        tex_source=tex_source,
        assets_dir=assets_dir,
        asset_report=asset_report,
        pdf=pdf,
        log=log,
        engine_kind=engine_kind,
        engine_executable=executable,
        log_report=log_report,
        source_date_epoch=str(args.source_date_epoch),
    )
    _cleanup_aux(output_dir, keep_aux=bool(args.keep_aux))
    print(f"[paper] PDF: {portable_path(pdf)}")
    print(f"[paper] SHA256: {sha256_file(pdf)}")
    print(f"[paper] manifest: {portable_path(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
