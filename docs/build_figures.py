#!/usr/bin/env python3
"""Generate CHL2 v2.0.0 paper figures from public producers.

The script produces one plot per figure file, uses the non-interactive Matplotlib
backend, and records source/output SHA-256 values in ``figures_manifest.json``.
Data-driven figures read canonical audit outputs; conceptual diagrams are
constructed deterministically by a public Matplotlib producer.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, Sequence

DOCS_DIR = Path(__file__).resolve().parent
if str(DOCS_DIR) not in sys.path:
    sys.path.insert(0, str(DOCS_DIR))

# Honor an externally supplied reproducible-build timestamp, otherwise use a
# stable epoch so PDF/SVG metadata is byte-reproducible across runs.
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

try:
    import matplotlib
except ImportError as exc:  # pragma: no cover - environment-dependent
    raise SystemExit(
        "Figure generation requires the paper dependencies. Run `uv sync --frozen` "
        "or install the project with `python -m pip install -e .`."
    ) from exc

# Publication-safe and deterministic vector output.  Type 42 avoids Matplotlib
# Type 3 fonts in PDF preflight, while a fixed SVG salt stabilizes element IDs.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.hashsalt"] = "chl2-v2.0.0"

from figure_scripts.common import parse_formats  # noqa: E402
from figure_scripts.fig_conceptual import build as build_conceptual  # noqa: E402
from figure_scripts.fig_chl2_chl1_gain import build as build_gain  # noqa: E402
from figure_scripts.fig_hrp_factorial import build as build_hrp  # noqa: E402
from figure_scripts.fig_orientation_lift import build as build_orientation  # noqa: E402
from figure_scripts.fig_scale_wave import build as build_scale_wave  # noqa: E402
from figure_scripts.fig_y_sweep_heatmap import build as build_y_sweep  # noqa: E402
from release_assets import output_record, portable_path, sha256_file, source_record, write_json  # noqa: E402


FigureBuilder = Callable[[Path, Path, list[str]], tuple[list[Path], list[Path]]]

BUILDERS: dict[str, FigureBuilder] = {
    "conceptual": build_conceptual,
    "chl2_chl1_gain": build_gain,
    "y_sweep_heatmap": build_y_sweep,
    "orientation_lift": build_orientation,
    "scale_wave": build_scale_wave,
    "hrp_factorial": build_hrp,
}


def parse_only(text: str | None) -> list[str]:
    if not text:
        return list(BUILDERS)
    requested = [item.strip() for item in text.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(BUILDERS))
    if unknown:
        raise ValueError(f"Unknown figure builder(s): {unknown}; choices={sorted(BUILDERS)}")
    return requested


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", default="outputs", help="Repository outputs directory")
    parser.add_argument(
        "--output-dir",
        default="outputs/v2_release_paper_assets/figures",
        help="Generated figure directory",
    )
    parser.add_argument("--formats", default="svg,pdf", help="Comma-separated svg,pdf,png")
    parser.add_argument("--only", default=None, help=f"Comma-separated subset: {','.join(BUILDERS)}")
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if a selected audit output is missing or incompatible",
    )
    args = parser.parse_args(argv)

    outputs_root = Path(args.outputs_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = parse_formats(args.formats)
    selected = parse_only(args.only)

    generated: list[Path] = []
    sources: dict[str, Path] = {}
    skipped: list[dict[str, str]] = []
    for key in selected:
        try:
            outputs, inputs = BUILDERS[key](outputs_root, output_dir, formats)
        except (FileNotFoundError, ValueError) as exc:
            if args.strict:
                raise
            skipped.append({"figure": key, "reason": str(exc)})
            print(f"[figures] SKIP {key}: {exc}", file=sys.stderr)
            continue
        generated.extend(outputs)
        for source in inputs:
            sources[source.as_posix()] = source
        for output in outputs:
            print(f"[figures] wrote {output}")

    manifest_path = output_dir / "figures_manifest.json"
    manifest = {
        "schema": "chl2-paper-figures@1",
        "producer": portable_path(Path(__file__)),
        "producer_sha256": sha256_file(Path(__file__)),
        "matplotlib_version": matplotlib.__version__,
        "source_date_epoch": os.environ["SOURCE_DATE_EPOCH"],
        "pdf_fonttype": int(matplotlib.rcParams["pdf.fonttype"]),
        "ps_fonttype": int(matplotlib.rcParams["ps.fonttype"]),
        "svg_hashsalt": str(matplotlib.rcParams["svg.hashsalt"]),
        "outputs_root": portable_path(outputs_root),
        "formats": formats,
        "selected": selected,
        "strict": bool(args.strict),
        "sources": [source_record(p) for p in sorted(sources.values(), key=lambda x: x.as_posix())],
        "outputs": [output_record(p) for p in sorted(generated, key=lambda x: x.as_posix())],
        "skipped": skipped,
    }
    write_json(manifest_path, manifest)
    print(f"[figures] wrote {manifest_path}")
    if args.strict and not generated:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
