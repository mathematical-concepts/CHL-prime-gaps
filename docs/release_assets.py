#!/usr/bin/env python3
"""Shared helpers for CHL2 v2.0.0 table and figure builders.

The public release deliberately does not version DS1 or scientific CSV outputs.
These helpers resolve the canonical local ``outputs/v2_release_*`` directories,
validate schemas, and attach SHA-256 provenance to generated paper assets.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


AUDIT_DIRS: Mapping[str, str] = {
    "A02": "v2_release_chl2_main",
    "A04": "v2_release_chl2_y_sweep",
    "A05": "v2_release_chl4d2_gap_population_bias",
    "A06A": "v2_release_chl2_os",
    "A07": "v2_release_chl4_residual_blocks_aligned",
    "A08": "v2_release_chl4d5_orientation_lift_os_replacement",
    "A10": "v2_release_chl2_factor_compatibility",
}

STRATUM_TEX: Mapping[str, str] = {
    "ALL": r"$S_{\mathrm{all}}$",
    "LOW_ONLY_LE58": r"$S_{\mathrm{dense}}$",
    "MID_59_120": r"$S_{\mathrm{trans}}$",
    "MID_121_240": r"$S_{121{:}240}$",
    "MID_121_400": r"$S_{121{:}400}$",
    "NO_58": r"$S_{>58}$",
    "NO_120": r"$S_{>120}$",
    "NO_240": r"$S_{>240}$",
}

# These labels are intentionally distinct from the likelihood strata above:
# A05 filters only on the candidate gap g2, not on max(g1, g2).
G2_FILTER_TEX: Mapping[str, str] = {
    "ALL": r"$B^{(g_2)}_{\mathrm{all}}$",
    "LOW_ONLY_LE58": r"$B^{(g_2)}_{\leq58}$",
    "MID_59_120": r"$B^{(g_2)}_{59{:}120}$",
    "MID_121_240": r"$B^{(g_2)}_{121{:}240}$",
    "MID_121_400": r"$B^{(g_2)}_{121{:}400}$",
    "NO_58": r"$B^{(g_2)}_{>58}$",
    "NO_120": r"$B^{(g_2)}_{>120}$",
    "NO_240": r"$B^{(g_2)}_{>240}$",
}

MODEL_DISPLAY: Mapping[str, str] = {
    "CHL1_ratio_only_cond_eta": "CHL1",
    "CHL2_path_excl_cond_eta": "CHL2",
    "CHL2_gap_excl_cond_eta": "CHL2-gap",
    "noPhi_cond_eta": "Row-mask",
    "noPhi_gap_excl_cond_eta": "Gap-only",
    "HL2_order0_eta": "HL2",
    "HL2_gap_excl_order0_eta": "HL2-gap",
    "Cramer_order0_exp": "Cramér",
    "Cramer_Granville_order0_exp": "Cramér–Granville",
    "Cramer_Granville_gap_excl_order0_exp": "Cramér–Granville-gap",
    "ROW_MASK_ETA": "ROW_MASK_ETA",
    "H2_COND_ETA": "H2_COND_ETA",
    "CHL1": "CHL1",
    "PATH_ONLY_ETA": "PATH_ONLY_ETA",
    "CG_MARKOV_ETA": "CG_MARKOV_ETA",
    "CG_PATH_ETA": "CG_PATH_ETA",
    "CHL2": "CHL2",
    "CG_MARKOV_PATH_ETA": "CG_MARKOV_PATH_ETA",
}


@dataclass(frozen=True)
class RawTex:
    """Cell content that must be passed to LaTeX without escaping."""

    text: str


def raw_tex(text: object) -> RawTex:
    return RawTex(str(text))


@dataclass(frozen=True)
class SourceSpec:
    audit: str
    filename: str

    def resolve(self, outputs_root: Path) -> Path:
        try:
            directory = AUDIT_DIRS[self.audit]
        except KeyError as exc:  # pragma: no cover - programmer error
            raise KeyError(f"Unknown audit key: {self.audit}") from exc
        return outputs_root / directory / self.filename


def portable_path(path: str | Path) -> str:
    """Return a repository-relative path when possible.

    Generated manifests must not leak checkout-specific absolute paths.  Paths
    outside the repository are left unchanged when relative, or normalized to
    POSIX form when absolute.
    """
    p = Path(path)
    try:
        resolved = p.resolve(strict=False)
        return resolved.relative_to(REPO_ROOT).as_posix()
    except (OSError, ValueError):
        return p.as_posix()


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_file(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Required release input is missing: {p}")
    return p


def read_csv_checked(path: str | Path, required_columns: Iterable[str]) -> pd.DataFrame:
    p = require_file(path)
    df = pd.read_csv(p)
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"{p} is missing required columns: {missing}")
    return df


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def finite_float(value: object) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"Expected a finite number, received {value!r}")
    return out


def fmt_fixed(value: object, digits: int = 6) -> str:
    x = float(value)
    if not math.isfinite(x):
        return "--"
    return f"{x:.{digits}f}"


def fmt_share(value: object, digits: int = 6) -> str | RawTex:
    """Format a probability/mass share without rounding nonzero values to zero."""
    x = float(value)
    if not math.isfinite(x):
        return "--"
    rendered = f"{x:.{digits}f}"
    if x != 0.0 and float(rendered) == 0.0:
        return fmt_gain(x, digits)
    return rendered


def fmt_gain(value: object, digits: int = 6) -> str | RawTex:
    x = float(value)
    if not math.isfinite(x):
        return "--"
    if x == 0.0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e4:
        exponent = int(math.floor(math.log10(abs(x))))
        mantissa = x / (10.0**exponent)
        return raw_tex(rf"${mantissa:.{digits}f}\times10^{{{exponent}}}$")
    return f"{x:.{digits}f}"


def fmt_integer(value: object) -> RawTex:
    x = int(round(float(value)))
    sign = "-" if x < 0 else ""
    groups = f"{abs(x):,}".split(",")
    return raw_tex(sign + r"\,".join(groups))


def fmt_count_ratio(numerator: object, denominator: object) -> str:
    return f"{int(numerator)}/{int(denominator)}"


def tex_escape(text: object) -> str:
    if isinstance(text, RawTex):
        return text.text
    s = str(text)
    # Whole-cell math fragments are already TeX.
    if s.startswith("$") and s.endswith("$"):
        return s
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in s)


def render_tabular(
    *,
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
    align: str | None = None,
    sources: Sequence[Path] = (),
    producer: str = "docs/build_tables.py",
) -> str:
    if align is None:
        align = "l" + "r" * (len(headers) - 1)
    if len(align) != len(headers):
        raise ValueError("LaTeX alignment length must match header count")
    lines = [
        f"% Generated by {producer}; do not edit by hand.",
    ]
    for source in sources:
        lines.append(f"% Source: {portable_path(source)}")
        lines.append(f"% Source-SHA256: {sha256_file(source)}")
    lines.extend([
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(tex_escape(h) for h in headers) + r" \\",
        r"\midrule",
    ])
    for row in rows:
        if len(row) != len(headers):
            raise ValueError("LaTeX row length does not match header count")
        lines.append(" & ".join(tex_escape(cell) for cell in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def write_text(path: str | Path, text: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", newline="\n")
    return p


def write_json(path: str | Path, payload: object) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p


def source_record(path: Path) -> dict[str, object]:
    return {
        "path": portable_path(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def output_record(path: Path) -> dict[str, object]:
    return {
        "path": portable_path(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
