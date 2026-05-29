"""Small runtime telemetry helpers for CHL audit scripts.

The telemetry written by this module is intentionally descriptive.  It records
wall-clock runtime, command-line arguments, platform information, and selected
script-specific counters.  It is not used by the mathematical kernels and does
not affect any reported likelihoods or probabilities.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def telemetry_start() -> dict[str, Any]:
    """Return a mutable telemetry dictionary initialized at script start."""
    now = time.time()
    return {
        "started_at_utc": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
        "start_time_unix": now,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "argv": sys.argv,
        "cwd": os.getcwd(),
    }


def _json_default(obj: Any) -> Any:
    """JSON serializer for common scientific-script objects."""
    try:
        import numpy as np  # type: ignore
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "__dict__"):
        return vars(obj)
    return str(obj)


def write_telemetry(path: str | Path, telemetry: Mapping[str, Any], **extra: Any) -> None:
    """Write a telemetry JSON file with elapsed wall-clock seconds.

    Parameters
    ----------
    path:
        Destination JSON file.
    telemetry:
        Dictionary previously returned by :func:`telemetry_start`.
    **extra:
        Extra script-specific counters or paths.
    """
    path = Path(path)
    out = dict(telemetry)
    out.update(extra)
    end = time.time()
    out["finished_at_utc"] = datetime.fromtimestamp(end, tz=timezone.utc).isoformat()
    out["end_time_unix"] = end
    if "start_time_unix" in out:
        out["elapsed_seconds"] = float(end - float(out["start_time_unix"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=_json_default, sort_keys=True)
