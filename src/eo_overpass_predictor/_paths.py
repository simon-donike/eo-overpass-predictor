"""Default cache locations, resolved at call time without creating directories."""

import os
import sys
from pathlib import Path


def _cache_dir(subdirectory: str) -> Path:
    """Return an application cache path using the host platform's conventions.

    Explicit paths passed to the fetching/loading functions always take precedence.
    Relative environment paths are ignored so defaults never depend on the cwd.
    """
    if sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        fallback = Path.home() / "AppData" / "Local"
    elif sys.platform == "darwin":
        root = fallback = Path.home() / "Library" / "Caches"
    else:
        fallback = Path.home() / ".cache"
        root = Path(os.environ.get("XDG_CACHE_HOME") or fallback)
    if not root.is_absolute():
        root = fallback
    return root / "eo-overpass-predictor" / subdirectory
