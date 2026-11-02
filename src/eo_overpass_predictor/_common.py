"""Mission definitions, shared time handling and notebook timing utilities."""

from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
from pyproj import Geod
from skyfield.api import load

CELESTRAK_GP_URL = "https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=TLE"
# Community CelesTrak mirror, used only if the primary service fails.
TLE_MIRROR_URL = "https://tle.ivanstanojevic.me/api/tle/{norad_id}"
ESA_S2_PLAN_PAGE = "https://sentinels.copernicus.eu/copernicus/sentinel-2/acquisition-plans"
ESA_BASE_URL = "https://sentinels.copernicus.eu"

EARTH_RADIUS_KM = 6371.0088
GEOD = Geod(ellps="WGS84")

# Swath widths are the full cross-track widths of nadir-looking instruments.
SATELLITE_CATALOGUE = {
    "LANDSAT 8": {
        "norad_id": 39084, "mission": "Landsat", "sensor": "OLI/TIRS",
        "swath_km": 185.0, "lat_limits": (-81.8, 81.8), "colour": "#F28E2B",
        "note": "USGS acquires ~all daytime land and near-coastal WRS-2 scenes.",
    },
    "LANDSAT 9": {
        "norad_id": 49260, "mission": "Landsat", "sensor": "OLI-2/TIRS-2",
        "swath_km": 185.0, "lat_limits": (-81.8, 81.8), "colour": "#E15759",
        "note": "USGS acquires ~all daytime land and near-coastal WRS-2 scenes.",
    },
    "SENTINEL-2A": {
        "norad_id": 40697, "mission": "Sentinel-2", "sensor": "MSI",
        "swath_km": 290.0, "lat_limits": (-56.0, 82.8), "colour": "#76B7B2",
        "note": "Extension campaign (to 31 Dec 2026): selected regions only - check the ESA plan.",
    },
    "SENTINEL-2B": {
        "norad_id": 42063, "mission": "Sentinel-2", "sensor": "MSI",
        "swath_km": 290.0, "lat_limits": (-56.0, 82.8), "colour": "#4E79A7",
        "note": "Nominal constellation (with S2C). Open ocean is not acquired.",
    },
    "SENTINEL-2C": {
        "norad_id": 60989, "mission": "Sentinel-2", "sensor": "MSI",
        "swath_km": 290.0, "lat_limits": (-56.0, 82.8), "colour": "#59A14F",
        "note": "Nominal constellation (with S2B). Open ocean is not acquired.",
    },
}


_TIMESCALE = None


def _ts():
    """Return a cached Skyfield timescale built from bundled data (no network needed)."""
    global _TIMESCALE
    if _TIMESCALE is None:
        _TIMESCALE = load.timescale(builtin=True)
    return _TIMESCALE


def _to_utc_timestamp(value) -> pd.Timestamp:
    """Coerce str / datetime / Timestamp (naive = UTC) to a tz-aware UTC Timestamp."""
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _resolve_satellites(satellites) -> list:
    """Normalise a satellite selection to catalogue keys (accepts names or NORAD ids)."""
    if satellites is None:
        return list(SATELLITE_CATALOGUE)
    by_id = {v["norad_id"]: k for k, v in SATELLITE_CATALOGUE.items()}
    out = []
    for s in satellites:
        if isinstance(s, (int, np.integer)) and int(s) in by_id:
            out.append(by_id[int(s)])
            continue
        key = str(s).upper().replace("_", " ").strip()
        key = {"L8": "LANDSAT 8", "L9": "LANDSAT 9", "S2A": "SENTINEL-2A",
               "S2B": "SENTINEL-2B", "S2C": "SENTINEL-2C"}.get(key, key)
        if key not in SATELLITE_CATALOGUE:
            raise ValueError(f"Unknown satellite '{s}'. Known: {list(SATELLITE_CATALOGUE)}")
        out.append(key)
    return out


@contextmanager
def process_timer(process_name="Process"):

    """
    Measure and report the execution time of a code block using a context manager.

    Records start and end wall-clock timestamps, prints readable timing
    information and the total elapsed duration in a human-friendly format.

    Args:
        process_name (str, optional): Descriptive name of the process being timed.
            Defaults to "Process".

    Yields:
        None: Control is passed to the wrapped block inside the ``with`` statement.
    """

    start_time = time.time()
    print(f"\n--- Starting {process_name} ---")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), elapsed % 60
        print(f"\n--- {process_name} Complete ---")
        if h:
            msg = f"{h}h {m}m {int(s)}s"
        elif m:
            msg = f"{m}m {int(s)}s"
        else:
            msg = f"{s:.1f}s"
        print("\x1b[92m" + f"Elapsed time: {msg}" + "\x1b[0m")
