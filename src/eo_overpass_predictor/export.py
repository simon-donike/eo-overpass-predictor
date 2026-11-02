"""Summary tables, next-pass countdowns and compatible CSV/vector exports."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import pandas as pd

from ._common import _to_utc_timestamp


def summarise_overpasses(passes: gpd.GeoDataFrame) -> pd.DataFrame:

    """
    Tabulate predicted overpasses in a compact, human-readable form.

    Args:
        passes (gpd.GeoDataFrame): Output of ``predict_overpasses``.

    Returns:
        pd.DataFrame: Satellite, local day/date, local time window, UTC time,
        direction, ROI coverage, off-nadir angle, sun elevation, expected flag and
        (if present) the ESA plan status.
    """

    if passes.empty:
        return pd.DataFrame()
    tz = passes["timezone"].iloc[0]
    df = pd.DataFrame({
        "Satellite": passes["satellite"],
        "Local date": passes["overpass_local"].dt.strftime("%a %d %b %Y"),
        f"Local time ({tz})": passes["roi_entry_local"].dt.strftime("%H:%M:%S") + " – "
                               + passes["roi_exit_local"].dt.strftime("%H:%M:%S"),
        "UTC offset": passes["utc_offset"],
        "Overpass (UTC)": passes["overpass_utc"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "Direction": passes["direction"],
        "ROI cover %": passes["roi_coverage_pct"],
        "View angle °": passes["view_angle_deg"],
        "Sun elev °": passes["sun_elevation_deg"],
        "Expected": passes["imaging_expected"].map({True: "✔", False: "✘"}),
    })
    if "esa_plan_status" in passes:
        df["ESA plan"] = passes["esa_plan_status"]
    return df


def next_overpasses(passes: gpd.GeoDataFrame, now_utc=None) -> pd.DataFrame:
    """Next upcoming overpass per satellite with a live countdown (n2yo-style)."""
    now = _to_utc_timestamp(now_utc) if now_utc is not None else pd.Timestamp.now(tz="UTC")
    up = passes[passes["roi_exit_utc"] >= now]
    if up.empty:
        return pd.DataFrame()
    nxt = up.sort_values("overpass_utc").groupby("satellite", sort=False).head(1)

    def _cd(t):
        s = int((t - now).total_seconds())
        if s <= 0:
            return "imaging now"
        d, r = divmod(s, 86400)
        h, r = divmod(r, 3600)
        return f"{d}d {h:02d}h {r // 60:02d}m"

    return pd.DataFrame({
        "Satellite": nxt["satellite"].values,
        "Countdown": [_cd(t) for t in nxt["roi_entry_utc"]],
        "Local": nxt["overpass_local"].dt.strftime("%a %d %b %H:%M:%S").values,
        "UTC": nxt["overpass_utc"].dt.strftime("%Y-%m-%d %H:%M:%S").values,
        "ROI cover %": nxt["roi_coverage_pct"].values,
    })


def export_overpass_csv(
    passes: gpd.GeoDataFrame,
    out_dir: Union[str, Path] = "OUTPUT",
    file_name: Optional[str] = None,
    expected_only: bool = False,
) -> Path:

    """
    Write the overpass report to CSV with both UTC and ROI-local times.

    Times are written as ISO-8601 strings: UTC columns end in ``Z``; local columns
    carry their explicit offset (``+11:00``) so the file is unambiguous across DST
    changes, plus split ``local_date`` / ``local_time`` columns for spreadsheets.
    The swath geometry is not written (see ``export_overpass_footprints``).

    Args:
        passes (gpd.GeoDataFrame): Output of ``predict_overpasses`` (optionally cross-checked).
        out_dir (str | Path, optional): Output folder. Defaults to "OUTPUT".
        file_name (str, optional): Override the file name. Defaults to
            ``overpasses_<roi>_<start>_<end>.csv``.
        expected_only (bool, optional): Write only rows with ``imaging_expected``.

    Returns:
        Path: Path of the written CSV.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(passes.drop(columns="geometry")).copy()
    if expected_only:
        df = df[df["imaging_expected"]]

    for c in ["roi_entry_utc", "overpass_utc", "roi_exit_utc", "tle_epoch_utc"]:
        if c in df:
            df[c] = df[c].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    for c in ["roi_entry_local", "overpass_local", "roi_exit_local"]:
        if c in df:
            df[c] = df[c].apply(lambda t: t.isoformat(timespec="seconds"))
    if len(df):
        loc = passes.loc[df.index, "overpass_local"]
        df.insert(df.columns.get_loc("timezone"), "local_date", loc.dt.strftime("%Y-%m-%d").values)
        df.insert(df.columns.get_loc("timezone"), "local_time", loc.dt.strftime("%H:%M:%S").values)
        df.insert(df.columns.get_loc("timezone"), "local_weekday", loc.dt.strftime("%A").values)

    if file_name is None:
        roi = re.sub(r"[^A-Za-z0-9_-]+", "_", str(passes.attrs.get("roi_name", "ROI")))
        s = passes.attrs.get("window_start_utc")
        e = passes.attrs.get("window_end_utc")
        span = f"_{s:%Y%m%d}_{e:%Y%m%d}" if s is not None and e is not None else ""
        file_name = f"overpasses_{roi}{span}.csv"
    path = out_dir / file_name
    df.to_csv(path, index=False)
    print(f"💾 {len(df)} overpass(es) written to {path}")
    return path


def export_overpass_footprints(passes: gpd.GeoDataFrame, path: Union[str, Path]) -> Path:
    """Write the per-pass swath strips to a GeoPackage / GeoJSON / shapefile for QGIS or ArcGIS."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    g = passes.copy()
    for c in g.columns:
        if pd.api.types.is_datetime64_any_dtype(g[c]):
            g[c] = g[c].astype(str)
    g.to_file(path)
    print(f"💾 {len(g)} footprint(s) written to {path}")
    return path
