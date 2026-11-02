"""ROI loading, local projections and offline IANA time-zone lookup."""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import geopandas as gpd
import numpy as np
import shapely
from pyproj import CRS, Transformer
from shapely import wkt as shp_wkt
from shapely.geometry import Point, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform, unary_union
from timezonefinder import TimezoneFinder

_TZ_FINDER = None


def load_roi(
    source,
    use_bbox: bool = False,
    buffer_km: float = 0.0,
    name: Optional[str] = None,
) -> gpd.GeoDataFrame:

    """
    Load a Region of Interest from almost any input and return it as a single EPSG:4326 geometry.

    Accepted inputs:
        * Path / str to any vector file GeoPandas can read (any CRS — reprojected).
        * A GeoDataFrame or GeoSeries (any CRS).
        * A shapely geometry or WKT string (assumed EPSG:4326).
        * ``(lon, lat)`` for a point ROI.
        * ``(min_lon, min_lat, max_lon, max_lat)`` for a bounding box.

    Multi-feature inputs are dissolved into one geometry. Invalid geometries are
    repaired. A point ROI is valid — the pass test then answers "is this point
    inside the swath" and coverage is reported as 0 or 100 %.

    Args:
        source: The ROI input (see above).
        use_bbox (bool, optional): Replace the geometry by its bounding box. Defaults to False.
        buffer_km (float, optional): Geodesic-ish buffer applied in a local equal-area
            projection, e.g. to catch scenes that only clip the ROI edge. Defaults to 0.
        name (str, optional): Label stored in the ``name`` column (used in output file names).

    Returns:
        gpd.GeoDataFrame: One row, EPSG:4326, columns ``name`` and ``geometry``.
    """

    if isinstance(source, gpd.GeoDataFrame):
        gdf = source.copy()
    elif isinstance(source, gpd.GeoSeries):
        gdf = gpd.GeoDataFrame(geometry=source)
    elif isinstance(source, BaseGeometry):
        gdf = gpd.GeoDataFrame(geometry=[source], crs="EPSG:4326")
    elif isinstance(source, (tuple, list)) and len(source) in (2, 4) and all(
            isinstance(v, (int, float, np.floating)) for v in source):
        geom = Point(source) if len(source) == 2 else box(*source)
        gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
    elif isinstance(source, str) and source.strip().upper().startswith(
            ("POINT", "POLYGON", "MULTIPOLYGON", "LINESTRING")):
        gdf = gpd.GeoDataFrame(geometry=[shp_wkt.loads(source)], crs="EPSG:4326")
    else:
        path = Path(source)
        gdf = gpd.read_file(path)
        name = name or path.stem

    if gdf.crs is None:
        warnings.warn("ROI has no CRS; assuming EPSG:4326.")
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs("EPSG:4326")
    geom = shapely.make_valid(unary_union([g for g in gdf.geometry if g is not None and not g.is_empty]))
    if geom.is_empty:
        raise ValueError("ROI geometry is empty after cleaning.")

    if use_bbox:
        geom = box(*geom.bounds)
    if buffer_km:
        fwd, inv = _local_equal_area(geom)
        geom = transform(inv, transform(fwd, geom).buffer(buffer_km * 1000.0))

    return gpd.GeoDataFrame({"name": [name or "ROI"]}, geometry=[geom], crs="EPSG:4326")


def _roi_anchor(geom: BaseGeometry) -> Point:
    """Centroid if it lies inside the ROI, otherwise a guaranteed interior point."""
    c = geom.centroid
    return c if geom.covers(c) else geom.representative_point()


def _local_equal_area(geom: BaseGeometry):
    """Forward/inverse transformers for a Lambert azimuthal equal-area CRS centred on the ROI (metres)."""
    a = _roi_anchor(geom)
    crs = CRS.from_proj4(f"+proj=laea +lat_0={a.y} +lon_0={a.x} +datum=WGS84 +units=m +no_defs")
    fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform
    return fwd, inv


def resolve_roi_timezone(roi_gdf: gpd.GeoDataFrame, override: Optional[str] = None, verbose: bool = True) -> str:

    """
    Resolve the IANA time zone of the ROI (e.g. "Australia/Melbourne").

    The zone is looked up at the ROI anchor point with ``timezonefinder`` (offline
    polygon database). Using an IANA zone rather than a fixed offset means
    daylight-saving transitions inside the prediction window are handled
    per-pass. The ROI corners are also checked: if they fall in different zones
    a warning is printed and the anchor-point zone is used. Offshore ROIs
    resolve to the nautical ``Etc/GMT±N`` zone.

    Args:
        roi_gdf (gpd.GeoDataFrame): ROI from ``load_roi``.
        override (str, optional): Force a specific IANA zone name.
        verbose (bool, optional): Print the resolved zone. Defaults to True.

    Returns:
        str: IANA time-zone name.
    """

    global _TZ_FINDER
    if override:
        ZoneInfo(override)                      # validates the name
        tz_name = override
    else:
        if _TZ_FINDER is None:
            _TZ_FINDER = TimezoneFinder()
        geom = roi_gdf.geometry.iloc[0]
        a = _roi_anchor(geom)
        tz_name = _TZ_FINDER.timezone_at(lng=a.x, lat=a.y)
        if tz_name is None:
            offset = int(round(a.x / 15.0))
            tz_name = "Etc/GMT" + (f"{-offset:+d}" if offset else "")
        minx, miny, maxx, maxy = geom.bounds
        corners = {_TZ_FINDER.timezone_at(lng=x, lat=y) for x, y in
                   [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]} - {None}
        if len(corners | {tz_name}) > 1 and verbose:
            print(f"\x1b[93m⚠ ROI spans several time zones {sorted(corners | {tz_name})}; "
                  f"using '{tz_name}' (ROI anchor). Pass override= to choose another.\x1b[0m")
    if verbose:
        now = datetime.now(ZoneInfo(tz_name))
        print(f"🕒 ROI time zone: {tz_name} (currently UTC{now.strftime('%z')[:3]}:{now.strftime('%z')[3:]}, {now.tzname()})")
    return tz_name
