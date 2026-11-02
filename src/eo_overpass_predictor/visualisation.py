"""Optional notebook maps and timelines; plotting libraries are imported only on use."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import geopandas as gpd
from shapely.geometry import mapping

from ._common import SATELLITE_CATALOGUE
from .roi import _roi_anchor

if TYPE_CHECKING:
    import folium


def visualise_roi_on_map(
    roi_gdf: gpd.GeoDataFrame,
    zoom_start: int = 8,
    tiles: str = "Esri.WorldImagery",
    polygon_color: str = "#FF3B30",
    save_path: Optional[str] = None,
) -> folium.Map:

    """
    Generate an interactive Folium map of the ROI for spatial verification.

    Requires the ``notebooks`` extra.

    Args:
        roi_gdf (gpd.GeoDataFrame): ROI from ``load_roi``.
        zoom_start (int, optional): Initial zoom (overridden by fit-to-bounds). Defaults to 8.
        tiles (str, optional): Folium / xyzservices tile name. Defaults to "Esri.WorldImagery".
        polygon_color (str, optional): ROI outline colour. Defaults to "#FF3B30".
        save_path (str, optional): Export the map to a standalone HTML file.

    Returns:
        folium.Map: Interactive map centred on the ROI.
    """

    import folium

    geom = roi_gdf.geometry.iloc[0]
    a = _roi_anchor(geom)
    m = folium.Map(location=[a.y, a.x], zoom_start=zoom_start, tiles=tiles)
    if geom.geom_type == "Point":
        folium.CircleMarker([a.y, a.x], radius=7, color=polygon_color, fill=True,
                            tooltip="ROI point").add_to(m)
    else:
        folium.GeoJson(mapping(geom), name="ROI", tooltip="ROI",
                       style_function=lambda _: {"color": polygon_color, "weight": 2,
                                                 "fillOpacity": 0.1}).add_to(m)
        minx, miny, maxx, maxy = geom.bounds
        m.fit_bounds([[miny, minx], [maxy, maxx]], padding=(40, 40))
    if save_path:
        m.save(save_path)
    return m


def visualise_overpass_footprints(
    passes: gpd.GeoDataFrame,
    roi_gdf: gpd.GeoDataFrame,
    tiles: str = "Esri.WorldImagery",
    expected_only: bool = True,
    max_passes: Optional[int] = None,
    save_path: Optional[str] = None,
) -> folium.Map:

    """
    Map the predicted swath strips over the ROI, one toggleable layer per satellite.

    Requires the ``notebooks`` extra.

    Hovering a strip shows satellite, local and UTC times, direction, ROI coverage
    and view angle.

    Args:
        passes (gpd.GeoDataFrame): Output of ``predict_overpasses``.
        roi_gdf (gpd.GeoDataFrame): ROI from ``load_roi``.
        tiles (str, optional): Basemap. Defaults to "Esri.WorldImagery".
        expected_only (bool, optional): Draw only passes where imaging is expected.
        max_passes (int, optional): Draw only the next N passes.
        save_path (str, optional): Export to standalone HTML.

    Returns:
        folium.Map: Interactive footprint map.
    """

    import folium

    m = visualise_roi_on_map(roi_gdf, tiles=tiles)
    p = passes[passes["imaging_expected"]] if expected_only else passes
    if max_passes:
        p = p.head(max_passes)
    tz = passes["timezone"].iloc[0] if len(passes) else ""

    for sat, grp in p.groupby("satellite", sort=False):
        colour = SATELLITE_CATALOGUE.get(sat, {}).get("colour", "#00E5FF")
        fg = folium.FeatureGroup(name=f"{sat} ({len(grp)})")
        for _, r in grp.iterrows():
            tip = (f"<b>{sat}</b><br>{r['overpass_local']:%a %d %b %Y %H:%M:%S} ({tz})"
                   f"<br>{r['overpass_utc']:%Y-%m-%d %H:%M:%S} UTC<br>{r['direction']} · "
                   f"ROI cover {r['roi_coverage_pct']:.0f}% · view {r['view_angle_deg']:.1f}°")
            if "esa_plan_status" in r:
                tip += f"<br>ESA plan: {r['esa_plan_status']}"
            folium.GeoJson(mapping(r.geometry), tooltip=folium.Tooltip(tip),
                           style_function=lambda _, c=colour: {"color": c, "weight": 1.5,
                                                               "fillColor": c, "fillOpacity": 0.12}
                           ).add_to(fg)
        fg.add_to(m)
    if len(p):
        minx, miny, maxx, maxy = p.total_bounds
        m.fit_bounds([[miny, minx], [maxy, maxx]], padding=(20, 20))
    folium.LayerControl(collapsed=False).add_to(m)
    if save_path:
        m.save(save_path)
    return m


def plot_overpass_timeline(passes: gpd.GeoDataFrame, save_path: Optional[str] = None, ax=None):

    """
    Plot a pass timeline: one row per satellite, local date on x, time-of-day annotated.

    Requires the ``notebooks`` extra.

    Filled markers are expected acquisitions; hollow markers are geometric passes
    flagged as not expected (only present with ``daytime_descending_only=False``)
    or not in the ESA plan (after ``crosscheck_s2_acquisition_plan``). Marker
    size scales with ROI coverage.

    Args:
        passes (gpd.GeoDataFrame): Output of ``predict_overpasses``.
        save_path (str, optional): Save the figure (PNG/SVG/PDF).
        ax (matplotlib Axes, optional): Draw into an existing axes.

    Returns:
        matplotlib.figure.Figure
    """

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    sats = [s for s in SATELLITE_CATALOGUE if s in set(passes["satellite"])]
    tz = passes["timezone"].iloc[0]
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 0.55 * len(sats) + 1.6))
    else:
        fig = ax.figure

    for y, sat in enumerate(sats):
        g = passes[passes["satellite"] == sat]
        filled = g["imaging_expected"].copy()
        if "esa_plan_status" in g:
            filled &= ~g["esa_plan_status"].str.startswith("not planned")
        x = g["overpass_local"].dt.tz_localize(None)
        c = SATELLITE_CATALOGUE[sat]["colour"]
        size = 30 + 1.4 * g["roi_coverage_pct"]
        ax.scatter(x[filled], [y] * filled.sum(), s=size[filled], color=c, zorder=3)
        ax.scatter(x[~filled], [y] * (~filled).sum(), s=size[~filled], facecolor="none",
                   edgecolor=c, linewidth=1.2, zorder=3)
        for xi, t in zip(x, g["overpass_local"]):
            ax.annotate(t.strftime("%H:%M"), (xi, y), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=7, color="#555")
    ax.set_yticks(range(len(sats)), sats)
    ax.set_ylim(-0.7, len(sats) - 0.3)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d\n%b"))
    ax.grid(axis="x", color="#ddd", linewidth=0.6)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    roi = passes.attrs.get("roi_name", "ROI")
    ax.set_title(f"Predicted overpasses — {roi} (local time, {tz})", loc="left", fontsize=11)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
