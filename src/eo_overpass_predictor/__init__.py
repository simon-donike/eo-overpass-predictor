"""Swath-aware Landsat and Sentinel-2 prediction, reporting and live tracking.

The public functions retain the notebook helper's names and inputs. Plotting and
widget dependencies are loaded only when used; install the ``notebooks`` extra
for interactive maps, timelines and the Jupyter tracker.
"""

from ._common import SATELLITE_CATALOGUE, process_timer
from .esa_plans import (
    crosscheck_s2_acquisition_plan,
    fetch_s2_acquisition_plans,
    load_s2_acquisition_plans,
)
from .export import (
    export_overpass_csv,
    export_overpass_footprints,
    next_overpasses,
    summarise_overpasses,
)
from .predict import (
    predict_overpasses,
    satellite_positions,
    solar_elevation_deg,
    subsolar_point,
)
from .roi import load_roi, resolve_roi_timezone
from .tle import fetch_tles, normalise_tle_line, parse_tle_text
from .tracker import LiveTracker, export_live_tracker_html, live_tracker
from .visualisation import (
    plot_overpass_timeline,
    visualise_overpass_footprints,
    visualise_roi_on_map,
)

__all__ = [
    "SATELLITE_CATALOGUE",
    "LiveTracker",
    "crosscheck_s2_acquisition_plan",
    "export_live_tracker_html",
    "export_overpass_csv",
    "export_overpass_footprints",
    "fetch_s2_acquisition_plans",
    "fetch_tles",
    "live_tracker",
    "load_roi",
    "load_s2_acquisition_plans",
    "next_overpasses",
    "normalise_tle_line",
    "parse_tle_text",
    "plot_overpass_timeline",
    "predict_overpasses",
    "process_timer",
    "resolve_roi_timezone",
    "satellite_positions",
    "solar_elevation_deg",
    "subsolar_point",
    "summarise_overpasses",
    "visualise_overpass_footprints",
    "visualise_roi_on_map",
]
