"""Compatibility imports for notebooks written before the package restructuring.

Install the project first (``uv sync --extra notebooks`` or ``pip install -e .``).
New code should import from ``eo_overpass_predictor``. This shim is retained for
one release; prediction and export implementations live in the package.
"""

from eo_overpass_predictor import *  # noqa: F403
from eo_overpass_predictor import __all__ as _public_names
from eo_overpass_predictor._common import (
    CELESTRAK_GP_URL,
    EARTH_RADIUS_KM,
    ESA_BASE_URL,
    ESA_S2_PLAN_PAGE,
    GEOD,
    TLE_MIRROR_URL,
)

__all__ = [
    *_public_names,
    "CELESTRAK_GP_URL",
    "EARTH_RADIUS_KM",
    "ESA_BASE_URL",
    "ESA_S2_PLAN_PAGE",
    "GEOD",
    "TLE_MIRROR_URL",
]
