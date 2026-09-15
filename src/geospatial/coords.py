"""
Geospatial coordinate handling.

Chandrayaan-2 OHRC/TMC-2/IIRS products distributed via ISRO's Pradan portal
carry geolocation metadata (e.g. embedded in PDS3/PDS4 labels or accompanying
.xml/.lbl files) that maps image pixel coordinates to selenographic
(lunar) latitude/longitude. This prototype does NOT ship that parser yet.

This module never fabricates coordinates. If no metadata file is supplied,
every function here returns `None` / an explicit "unavailable" status instead
of guessing.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os


@dataclass
class GeoMetadata:
    available: bool
    lat_top_left: float | None = None
    lon_top_left: float | None = None
    lat_bottom_right: float | None = None
    lon_bottom_right: float | None = None
    source_file: str | None = None
    note: str = ""


def load_geo_metadata(metadata_path: str | None) -> GeoMetadata:
    """
    Attempt to load a simple JSON sidecar with the following optional shape:
        {"lat_top_left": .., "lon_top_left": .., "lat_bottom_right": .., "lon_bottom_right": ..}

    Real Chandrayaan-2 PDS label parsing is listed under FUTURE WORK — see
    LIMITATIONS.md. Returns GeoMetadata(available=False) if nothing usable
    is found; never invents values.
    """
    if not metadata_path or not os.path.exists(metadata_path):
        return GeoMetadata(available=False, note="Geospatial metadata unavailable for this dataset.")

    try:
        with open(metadata_path, "r") as f:
            data = json.load(f)
        required = ["lat_top_left", "lon_top_left", "lat_bottom_right", "lon_bottom_right"]
        if not all(k in data for k in required):
            return GeoMetadata(available=False, note="Metadata file missing required fields.")
        return GeoMetadata(
            available=True,
            lat_top_left=data["lat_top_left"], lon_top_left=data["lon_top_left"],
            lat_bottom_right=data["lat_bottom_right"], lon_bottom_right=data["lon_bottom_right"],
            source_file=metadata_path,
            note="Loaded from user-supplied metadata sidecar.",
        )
    except Exception as e:
        return GeoMetadata(available=False, note=f"Failed to parse metadata file: {e}")


def pixel_to_latlon(x: float, y: float, image_shape, geo: GeoMetadata):
    """Bilinear interpolation of lat/lon across the image footprint. Returns None if unavailable."""
    if not geo.available:
        return None
    h, w = image_shape[:2]
    fx, fy = x / w, y / h
    lat = geo.lat_top_left + fy * (geo.lat_bottom_right - geo.lat_top_left)
    lon = geo.lon_top_left + fx * (geo.lon_bottom_right - geo.lon_top_left)
    return lat, lon
