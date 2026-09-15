"""
Lunar map visualization.

Draws the image footprint and the correspondence region on a simple lunar
basemap ONLY when real geospatial metadata is available (see
src/geospatial/coords.py). This is an optional visualization layer, not part
of the core matching algorithm, per the project scope.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.geospatial.coords import GeoMetadata


def plot_footprint_on_basemap(geo: GeoMetadata, correspondence_bounds=None):
    """
    Returns a matplotlib Figure showing the reference image's lat/lon
    footprint as a rectangle on a blank lunar-longitude/latitude grid.

    If geo.available is False, returns None — callers must show the
    "Geospatial metadata unavailable for this dataset." message instead of
    calling this function.
    """
    if not geo.available:
        return None

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor("#111111")
    fig.patch.set_facecolor("#111111")

    lon_min = min(geo.lon_top_left, geo.lon_bottom_right)
    lon_max = max(geo.lon_top_left, geo.lon_bottom_right)
    lat_min = min(geo.lat_top_left, geo.lat_bottom_right)
    lat_max = max(geo.lat_top_left, geo.lat_bottom_right)

    pad_lon = max((lon_max - lon_min) * 2, 1.0)
    pad_lat = max((lat_max - lat_min) * 2, 1.0)

    ax.set_xlim(lon_min - pad_lon, lon_max + pad_lon)
    ax.set_ylim(lat_min - pad_lat, lat_max + pad_lat)
    ax.set_xlabel("Selenographic Longitude (deg)", color="white")
    ax.set_ylabel("Selenographic Latitude (deg)", color="white")
    ax.tick_params(colors="white")
    ax.grid(True, alpha=0.2, color="gray")

    rect = patches.Rectangle(
        (lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
        linewidth=2, edgecolor="#00E5FF", facecolor="#00E5FF", alpha=0.25,
        label="Image footprint",
    )
    ax.add_patch(rect)

    if correspondence_bounds is not None:
        clon_min, clat_min, clon_max, clat_max = correspondence_bounds
        rect2 = patches.Rectangle(
            (clon_min, clat_min), clon_max - clon_min, clat_max - clat_min,
            linewidth=2, edgecolor="#FFD400", facecolor="none",
            label="Correspondence region",
        )
        ax.add_patch(rect2)

    ax.legend(loc="upper right", facecolor="#222222", labelcolor="white")
    ax.set_title("Lunar Footprint Map", color="white")
    return fig
