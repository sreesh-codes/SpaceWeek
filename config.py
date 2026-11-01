"""Shared settings for all Bill of Orbit data scripts.

Edit the AOI boxes after eyeballing them in Copernicus Browser — they are
approximate boxes around each port/anchorage. Keep each box under ~0.18 deg
per side so Earth Engine direct downloads stay under its size limit.
"""
import os

# (lon_min, lat_min, lon_max, lat_max)
AOIS = {
    "fujairah_anchorage": (56.36, 25.05, 56.52, 25.22),
    "khor_fakkan":        (56.34, 25.31, 56.48, 25.42),
    "sohar":              (56.58, 24.42, 56.74, 24.56),
}

# Imagery / jamming window (UTC dates, end exclusive)
START = "2026-09-01"
END   = "2026-11-01"

# UTM zone 40N covers all three sites
CRS = "EPSG:32640"

# Box used to keep only Gulf-region jamming cells
GULF_BBOX = (50.0, 22.0, 60.0, 28.0)  # lon_min, lat_min, lon_max, lat_max

# aisstream bounding box: [[lat, lon], [lat, lon]] (opposite corners)
AIS_BOXES = [[[24.30, 56.20], [25.70, 57.00]]]

# Secrets come from environment variables, never hard-code them
GEE_PROJECT = os.environ.get("GEE_PROJECT", "")
GFW_TOKEN = os.environ.get("GFW_TOKEN", "")
AISSTREAM_API_KEY = os.environ.get("AISSTREAM_API_KEY", "")


def polygon(aoi):
    """GeoJSON polygon for an AOI tuple."""
    x0, y0, x1, y1 = aoi
    return {
        "type": "Polygon",
        "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
    }
