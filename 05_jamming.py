"""GPS jamming layer for the Gulf from Stanford's GNSS interference monitor
(aircraft ADS-B integrity, H3 hex cells, daily + hourly).

Needs Python 3.8-3.11 (the package pins h3==3.7.6). Use a separate venv:

    python3.11 -m venv venv-jam && source venv-jam/bin/activate
    pip install rfi-fileparser pandas
    python 05_jamming.py --start 2026-10-01 --end 2026-10-31

Tip: start with just the dates of the Sentinel-1 passes you picked —
a full month can take a while to download.
"""
import argparse
import csv
import datetime as dt
import glob
import json
import os

import h3
from rfi_fileparser import downloader

from config import GULF_BBOX

ROOT = "downloaded_json_files/jamming"  # where rfi_fileparser saves files
OUT = "data/jamming"


def centroid(h):
    if hasattr(h3, "h3_to_geo"):        # h3 v3
        return h3.h3_to_geo(h)
    return h3.cell_to_latlng(h)          # h3 v4


def find_cells(obj):
    """Yield every dict that has an h3Index, wherever it sits in the JSON."""
    if isinstance(obj, dict):
        if "h3Index" in obj:
            yield obj
        for v in obj.values():
            yield from find_cells(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from find_cells(v)


def in_gulf(lat, lon):
    x0, y0, x1, y1 = GULF_BBOX
    return x0 <= lon <= x1 and y0 <= lat <= y1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()
    s = dt.date.fromisoformat(args.start)
    e = dt.date.fromisoformat(args.end)

    if not args.skip_download:
        downloader.download_files(s.strftime("%Y/%m/%d"), e.strftime("%Y/%m/%d"), "jamming")

    os.makedirs(OUT, exist_ok=True)
    rows = []
    for path in glob.glob(os.path.join(ROOT, "*", "*", "*", "**", "heatmap.json"), recursive=True):
        parts = os.path.relpath(path, ROOT).split(os.sep)
        date = "-".join(parts[:3])
        hour = parts[3] if len(parts) == 5 else "daily"
        with open(path) as f:
            js = json.load(f)
        for c in find_cells(js):
            lat, lon = centroid(c["h3Index"])
            if not in_gulf(lat, lon):
                continue
            low = c.get("lowQualityCount", 0) or 0
            tot = c.get("totalAircraftCount", 0) or 0
            rows.append({
                "date": date, "hour": hour, "h3": c["h3Index"],
                "lat": round(lat, 4), "lon": round(lon, 4),
                "low_quality": low, "total_aircraft": tot,
                # same false-alarm damping GPSJam uses: max(0, bad-1)/total
                "jam_score": round(max(0, low - 1) / tot, 3) if tot else None,
            })

    out = os.path.join(OUT, "gulf_jamming_cells.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "hour", "h3", "lat", "lon",
                                          "low_quality", "total_aircraft", "jam_score"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} Gulf cell-rows -> {out}")
    if not rows:
        print("No Gulf cells found. Check the dates, or fall back to gpsjam.org "
              "(daily map, CC-BY) for the same dates.")


if __name__ == "__main__":
    main()
