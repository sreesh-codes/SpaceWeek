"""Reference Sentinel-1 ship detections from Global Fishing Watch (4Wings API).

Gives gridded daily detection counts per AOI, split into all detections and
'dark' ones (no matching AIS). Data runs from 2017 to ~5 days ago.

    export GFW_TOKEN=...      # from your GFW API access token page
    python 03_gfw_sar.py
    python 03_gfw_sar.py --start 2026-10-01 --end 2026-10-27
"""
import argparse
import csv
import datetime as dt
import os
import time

import requests

from config import AOIS, START, GFW_TOKEN, polygon

URL = "https://gateway.api.globalfishingwatch.org/v3/4wings/report"
OUT = "data/gfw"


def report(geojson, start, end, dark=False, retries=3):
    params = {
        "spatial-resolution": "HIGH",
        "temporal-resolution": "DAILY",
        "spatial-aggregation": "false",
        "datasets[0]": "public-global-sar-presence:latest",
        "date-range": f"{start},{end}",
        "format": "JSON",
    }
    if dark:
        params["filters[0]"] = "matched='false'"
    headers = {"Authorization": f"Bearer {GFW_TOKEN}", "Content-Type": "application/json"}
    for attempt in range(retries):
        r = requests.post(URL, params=params, json={"geojson": geojson}, headers=headers, timeout=200)
        if r.status_code in (429, 524):  # rate limit / gateway timeout
            wait = 30 * (attempt + 1)
            print(f"    HTTP {r.status_code}, retrying in {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("GFW report failed after retries")


def flatten(js):
    rows = []
    for entry in js.get("entries", []):
        for dataset, items in entry.items():
            for it in items or []:
                rows.append({"dataset": dataset, **it})
    return rows


def main():
    if not GFW_TOKEN:
        raise SystemExit("Set GFW_TOKEN env var")
    default_end = (dt.date.today() - dt.timedelta(days=5)).isoformat()
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=default_end)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    for name, box in AOIS.items():
        for dark in (False, True):
            tag = "dark" if dark else "all"
            print(f"{name} [{tag}] {args.start} -> {args.end}")
            rows = flatten(report(polygon(box), args.start, args.end, dark=dark))
            path = os.path.join(OUT, f"{name}_sar_{tag}.csv")
            if rows:
                keys = sorted({k for r in rows for k in r})
                with open(path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=keys)
                    w.writeheader()
                    w.writerows(rows)
            print(f"    {len(rows)} grid-cell rows -> {path if rows else '(none)'}")


if __name__ == "__main__":
    main()
