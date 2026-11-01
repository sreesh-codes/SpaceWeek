"""Pull Sentinel-1 (radar) and Sentinel-2 (optical) clips for each AOI from
Google Earth Engine, and log every acquisition time to acquisitions.csv.

    python 01_gee_sentinel.py --list-only     # fast: just list passes + times
    python 01_gee_sentinel.py                 # list + download all clips
    python 01_gee_sentinel.py --aoi sohar --sensor s1

Sentinel-1 values from COPERNICUS/S1_GRD are sigma0 in dB (VV, VH bands).
"""
import argparse
import csv
import datetime as dt
import os

import ee
import requests

from config import AOIS, START, END, CRS, GEE_PROJECT

OUT = "data/sentinel"
GROUP_SECONDS = 120  # slices of the same pass are seconds apart -> mosaic them


def init():
    if not GEE_PROJECT:
        raise SystemExit("Set GEE_PROJECT env var to your Earth Engine Cloud project id")
    try:
        ee.Initialize(project=GEE_PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT)


def s1_collection(geom):
    return (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geom)
        .filterDate(START, END)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )


def s2_collection(geom, max_cloud=20):
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geom)
        .filterDate(START, END)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
    )


def group_passes(col, extra_props):
    """Return [(utc_datetime, [asset_ids], props)] grouped by acquisition pass."""
    ids = col.aggregate_array("system:id").getInfo()
    times = col.aggregate_array("system:time_start").getInfo()
    extras = {p: col.aggregate_array(p).getInfo() for p in extra_props}
    rows = sorted(zip(times, ids, *[extras[p] for p in extra_props]))
    groups = []
    for row in rows:
        t, asset = row[0], row[1]
        props = dict(zip(extra_props, row[2:]))
        if groups and (t - groups[-1][0]) / 1000 <= GROUP_SECONDS:
            groups[-1][1].append(asset)
        else:
            groups.append([t, [asset], props])
    return [(dt.datetime.utcfromtimestamp(t / 1000), a, p) for t, a, p in groups]


def download(image, geom, path):
    url = image.getDownloadURL(
        {"region": geom, "scale": 10, "crs": CRS, "format": "GEO_TIFF"}
    )
    r = requests.get(url, timeout=600)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--aoi", choices=list(AOIS), help="only this AOI")
    ap.add_argument("--sensor", choices=["s1", "s2", "both"], default="both")
    args = ap.parse_args()

    init()
    os.makedirs(OUT, exist_ok=True)
    log_path = os.path.join(OUT, "acquisitions.csv")
    new_log = not os.path.exists(log_path)
    log = open(log_path, "a", newline="")
    w = csv.writer(log)
    if new_log:
        w.writerow(["aoi", "sensor", "utc_time", "platform", "orbit_pass", "n_slices", "file"])

    for name, box in AOIS.items():
        if args.aoi and name != args.aoi:
            continue
        geom = ee.Geometry.Rectangle(list(box))

        jobs = []
        if args.sensor in ("s1", "both"):
            for t, assets, p in group_passes(
                s1_collection(geom), ["platform_number", "orbitProperties_pass"]
            ):
                jobs.append(("s1", t, assets, p.get("platform_number"), p.get("orbitProperties_pass")))
        if args.sensor in ("s2", "both"):
            for t, assets, p in group_passes(s2_collection(geom), ["SPACECRAFT_NAME"]):
                jobs.append(("s2", t, assets, p.get("SPACECRAFT_NAME"), ""))

        print(f"\n{name}: {len(jobs)} passes")
        for sensor, t, assets, platform, opass in sorted(jobs, key=lambda j: j[1]):
            stamp = t.strftime("%Y%m%dT%H%M%S")
            fname = f"{name}_{sensor}_{stamp}.tif"
            path = os.path.join(OUT, fname)
            print(f"  {sensor} {t:%Y-%m-%d %H:%M:%S}Z  {platform} {opass}  slices={len(assets)}")
            if not args.list_only and not os.path.exists(path):
                imgs = [ee.Image(a) for a in assets]
                bands = ["VV", "VH"] if sensor == "s1" else ["B4", "B3", "B2", "B8"]
                mosaic = ee.ImageCollection.fromImages(imgs).mosaic().select(bands).clip(geom)
                if sensor == "s2":
                    mosaic = mosaic.toUint16()
                try:
                    download(mosaic, geom, path)
                except Exception as e:  # too big or transient: fall back to Drive export
                    print(f"    direct download failed ({e}); queuing Drive export")
                    ee.batch.Export.image.toDrive(
                        image=mosaic, description=fname[:-4], folder="bill_of_orbit",
                        region=geom, scale=10, crs=CRS, maxPixels=1e9,
                    ).start()
            w.writerow([name, sensor, t.isoformat() + "Z", platform, opass, len(assets),
                        "" if args.list_only else fname])
    log.close()
    print(f"\nLogged to {log_path}")


if __name__ == "__main__":
    main()
