"""Run the SAR ship detector on chips and on full Earth Engine Sentinel-1 clips.

Works with both the rotated-box (OBB) model and the older axis-aligned (detect)
model. Every detection comes back as a dict with pixel centre, length/width,
heading and confidence; detect_scene() also adds lat/lon and metres.

Length: OBB long side. For axis-aligned boxes the best available proxy is
max(w, h), which underestimates ships lying diagonally.
"""
import math
import os

import numpy as np

from sar_preprocess import CHIP, gee_tif_tiles

PIXEL_M = 10.0  # OSSDD GRD chips and the GEE clips (scale=10) are both 10 m/px
REPO = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = [
    os.path.join(REPO, "models", "sar_ship_obb", "weights", "best.pt"),
    os.path.join(REPO, "models", "sar_ship_yolo", "weights", "best.pt"),
]


def load_model(weights=None):
    from ultralytics import YOLO
    for w in ([weights] if weights else DEFAULT_WEIGHTS):
        if w and os.path.exists(w):
            return YOLO(w)
    raise FileNotFoundError(f"No detector weights found in {DEFAULT_WEIGHTS}")


def _result_to_dets(r):
    dets = []
    if getattr(r, "obb", None) is not None and len(r.obb):
        for (cx, cy, w, h, rot), c in zip(r.obb.xywhr.cpu().numpy(), r.obb.conf.cpu().numpy()):
            length, width = max(w, h), min(w, h)
            # heading of the long axis, degrees in [0, 180)
            ang = math.degrees(rot if w >= h else rot + math.pi / 2) % 180
            dets.append(dict(cx=float(cx), cy=float(cy), length_px=float(length),
                             width_px=float(width), heading_deg=float(ang), conf=float(c)))
    elif getattr(r, "boxes", None) is not None and len(r.boxes):
        for (x0, y0, x1, y1), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            w, h = x1 - x0, y1 - y0
            dets.append(dict(cx=float((x0 + x1) / 2), cy=float((y0 + y1) / 2),
                             length_px=float(max(w, h)), width_px=float(min(w, h)),
                             heading_deg=None, conf=float(c)))
    return dets


def detect_chips(model, rgbs, conf=0.25, imgsz=704, device=None):
    """Detect on a list of uint8 RGB chips -> list of detection lists."""
    # ultralytics expects BGR numpy input
    res = model.predict([x[..., ::-1] for x in rgbs], conf=conf, imgsz=imgsz,
                        device=device, verbose=False)
    return [_result_to_dets(r) for r in res]


def _dedupe(dets):
    """Drop duplicates from overlapping tiles: keep the most confident detection
    among those whose centres are closer than half the smaller ship's length."""
    keep = []
    for d in sorted(dets, key=lambda d: -d["conf"]):
        if all(math.hypot(d["cx"] - k["cx"], d["cy"] - k["cy"])
               > max(3.0, 0.5 * min(d["length_px"], k["length_px"])) for k in keep):
            keep.append(d)
    return keep


def detect_scene(model, tif_path, conf=0.25, imgsz=704, device=None, batch=4):
    """Detect ships in an Earth Engine S1 clip. Returns detections with
    lat/lon, length_m and width_m added (pixel coords are scene-level)."""
    import rasterio
    from rasterio.warp import transform as warp

    tiles = list(gee_tif_tiles(tif_path, size=CHIP, overlap=100))
    dets = []
    for i in range(0, len(tiles), batch):
        chunk = tiles[i:i + batch]
        for (x0, y0, _), ds in zip(chunk, detect_chips(model, [t[2] for t in chunk], conf, imgsz, device)):
            for d in ds:
                d["cx"] += x0
                d["cy"] += y0
                dets.append(d)
    dets = _dedupe(dets)
    if not dets:
        return []
    with rasterio.open(tif_path) as src:
        xs, ys = rasterio.transform.xy(src.transform, [d["cy"] for d in dets], [d["cx"] for d in dets])
        lons, lats = warp(src.crs, "EPSG:4326", list(xs), list(ys))
        # GEE exports at 10 m; read the real pixel size in case that changes
        px = abs(src.transform.a)
    for d, lat, lon in zip(dets, lats, lons):
        d.update(lat=float(lat), lon=float(lon),
                 length_m=round(d["length_px"] * px, 1), width_m=round(d["width_px"] * px, 1))
    return dets
