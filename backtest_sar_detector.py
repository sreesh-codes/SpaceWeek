"""Backtest SAR ship detectors on the OSSDD splits with one shared metric, so
axis-aligned (detect) and rotated (OBB) models can be compared fairly.

    python backtest_sar_detector.py                       # all available models
    python backtest_sar_detector.py --weights a.pt b.pt

For each model:
  1. Ultralytics' own mAP on the held-out test split (native box type).
  2. The confidence threshold that maximises F1 is chosen on VAL only, then
     precision / recall / F1 are reported on TEST at that threshold (no tuning
     on test). Matching: greedy by confidence, IoU >= 0.3 between the
     axis-aligned envelopes of prediction and ground truth.
  3. Ship-length error on matched ships (metres, 10 m/px), overall and by size.
Results go to reports/sar_backtest.json and are printed as a table.
"""
import argparse
import glob
import json
import math
import os

import numpy as np
from PIL import Image

from sar_inference import DEFAULT_WEIGHTS, PIXEL_M, _result_to_dets, load_model

REPO = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(REPO, "data", "ossdd_yolo")
SIZE_BINS = [(0, 50), (50, 100), (100, 200), (200, 1000)]


def load_gt(label_path, w, h):
    ships = []
    for ln in open(label_path).read().split("\n"):
        v = ln.split()
        if not v:
            continue
        v = list(map(float, v[1:]))
        if len(v) == 8:  # OBB corners
            p = np.array(v).reshape(4, 2) * [w, h]
            e1, e2 = np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[1])
            ships.append(dict(box=[*p.min(0), *p.max(0)], length_px=max(e1, e2)))
        else:  # cx cy bw bh
            cx, cy, bw, bh = v[0] * w, v[1] * h, v[2] * w, v[3] * h
            ships.append(dict(box=[cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2],
                              length_px=max(bw, bh)))
    return ships


def pred_box(d):
    """Axis-aligned envelope of a (possibly rotated) prediction."""
    L, W = d["length_px"], d["width_px"]
    if d["heading_deg"] is None:  # detect boxes are already envelopes
        return [d["cx"] - d["_bw"] / 2, d["cy"] - d["_bh"] / 2, d["cx"] + d["_bw"] / 2, d["cy"] + d["_bh"] / 2]
    a = math.radians(d["heading_deg"])
    ex = abs(L / 2 * math.cos(a)) + abs(W / 2 * math.sin(a))
    ey = abs(L / 2 * math.sin(a)) + abs(W / 2 * math.cos(a))
    return [d["cx"] - ex, d["cy"] - ey, d["cx"] + ex, d["cy"] + ey]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def run_split(model, split, device, imgsz):
    imgs = sorted(glob.glob(os.path.join(DATA, "images", split, "*.jpg")))
    out = []
    for i in range(0, len(imgs), 8):
        batch = imgs[i:i + 8]
        rgbs = [np.asarray(Image.open(p).convert("RGB")) for p in batch]
        res = model.predict([x[..., ::-1] for x in rgbs], conf=0.05, imgsz=imgsz, device=device, verbose=False)
        for p, x, r in zip(batch, rgbs, res):
            dets = _result_to_dets(r)
            if getattr(r, "boxes", None) is not None and len(r.boxes):  # keep envelopes for detect models
                for d, (x0, y0, x1, y1) in zip(dets, r.boxes.xyxy.cpu().numpy()):
                    d["_bw"], d["_bh"] = x1 - x0, y1 - y0
            h, w = x.shape[:2]
            gt = load_gt(p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt", w, h)
            out.append((dets, gt))
    return out


def score(pairs, thr):
    tp = fp = fn = 0
    lens = []
    for dets, gt in pairs:
        dets = sorted([d for d in dets if d["conf"] >= thr], key=lambda d: -d["conf"])
        used = set()
        for d in dets:
            pb = pred_box(d)
            best, bi = 0.3, None
            for j, g in enumerate(gt):
                if j not in used:
                    v = iou(pb, g["box"])
                    if v >= best:
                        best, bi = v, j
            if bi is None:
                fp += 1
            else:
                used.add(bi)
                tp += 1
                lens.append((gt[bi]["length_px"] * PIXEL_M, d["length_px"] * PIXEL_M))
        fn += len(gt) - len(used)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return dict(precision=p, recall=r, f1=f1, tp=tp, fp=fp, fn=fn), lens


def length_stats(lens):
    if not lens:
        return {}
    g, pr = np.array(lens).T
    rel = np.abs(pr - g) / g
    stats = {"all": dict(n=len(g), mae_m=float(np.mean(np.abs(pr - g))), bias_m=float(np.mean(pr - g)),
                         median_rel_err=float(np.median(rel)), within_20pct=float(np.mean(rel <= 0.2)))}
    for lo, hi in SIZE_BINS:
        m = (g >= lo) & (g < hi)
        if m.sum():
            stats[f"{lo}-{hi}m"] = dict(n=int(m.sum()), mae_m=float(np.mean(np.abs(pr[m] - g[m]))),
                                        median_rel_err=float(np.median(rel[m])),
                                        within_20pct=float(np.mean(rel[m] <= 0.2)))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="*", default=[w for w in DEFAULT_WEIGHTS if os.path.exists(w)])
    ap.add_argument("--device", default=None)
    ap.add_argument("--imgsz", type=int, default=704)
    args = ap.parse_args()
    report = {}
    for w in args.weights:
        name = os.path.basename(os.path.dirname(os.path.dirname(w)))
        print(f"\n=== {name} ({w})")
        model = load_model(w)
        native = model.val(data=os.path.join(DATA, "data.yaml"), split="test", imgsz=args.imgsz,
                           batch=8, device=args.device, plots=False, verbose=False,
                           project=os.path.join(REPO, "runs"), name=f"bt_{name}", exist_ok=True)
        val_pairs = run_split(model, "val", args.device, args.imgsz)
        thrs = np.round(np.arange(0.05, 0.9, 0.025), 3)
        f1s = [score(val_pairs, t)[0]["f1"] for t in thrs]
        thr = float(thrs[int(np.argmax(f1s))])
        test_pairs = run_split(model, "test", args.device, args.imgsz)
        val_m, _ = score(val_pairs, thr)
        test_m, lens = score(test_pairs, thr)
        report[name] = dict(task=model.task, weights=w, conf_threshold_from_val=thr,
                            native_test_mAP50=float(native.box.map50), native_test_mAP50_95=float(native.box.map),
                            val=val_m, test=test_m, length_test=length_stats(lens))
        print(json.dumps(report[name], indent=1))
    os.makedirs(os.path.join(REPO, "reports"), exist_ok=True)
    with open(os.path.join(REPO, "reports", "sar_backtest.json"), "w") as f:
        json.dump(report, f, indent=1)
    print("\nmodel              task    thr   test P   test R  test F1  len MAE  len ±20%")
    for n, r in report.items():
        L = r["length_test"].get("all", {})
        print(f"{n:18s} {r['task']:6s} {r['conf_threshold_from_val']:.3f}  {r['test']['precision']:.3f}    "
              f"{r['test']['recall']:.3f}    {r['test']['f1']:.3f}   {L.get('mae_m', float('nan')):6.1f}m   "
              f"{L.get('within_20pct', float('nan')):.2f}")


if __name__ == "__main__":
    main()
