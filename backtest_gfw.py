"""Real-world backtest: run the SAR detector on our Sentinel-1 clips and compare
with Global Fishing Watch's independent detections from the same passes.

GFW (public-global-sar-presence) runs its own detector on the same Sentinel-1
scenes and reports ships on a 0.01 deg grid (~1.1 km). It is not perfect ground
truth: it drops detections close to shore and fixed infrastructure, and very
small boats. So read the numbers as agreement, not accuracy:
  * gfw_recall     share of GFW ship detections with one of ours within MATCH_KM
  * agreement      share of our detections with a GFW detection within MATCH_KM
  * count correlation across passes
Only passes whose date has GFW rows for that AOI are scored (GFW data ends ~5
days before download, and a date with no rows may simply be unprocessed).

Scene threshold: the detector is less confident on full Earth Engine scenes
than on OSSDD chips (different processing chain), so the chip-tuned threshold
misses many real ships. The scene threshold is chosen on passes before
TUNE_END (max F1 of gfw_recall and agreement) and judged on later passes only.
It is written to reports/gfw_backtest.json and used by pipeline.py.

    python backtest_gfw.py [--weights models/sar_ship_obb/weights/best.pt]
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from sar_inference import detect_scene, load_model

REPO = os.path.dirname(os.path.abspath(__file__))
SENTINEL = os.path.join(REPO, "data", "sentinel")
GFW = os.path.join(REPO, "data", "gfw")
MATCH_KM = 1.2  # one GFW grid cell plus margin
TUNE_END = "2026-10-01"
THRESHOLDS = [0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]


def km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def score_pass(dets, day):
    dl = np.array([[d["lat"], d["lon"]] for d in dets]) if dets else np.zeros((0, 2))
    gl, w = day[["lat", "lon"]].to_numpy(), day.detections.to_numpy()
    if len(dl) and len(gl):
        D = km(dl[:, None, 0], dl[:, None, 1], gl[None, :, 0], gl[None, :, 1])
        gfw_hit, our_hit = D.min(0) <= MATCH_KM, D.min(1) <= MATCH_KM
    else:
        gfw_hit, our_hit = np.zeros(len(gl), bool), np.zeros(len(dl), bool)
    return dict(gfw_ships=int(w.sum()), ours=len(dl), gfw_found=int((gfw_hit * w).sum()),
                ours_agree=int(our_hit.sum()))


def summarise(df):
    r = df.gfw_found.sum() / max(1, df.gfw_ships.sum())
    a = df.ours_agree.sum() / max(1, df.ours.sum())
    return dict(passes=len(df), gfw_ships=int(df.gfw_ships.sum()), our_detections=int(df.ours.sum()),
                gfw_recall=float(r), agreement=float(a), f1=float(2 * r * a / (r + a)) if r + a else 0.0,
                count_corr=float(np.corrcoef(df.gfw_ships, df.ours)[0, 1]) if len(df) > 2 else None)


def table(passes, thr):
    rows = []
    for p in passes:
        dets = [d for d in p["dets"] if d["conf"] >= thr]
        rows.append(dict(aoi=p["aoi"], date=p["date"], file=p["file"], **score_pass(dets, p["day"])))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--sentinel-dir", default=SENTINEL, help="folder with the S1 .tif clips")
    args = ap.parse_args()

    model = load_model(args.weights)
    try:
        bt = json.load(open(os.path.join(REPO, "reports", "sar_backtest.json")))
        chip_conf = bt[os.path.basename(os.path.dirname(os.path.dirname(model.ckpt_path)))]["conf_threshold_from_val"]
    except Exception:
        chip_conf = 0.25
    print(f"model {model.ckpt_path}  chip threshold {chip_conf}")

    acq = pd.read_csv(os.path.join(args.sentinel_dir, "acquisitions.csv"))
    acq = acq[(acq.sensor == "s1") & acq.file.notna()]
    passes = []
    for aoi, g in acq.groupby("aoi"):
        path = os.path.join(GFW, f"{aoi}_sar_all.csv")
        if not os.path.exists(path):
            continue
        cells = pd.read_csv(path).groupby(["date", "lat", "lon"], as_index=False)["detections"].sum()
        for _, a in g.iterrows():
            date = a.utc_time[:10]
            tif = os.path.join(args.sentinel_dir, a.file)
            day = cells[cells.date == date]
            if day.empty or not os.path.exists(tif):
                continue
            if (g.utc_time.str[:10] == date).sum() > 1:
                continue  # two passes on one date: can't tell which GFW rows belong to which
            passes.append(dict(aoi=aoi, date=date, file=a.file, day=day,
                               dets=detect_scene(model, tif, conf=min(THRESHOLDS), device=args.device)))
            print(f"  {aoi:20s} {date}  {len(passes[-1]['dets'])} candidates")

    tune = [p for p in passes if p["date"] < TUNE_END]
    held = [p for p in passes if p["date"] >= TUNE_END]
    sweep = {t: dict(tune=summarise(table(tune, t)), held_out=summarise(table(held, t))) for t in THRESHOLDS}
    scene_conf = max(THRESHOLDS, key=lambda t: sweep[t]["tune"]["f1"])
    df = table(passes, scene_conf)
    summ = {"held_out": sweep[scene_conf]["held_out"], "tune": sweep[scene_conf]["tune"]}
    for aoi, sub in df[df.date >= TUNE_END].groupby("aoi"):
        summ[f"held_out_{aoi}"] = summarise(sub)
    lengths = [d["length_m"] for p in held for d in p["dets"] if d["conf"] >= scene_conf]
    summ["our_length_m_quantiles"] = dict(zip(["p10", "p50", "p90", "max"],
                                              map(float, np.percentile(lengths, [10, 50, 90, 100])))) if lengths else {}

    os.makedirs(os.path.join(REPO, "reports"), exist_ok=True)
    df.to_csv(os.path.join(REPO, "reports", "gfw_backtest_passes.csv"), index=False)
    json.dump(dict(weights=model.ckpt_path, match_km=MATCH_KM, tune_dates=f"< {TUNE_END}",
                   chip_conf=chip_conf, scene_conf=scene_conf,
                   held_out_at_chip_conf=summarise(table(held, chip_conf)) if held else None,
                   summary=summ, sweep={str(k): v for k, v in sweep.items()}),
              open(os.path.join(REPO, "reports", "gfw_backtest.json"), "w"), indent=1)
    print(f"\nscene threshold chosen on passes before {TUNE_END}: {scene_conf}")
    print(f"{'thr':>6} {'tune rec':>9} {'tune agr':>9} {'held rec':>9} {'held agr':>9}")
    for t, v in sweep.items():
        print(f"{t:6.3f} {v['tune']['gfw_recall']:9.3f} {v['tune']['agreement']:9.3f} "
              f"{v['held_out']['gfw_recall']:9.3f} {v['held_out']['agreement']:9.3f}")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
