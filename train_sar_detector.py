"""Train the SAR ship detector on the OSSDD YOLO subset (02b_ossdd_to_yolo.py).

Uses a rotated-box (OBB) model: ship length is read from the box's long side,
and axis-aligned boxes overstate the length of any ship not lying N-S or E-W.

Settings aim at generalising to unseen Sentinel-1 scenes, not at the val set:
  * SAR-safe augmentation: flips + 90-degree-free rotation, mild scale, no hue/
    saturation jitter (the channels are VV / VH / mean, not colours).
  * imgsz 704 so 700-px chips are not downscaled (ships are often 10-20 px).
  * early stopping on val; the held-out OSSDD *test* split (different scenes)
    is only touched by backtest_sar_detector.py after training.
"""
import argparse
import os
import shutil

from ultralytics import YOLO

REPO = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(REPO, "data", "ossdd_yolo", "data.yaml"))
    ap.add_argument("--model", default="yolo11n-obb.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=704)
    ap.add_argument("--device", default=None, help="e.g. 0, mps, cpu (default: auto)")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=2, patience=10, seed=0, deterministic=True,
        project=os.path.join(REPO, "runs"), name="sar_ship_obb", exist_ok=True,
        # augmentation suited to SAR chips
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.15, fliplr=0.5, flipud=0.5, degrees=180,
        scale=0.3, translate=0.1, mosaic=1.0, close_mosaic=5, mixup=0.0,
        weight_decay=0.0005, cos_lr=True, plots=True,
    )
    run = os.path.join(REPO, "runs", "sar_ship_obb")
    dst = os.path.join(REPO, "models", "sar_ship_obb")
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(run, dst, ignore=shutil.ignore_patterns("last.pt", "*batch*.jpg"))
    print(f"Best weights: {dst}/weights/best.pt  -> run backtest_sar_detector.py next")


if __name__ == "__main__":
    main()
