"""Stream OSSDD from Hugging Face straight into a small YOLO dataset, without
ever storing the raw 32-bit chips (the full repo is ~72 GB).

    python 02b_ossdd_to_yolo.py                              # 4000 train + 800 val (~1 GB)
    python 02b_ossdd_to_yolo.py --max-train 2000 --max-val 400   # ~0.5 GB, start here if tight
    python 02b_ossdd_to_yolo.py --labels aabb                 # axis-aligned boxes instead

Then:
    yolo obb train data=data/ossdd_yolo/data.yaml model=yolo11n-obb.pt

* Each shard is read sequentially over HTTP and only its first `cap` samples
  are used, then the connection is dropped, so only those bytes are fetched.
  Shards are visited in a seeded random order, so the subset spans many scenes.
* Every chip goes through sar_preprocess.ossdd_to_rgb -> 8-bit JPEG. Run Earth
  Engine clips through sar_preprocess.gee_tif_tiles at detection time.
* Empty chips (no ships) are kept only up to --empty-frac of the output.
* Re-running resumes: finished shards are recorded in progress.json.
License of the source data: CC-BY-NC-SA-4.0 (non-commercial).
"""
import argparse
import io
import json
import math
import os
import random
import tarfile
import warnings

import numpy as np
import rasterio
import requests
from huggingface_hub import HfApi, hf_hub_url
from PIL import Image

from sar_preprocess import ossdd_to_rgb

REPO = "sylviaHoch/OpenSARShip-Ship-Detection-Dataset"
OUT = "data/ossdd_yolo"
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


def read_tif(data):
    with rasterio.open(io.BytesIO(data)) as src:
        return src.read(1).astype(np.float32)


def parse_boxes(text):
    """OSSDD box file -> list of number lists (index column dropped).
    First line is the box count; values are whitespace separated."""
    lines = text.decode("utf-8").strip().splitlines()
    return [[float(v) for v in ln.split()[1:]] for ln in lines[1:] if ln.split()]


def yolo_lines(sample, w, h, fmt):
    lines = []
    if fmt == "obb":
        for b in parse_boxes(sample["rbb.txt"]):
            pts = np.array(b[:8]).reshape(4, 2)
            if np.ptp(pts[:, 0]) < 1 or np.ptp(pts[:, 1]) < 1:
                continue  # degenerate box
            pts = np.clip(pts / [w, h], 0, 1)
            lines.append("0 " + " ".join(f"{v:.6f}" for v in pts.ravel()))
    else:
        for x0, y0, x1, y1, *_ in parse_boxes(sample["aabb.txt"]):
            x0, x1 = max(0, x0), min(w, x1 + 1)  # max coords are inclusive pixels
            y0, y1 = max(0, y0), min(h, y1 + 1)
            if x1 - x0 < 1 or y1 - y0 < 1:
                continue
            lines.append(f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} "
                         f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
    return lines


def samples(url, token):
    """Yield dicts of {field: bytes, '__key__': key} streamed from one tar shard."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with requests.get(url, stream=True, timeout=120, headers=headers) as r:
        r.raise_for_status()
        tar = tarfile.open(fileobj=r.raw, mode="r|")
        cur = {}
        for m in tar:
            if not m.isfile():
                continue
            key, field = m.name.split(".", 1)
            if cur and cur["__key__"] != key:
                yield cur
                cur = {}
            cur["__key__"] = key
            cur[field] = tar.extractfile(m).read()
        if cur:
            yield cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train", type=int, default=4000)
    ap.add_argument("--max-val", type=int, default=800)
    ap.add_argument("--labels", choices=["obb", "aabb"], default="obb")
    ap.add_argument("--empty-frac", type=float, default=0.15)
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN")

    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    prog_path = os.path.join(OUT, "progress.json")
    os.makedirs(OUT, exist_ok=True)
    prog = json.load(open(prog_path)) if os.path.exists(prog_path) else {}
    if prog and prog.get("labels") != args.labels:
        raise SystemExit(f"{OUT} was built with --labels {prog.get('labels')}; delete it or match it")
    prog["labels"] = args.labels

    for split, target in (("train", args.max_train), ("val", args.max_val)):
        img_dir, lbl_dir = (os.path.join(OUT, d, split) for d in ("images", "labels"))
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        st = prog.setdefault(split, {"done": [], "kept": 0, "empty": 0})
        shards = sorted(f for f in files if f.startswith(f"webdataset/{split}/") and f.endswith(".tar"))
        random.Random(args.seed).shuffle(shards)
        # spread the subset over ~2/3 of the shards; empties are filtered, hence the 1.3x
        cap = max(20, math.ceil(1.3 * target / max(1, round(len(shards) * 2 / 3))))
        print(f"\n{split}: {len(shards)} shards, target {target}, up to {cap} chips read per shard")

        for shard in shards:
            if st["kept"] >= target:
                break
            if shard in st["done"]:
                continue
            got = 0
            try:
                for i, s in enumerate(samples(hf_hub_url(REPO, shard, repo_type="dataset"), token)):
                    if i >= cap or st["kept"] >= target:
                        break
                    if not all(k in s for k in ("vv.tif", "vh.tif", "rbb.txt", "aabb.txt")):
                        continue
                    vv, vh = read_tif(s["vv.tif"]), read_tif(s["vh.tif"])
                    h, w = vv.shape
                    lines = yolo_lines(s, w, h, args.labels)
                    if not lines:
                        if st["empty"] + 1 > args.empty_frac * (st["kept"] + 1):
                            continue
                        st["empty"] += 1
                    Image.fromarray(ossdd_to_rgb(vv, vh)).save(
                        os.path.join(img_dir, s["__key__"] + ".jpg"), quality=args.quality)
                    with open(os.path.join(lbl_dir, s["__key__"] + ".txt"), "w") as f:
                        f.write("\n".join(lines) + ("\n" if lines else ""))
                    st["kept"] += 1
                    got += 1
            except Exception as e:
                print(f"  skipped shard {shard}: {e!r}")
                continue
            st["done"].append(shard)
            json.dump(prog, open(prog_path, "w"), indent=1)
            print(f"  {shard}: +{got}  -> {st['kept']}/{target} ({st['empty']} empty)")

    with open(os.path.join(OUT, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(OUT)}\ntrain: images/train\nval: images/val\n"
                f"names:\n  0: ship\n")
    print(f"\nWrote {OUT}/data.yaml  (labels: {args.labels})")


if __name__ == "__main__":
    main()
