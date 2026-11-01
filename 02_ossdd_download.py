"""Download the OSSDD Sentinel-1 ship-detection dataset from Hugging Face.

The full repo is ~72 GB (zip ~14.4 GB + WebDataset shards). Pick a mode:

    python 02_ossdd_download.py --mode sample   # val+test shards + 5 train shards (start here)
    python 02_ossdd_download.py --mode shards --train-shards 20
    python 02_ossdd_download.py --mode zip      # full 14.4 GB zip (do this on the HPC)

Note: the dataset card's code says repo "OpenSARShip-Detection-Dataset" —
the real repo id is "OpenSARShip-Ship-Detection-Dataset" (used below).
License: CC-BY-NC-SA-4.0 (non-commercial).
"""
import argparse
import io
import os
import zipfile

from huggingface_hub import HfApi, hf_hub_download

REPO = "sylviaHoch/OpenSARShip-Ship-Detection-Dataset"
OUT = "data/ossdd"


def get(filename):
    return hf_hub_download(repo_id=REPO, filename=filename, repo_type="dataset", local_dir=OUT)


def sanity_check(shard_path):
    """Open one sample from a shard and print what's inside."""
    import tarfile
    import numpy as np
    import rasterio

    with tarfile.open(shard_path) as tar:
        members = {m.name: m for m in tar.getmembers()}
        vv = next(n for n in members if n.endswith("vv.tif"))
        key = vv[: -len("vv.tif")]
        data = tar.extractfile(members[vv]).read()
        with rasterio.open(io.BytesIO(data)) as src:
            arr = src.read(1).astype(np.float32)
        boxes = tar.extractfile(members[key + "aabb.txt"]).read().decode().splitlines()
    print(f"sample {key}: shape={arr.shape} min={arr.min():.1f} "
          f"median={np.median(arr):.1f} max={arr.max():.1f}  boxes={boxes[0].strip()}")
    print("Values are raw GRD amplitude (linear). Apply log10 + per-patch robust z-score,")
    print("and do the same to Earth Engine dB clips at inference so the two match.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sample", "shards", "zip"], default="sample")
    ap.add_argument("--train-shards", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    get("ship_dataset.py")
    get("README.md")

    if args.mode == "zip":
        path = get("ossdd.zip")
        print("Extracting...")
        with zipfile.ZipFile(path) as z:
            z.extractall(os.path.join(OUT, "ossdd"))
        print(f"Done: {OUT}/ossdd  (use ship_dataset.ShipDataset on it)")
        return

    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    shards = sorted(f for f in files if f.startswith("webdataset/") and f.endswith(".tar"))
    train = [f for f in shards if "/train/" in f]
    other = [f for f in shards if "/train/" not in f]
    n_train = args.train_shards if args.mode in ("sample", "shards") else len(train)
    wanted = other + train[:n_train]
    print(f"{len(train)} train shards available; downloading {len(wanted)} shards total")
    first = None
    for f in wanted:
        p = get(f)
        first = first or p
        print("  got", f)
    if first:
        sanity_check(first)


if __name__ == "__main__":
    main()
