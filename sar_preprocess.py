"""One SAR -> 8-bit RGB conversion shared by OSSDD training chips and Earth
Engine Sentinel-1 clips, so the detector sees the same kind of image in both.

The two sources are on different scales:
  * OSSDD chips are raw GRD amplitude DN (linear, uncalibrated, can be <= 0).
  * Earth Engine COPERNICUS/S1_GRD is calibrated sigma0 power in dB.
20*log10(DN) and sigma0_dB differ (to first order) by a constant offset per
scene, so both are brought to dB and then robust-z-scored per chip, which
cancels the offset. Chips are normalised independently, so Earth Engine scenes
must be normalised in chip-sized blocks too (see gee_tif_to_rgb / gee_tif_tiles).

Channels: R = VV, G = VH, B = mean(VV, VH), each mapped from z in [Z_LO, Z_HI].
"""
import numpy as np

Z_LO, Z_HI = -3.0, 10.0   # sea clutter sits near 0, ships are strong positive outliers
CHIP = 700                # OSSDD training chip size; normalise GEE scenes in blocks of this


def dn_to_db(dn):
    """OSSDD amplitude DN -> dB. Clips non-positive values and extreme outliers
    (> 20000), the same way the dataset's ShipDataset class does."""
    dn = np.asarray(dn, dtype=np.float32)
    return 20.0 * np.log10(np.clip(dn, 1.0, 20000.0))


def robust_z(db):
    """Per-chip robust z-score (median / MAD). NaNs (no-data) stay NaN."""
    finite = db[np.isfinite(db)]
    if finite.size == 0:
        return np.zeros_like(db)
    med = np.median(finite)
    mad = np.median(np.abs(finite - med)) * 1.4826
    return (db - med) / max(mad, 1e-3)


def _to_u8(z):
    z = np.nan_to_num(z, nan=Z_LO)
    return np.clip((z - Z_LO) / (Z_HI - Z_LO) * 255.0, 0, 255).astype(np.uint8)


def db_to_rgb(vv_db, vh_db):
    """Two dB arrays (H, W) for one chip -> uint8 RGB (H, W, 3)."""
    zvv, zvh = robust_z(vv_db), robust_z(vh_db)
    return np.dstack([_to_u8(zvv), _to_u8(zvh), _to_u8((zvv + zvh) / 2)])


def ossdd_to_rgb(vv_dn, vh_dn):
    """OSSDD sample (raw DN arrays) -> uint8 RGB."""
    return db_to_rgb(dn_to_db(vv_dn), dn_to_db(vh_dn))


def _read_gee(path):
    import rasterio
    with rasterio.open(path) as src:
        vv, vh = src.read(1).astype(np.float32), src.read(2).astype(np.float32)
        if src.nodata is not None:
            vv[vv == src.nodata] = np.nan
            vh[vh == src.nodata] = np.nan
    return vv, vh


def gee_tif_to_rgb(path, block=CHIP):
    """Earth Engine S1 clip (bands VV, VH in dB) -> uint8 RGB of the full scene,
    normalised block by block so each block matches a training chip."""
    vv, vh = _read_gee(path)
    out = np.zeros(vv.shape + (3,), np.uint8)
    for y in range(0, vv.shape[0], block):
        for x in range(0, vv.shape[1], block):
            s = np.s_[y:y + block, x:x + block]
            out[s] = db_to_rgb(vv[s], vh[s])
    return out


def gee_tif_tiles(path, size=CHIP, overlap=100):
    """Yield (x0, y0, rgb) overlapping tiles of an Earth Engine S1 clip, each
    normalised on its own exactly like a training chip. Use this for detection,
    then shift boxes by (x0, y0) and de-duplicate across overlaps (NMS)."""
    vv, vh = _read_gee(path)
    h, w = vv.shape
    step = size - overlap
    ys = list(range(0, max(h - size, 0) + 1, step)) + ([h - size] if h > size and (h - size) % step else [])
    xs = list(range(0, max(w - size, 0) + 1, step)) + ([w - size] if w > size and (w - size) % step else [])
    for y in ys:
        for x in xs:
            s = np.s_[y:y + size, x:x + size]
            yield x, y, db_to_rgb(vv[s], vh[s])
