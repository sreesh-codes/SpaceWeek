"""Ship-to-Ship (STS) Transfer and Dark Vessel Detection Module.

Part of Bill of Orbit (Theme 3 - Seas of Data).
Flags possible undisclosed offshore cargo transfers near the claimed vessel.

Evidence, strongest first:
  1. Our own Sentinel-1 detections (when the pipeline passes them in): two
     hulls side by side - centres closer than half their summed lengths plus
     100 m, with headings within 25 degrees.
  2. Global Fishing Watch SAR presence: GFW reports ships on a 0.01 deg grid
     (~1.1 km), so it cannot resolve two ships 100-250 m apart; the usable
     signal is a single cell holding >= 2 detections on one date.
  3. Dark detections (no AIS match) inside the search radius.
(The earlier version looked for two GFW cells < 250 m apart, which the grid
makes impossible, so it never fired.)
"""
import glob
import os
from datetime import timedelta

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "gfw")


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2.0) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2)
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _cells(files):
    dfs = [pd.read_csv(f) for f in files]
    if not dfs:
        return pd.DataFrame(columns=["date", "lat", "lon", "detections"])
    df = pd.concat(dfs, ignore_index=True)
    return df.groupby(["date", "lat", "lon"], as_index=False)["detections"].sum()


def side_by_side_pairs(dets, max_heading_diff=25.0, gap_m=100.0):
    """Pairs of SAR detections (dicts with lat, lon, length_m, heading_deg)
    lying alongside each other."""
    pairs = []
    for i in range(len(dets)):
        for j in range(i + 1, len(dets)):
            a, b = dets[i], dets[j]
            d_m = haversine_km(a["lat"], a["lon"], b["lat"], b["lon"]) * 1000
            if d_m > (a["length_m"] + b["length_m"]) / 2 + gap_m:
                continue
            if a.get("heading_deg") is not None and b.get("heading_deg") is not None:
                dh = abs(a["heading_deg"] - b["heading_deg"]) % 180
                if min(dh, 180 - dh) > max_heading_diff:
                    continue
            pairs.append((i, j, d_m))
    return pairs


class STSDetector:
    def __init__(self, data_dir=DATA_DIR):
        self.cells_all = _cells(glob.glob(os.path.join(data_dir, "*_sar_all.csv")))
        self.cells_dark = _cells(glob.glob(os.path.join(data_dir, "*_sar_dark.csv")))
        print(f"STSDetector: {int(self.cells_all.detections.sum())} GFW SAR detections, "
              f"{int(self.cells_dark.detections.sum())} dark.")

    def _near(self, cells, lat, lon, start, end, radius_km):
        sub = cells[(cells.date >= start) & (cells.date <= end)].copy()
        if sub.empty:
            return sub.assign(dist_m=[])
        sub["dist_m"] = haversine_km(lat, lon, sub.lat.to_numpy(), sub.lon.to_numpy()) * 1000
        return sub[sub.dist_m <= radius_km * 1000]

    def detect_sts_transfer(self, target_lat, target_lon, target_date, radius_km=2.0,
                            window_days=1, sar_detections=None):
        """sar_detections: optional list of our detections from the matched
        Sentinel-1 pass (sar_inference.detect_scene output)."""
        t = pd.to_datetime(target_date)
        start = (t - timedelta(days=window_days)).strftime("%Y-%m-%d")
        end = (t + timedelta(days=window_days)).strftime("%Y-%m-%d")

        near_all = self._near(self.cells_all, target_lat, target_lon, start, end, radius_km)
        near_dark = self._near(self.cells_dark, target_lat, target_lon, start, end, radius_km)
        crowded = near_all[near_all.detections >= 2]
        dark_count = int(near_dark.detections.sum()) if len(near_dark) else 0

        sar_pairs = []
        if sar_detections:
            near = [d for d in sar_detections
                    if haversine_km(target_lat, target_lon, d["lat"], d["lon"]) <= radius_km]
            sar_pairs = [(near[i], near[j], d) for i, j, d in side_by_side_pairs(near)]

        sts_detected = bool(sar_pairs) or len(crowded) > 0
        dark_nearby = dark_count > 0
        if sar_pairs and dark_nearby:
            risk = 0.95
            notes = (f"CRITICAL: two hulls side by side ({sar_pairs[0][2]:.0f} m apart) on Sentinel-1 "
                     f"with {dark_count} dark (non-AIS) detection(s) within {radius_km} km.")
        elif sts_detected and dark_nearby:
            risk = 0.85
            notes = (f"HIGH RISK: vessels clustered within one ~1 km radar cell and {dark_count} dark "
                     f"detection(s) within {radius_km} km.")
        elif sar_pairs:
            risk = 0.75
            notes = f"HIGH RISK: two hulls side by side ({sar_pairs[0][2]:.0f} m apart) on Sentinel-1."
        elif sts_detected:
            risk = 0.6
            notes = f"ELEVATED: {len(crowded)} radar cell(s) within {radius_km} km hold 2+ vessels on one date."
        elif dark_nearby:
            risk = 0.45
            notes = f"SUSPICIOUS: {dark_count} dark vessel detection(s) within {radius_km} km around {target_date}."
        else:
            risk = 0.05
            notes = "NORMAL: no vessel clustering or dark detections near the claimed position."
        if near_all.empty and near_dark.empty and not sar_detections:
            notes += " (No GFW coverage for this place/date: absence of evidence only.)"

        return {
            "sts_detected": sts_detected,
            "dark_vessel_nearby": dark_nearby,
            "nearby_vessels_count": int(near_all.detections.sum()) if len(near_all) else 0,
            "dark_vessels_count": dark_count,
            "crowded_cells": int(len(crowded)),
            "sar_side_by_side_pairs": len(sar_pairs),
            "min_distance_m": round(float(near_all.dist_m.min()), 1) if len(near_all) else None,
            "sts_risk_score": round(risk, 3),
            "notes": notes,
        }


if __name__ == "__main__":
    detector = STSDetector()
    for name, lat, lon, d in [("Fujairah Anchorage", 25.13, 56.48, "2026-10-16"),
                              ("Khor Fakkan", 25.35, 56.40, "2026-09-15")]:
        print(f"\n--- {name} ({d}) ---")
        for k, v in detector.detect_sts_transfer(lat, lon, d).items():
            print(f"  {k}: {v}")
