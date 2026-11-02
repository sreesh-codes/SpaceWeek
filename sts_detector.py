"""Ship-to-Ship (STS) Transfer and Dark Vessel Detection Module.

Part of Bill of Orbit (Theme 3 - Seas of Data).
Identifies illicit or undisclosed offshore cargo transfers (such as Russian or Iranian
crude STS transfers near Fujairah and Sohar) by analyzing vessel proximity,
dual-hull radar signatures, and Global Fishing Watch dark SAR detections.
"""
import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "gfw")


def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in kilometers."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2.0) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2)
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class STSDetector:
    """Detects Ship-to-Ship transfers and dark vessel proximity."""

    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.df_all = None
        self.df_dark = None
        self._load_gfw_data()

    def _load_gfw_data(self):
        """Load GFW all-SAR and dark-SAR detection datasets."""
        all_files = glob.glob(os.path.join(self.data_dir, "*_sar_all.csv"))
        dark_files = glob.glob(os.path.join(self.data_dir, "*_sar_dark.csv"))

        dfs_all = [pd.read_csv(f) for f in all_files if os.path.exists(f)]
        dfs_dark = [pd.read_csv(f) for f in dark_files if os.path.exists(f)]

        self.df_all = pd.concat(dfs_all, ignore_index=True) if dfs_all else pd.DataFrame()
        self.df_dark = pd.concat(dfs_dark, ignore_index=True) if dfs_dark else pd.DataFrame()

        print(f"STSDetector loaded {len(self.df_all)} GFW radar detections and {len(self.df_dark)} dark vessel detections.")

    def detect_sts_transfer(self, target_lat, target_lon, target_date, radius_km=1.5, window_days=1):
        """
        Check if the target vessel was involved in or adjacent to a ship-to-ship transfer
        or dark vessel interaction.

        Returns:
            dict containing:
                - sts_detected: bool
                - dark_vessel_nearby: bool
                - nearby_vessels_count: int
                - dark_vessels_count: int
                - min_distance_m: float
                - involved_vessels: list
                - risk_score: float (0.0 to 1.0)
                - notes: str
        """
        if self.df_all.empty:
            return {"sts_detected": False, "notes": "No GFW data available"}

        target_dt = pd.to_datetime(target_date)
        date_start = (target_dt - timedelta(days=window_days)).strftime("%Y-%m-%d")
        date_end = (target_dt + timedelta(days=window_days)).strftime("%Y-%m-%d")

        # 1. Check all vessels in time window
        sub_all = self.df_all[
            (self.df_all["date"] >= date_start) & (self.df_all["date"] <= date_end)
        ].copy()

        # 2. Check dark vessels in time window
        sub_dark = self.df_dark[
            (self.df_dark["date"] >= date_start) & (self.df_dark["date"] <= date_end)
        ].copy()

        sts_detected = False
        dark_nearby = False
        min_dist_m = 99999.0
        nearby_ships = []
        dark_count = 0

        if not sub_all.empty:
            dists = haversine_km(target_lat, target_lon, sub_all["lat"].values, sub_all["lon"].values) * 1000.0
            sub_all["dist_m"] = dists
            close = sub_all[(sub_all["dist_m"] <= radius_km * 1000.0) & (sub_all["dist_m"] > 10.0)]
            if len(close) > 0:
                min_dist_m = float(close["dist_m"].min())
                for _, r in close.iterrows():
                    ship = r.get("shipName") if pd.notnull(r.get("shipName")) else f"MMSI:{r.get('mmsi', 'Unknown')}"
                    nearby_ships.append({
                        "ship": str(ship),
                        "distance_m": round(float(r["dist_m"]), 1),
                        "date": str(r["date"]),
                        "type": str(r.get("vesselType", "UNKNOWN")),
                    })
                # If two vessels are moored side-by-side (< 250m apart) in open anchorage
                if min_dist_m < 250.0:
                    sts_detected = True

        if not sub_dark.empty:
            dists_dark = haversine_km(target_lat, target_lon, sub_dark["lat"].values, sub_dark["lon"].values) * 1000.0
            sub_dark["dist_m"] = dists_dark
            close_dark = sub_dark[sub_dark["dist_m"] <= radius_km * 1000.0]
            dark_count = len(close_dark)
            if dark_count > 0:
                dark_nearby = True
                min_dark_dist = float(close_dark["dist_m"].min())
                min_dist_m = min(min_dist_m, min_dark_dist)
                if min_dark_dist < 300.0:
                    sts_detected = True

        # Calculate STS risk score
        risk = 0.0
        if sts_detected and dark_nearby:
            risk = 0.95
            notes = f"CRITICAL: Probable illicit Ship-to-Ship (STS) transfer detected! Vessel was moored within {min_dist_m:.0f}m of an untracked (dark) vessel."
        elif sts_detected:
            risk = 0.75
            notes = f"HIGH RISK: Ship-to-Ship (STS) transfer observed within {min_dist_m:.0f}m of another vessel in offshore anchorage."
        elif dark_nearby:
            risk = 0.45
            notes = f"SUSPICIOUS: {dark_count} dark vessel(s) detected by radar within {radius_km} km on {target_date}."
        else:
            risk = 0.05
            notes = "NORMAL: No abnormal vessel clustering or STS transfers detected."

        return {
            "sts_detected": sts_detected,
            "dark_vessel_nearby": dark_nearby,
            "nearby_vessels_count": len(nearby_ships),
            "dark_vessels_count": dark_count,
            "min_distance_m": round(min_dist_m, 1) if min_dist_m < 90000 else None,
            "nearby_vessels": nearby_ships[:5],
            "sts_risk_score": round(risk, 3),
            "notes": notes,
        }


if __name__ == "__main__":
    detector = STSDetector()
    print("\n--- Test Query 1: Fujairah Anchorage (2026-10-16) ---")
    res1 = detector.detect_sts_transfer(25.13, 56.48, "2026-10-16")
    for k, v in res1.items():
        print(f"  {k}: {v}")

    print("\n--- Test Query 2: Khor Fakkan (2026-09-15) ---")
    res2 = detector.detect_sts_transfer(25.35, 56.40, "2026-09-15")
    for k, v in res2.items():
        print(f"  {k}: {v}")
