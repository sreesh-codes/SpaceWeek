"""Jamming-Trust Map and Anomaly Detection Model for Gulf Waters.

Part of Bill of Orbit (Theme 3 - Seas of Data).
Trained on GNSS interference data in the Persian Gulf, Strait of Hormuz,
and Gulf of Oman (Fujairah / Khor Fakkan / Sohar).

Maps GPS jamming events, detects spatial-temporal anomalies, and provides
an AIS-trust multiplier so that fraudulent or spoofed AIS tracks inside
jammed corridors are discounted and Sentinel-1 SAR radar takes precedence.
"""
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "jamming", "gulf_jamming_cells.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
JAMMING_MODEL_FILE = os.path.join(MODEL_DIR, "jamming_model.pkl")

# Strategic coordinates in the Gulf for proximity features
CHOKEPOINTS = {
    "hormuz_center": (26.56, 56.25),
    "fujairah_anchorage": (25.13, 56.44),
    "khor_fakkan": (25.36, 56.41),
    "sohar_port": (24.49, 56.66),
    "dubai_offshore": (25.30, 55.10),
}


def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in kilometers."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2.0) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2)
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class JammingTrustModel:
    """Spatio-temporal Jamming and AIS Trust model."""

    def __init__(self):
        self.regressor = None
        self.anomaly_detector = None
        self.kdtree = None
        self.grid_coords = None
        self.daily_records = None
        self.is_trained = False

    def _extract_features(self, df):
        """Build spatial, temporal, and proximity features."""
        dates = pd.to_datetime(df["date"])
        day_of_year = dates.dt.dayofyear.values
        day_of_week = dates.dt.dayofweek.values

        lat = df["lat"].values
        lon = df["lon"].values

        feats = [
            lat,
            lon,
            day_of_year,
            day_of_week,
            np.sin(2 * np.pi * day_of_year / 365.25),
            np.cos(2 * np.pi * day_of_year / 365.25),
            np.sin(2 * np.pi * day_of_week / 7.0),
            np.cos(2 * np.pi * day_of_week / 7.0),
        ]

        # Add proximity to key anchorages and chokepoints
        for name, (c_lat, c_lon) in CHOKEPOINTS.items():
            dist = haversine_km(lat, lon, c_lat, c_lon)
            feats.append(dist)

        return np.column_stack(feats)

    def fit(self, data_path=DATA_PATH, sample_size=100000):
        """Fit spatial-temporal regressor and anomaly detector."""
        print(f"Loading jamming cells from {data_path}...")
        df = pd.read_csv(data_path)
        
        # Focus on daily aggregated metrics for robust spatial trends
        df_daily = df[df["hour"] == "daily"].copy()
        if len(df_daily) == 0:
            df_daily = df.copy()

        print(f"Dataset has {len(df_daily):,} records.")
        self.daily_records = df_daily[["date", "lat", "lon", "jam_score", "low_quality", "total_aircraft"]].copy()

        # Build KD-Tree for nearest-neighbor lookups on unique grid centroids
        unique_cells = df_daily.drop_duplicates(subset=["lat", "lon"])[["lat", "lon"]].values
        self.grid_coords = unique_cells
        self.kdtree = cKDTree(unique_cells)

        # Subsample for faster model fitting if needed
        if len(df_daily) > sample_size:
            df_train = df_daily.sample(n=sample_size, random_state=42)
        else:
            df_train = df_daily

        X = self._extract_features(df_train)
        y = df_train["jam_score"].values

        print("Fitting HistGradientBoostingRegressor for spatial-temporal jamming prediction...")
        self.regressor = HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.08,
            max_depth=8,
            random_state=42
        )
        self.regressor.fit(X, y)

        print("Fitting IsolationForest for interference anomaly detection...")
        self.anomaly_detector = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42,
            n_jobs=-1
        )
        self.anomaly_detector.fit(X)

        self.is_trained = True
        print("Jamming-Trust model successfully trained.")

    def save(self, filepath=JAMMING_MODEL_FILE):
        """Save model bundle to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        bundle = {
            "regressor": self.regressor,
            "anomaly_detector": self.anomaly_detector,
            "grid_coords": self.grid_coords,
            "daily_records": self.daily_records,
            "is_trained": self.is_trained,
        }
        with open(filepath, "wb") as f:
            pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath=JAMMING_MODEL_FILE):
        """Load model bundle from disk."""
        inst = cls()
        with open(filepath, "rb") as f:
            bundle = pickle.load(f)
        inst.regressor = bundle["regressor"]
        inst.anomaly_detector = bundle["anomaly_detector"]
        inst.grid_coords = bundle["grid_coords"]
        inst.daily_records = bundle["daily_records"]
        inst.kdtree = cKDTree(inst.grid_coords)
        inst.is_trained = bundle["is_trained"]
        return inst

    def predict(self, lat, lon, date_str):
        """
        Predict jamming intensity and compute AIS trust factor for (lat, lon, date).

        Returns:
            dict containing:
                - jam_score: float in [0.0, 1.0]
                - is_jammed: bool
                - ais_trust_factor: float in [0.0, 1.0] (1.0 = trusted, 0.0 = completely spoofed/jammed)
                - is_anomaly: bool
                - risk_category: str ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
                - explanation: str
        """
        # 1. First, check if exact historical observation exists within 15 km on that date
        dist, idx = self.kdtree.query([lat, lon], k=1)
        dist_km = haversine_km(lat, lon, self.grid_coords[idx][0], self.grid_coords[idx][1])
        
        hist_score = None
        if dist_km <= 25.0 and self.daily_records is not None:
            c_lat, c_lon = self.grid_coords[idx]
            match = self.daily_records[
                (self.daily_records["date"] == date_str) &
                (np.isclose(self.daily_records["lat"], c_lat, atol=0.05)) &
                (np.isclose(self.daily_records["lon"], c_lon, atol=0.05))
            ]
            if len(match) > 0:
                hist_score = float(match["jam_score"].iloc[0])

        # 2. Compute ML predicted score
        df_query = pd.DataFrame([{"lat": lat, "lon": lon, "date": date_str}])
        X_q = self._extract_features(df_query)
        pred_score = float(np.clip(self.regressor.predict(X_q)[0], 0.0, 1.0))
        is_anomaly = bool(self.anomaly_detector.predict(X_q)[0] == -1)

        # Combine historical ground-truth if available, else use ML spatial prediction
        final_score = hist_score if hist_score is not None else pred_score
        final_score = float(np.clip(final_score, 0.0, 1.0))

        # AIS Trust Factor inversely decays with jamming score
        # When jam_score >= 0.5, AIS confidence falls sharply
        ais_trust = float(np.clip(1.0 / (1.0 + np.exp(8.0 * (final_score - 0.35))), 0.05, 1.0))

        if final_score < 0.10:
            risk = "LOW"
            note = f"Clear GNSS environment (jam score {final_score:.3f}). AIS transmissions reliable."
        elif final_score < 0.30:
            risk = "MEDIUM"
            note = f"Moderate GNSS interference (jam score {final_score:.3f}). AIS reliability slightly impaired."
        elif final_score < 0.60:
            risk = "HIGH"
            note = f"Severe GNSS jamming detected (jam score {final_score:.3f}). AIS positions unreliable; SAR satellite verification mandatory."
        else:
            risk = "CRITICAL"
            note = f"Intense electronic warfare / GPS denial zone (jam score {final_score:.3f}). AIS signals likely spoofed or absent."

        return {
            "lat": lat,
            "lon": lon,
            "date": date_str,
            "jam_score": round(final_score, 4),
            "is_jammed": final_score >= 0.25,
            "ais_trust_factor": round(ais_trust, 4),
            "is_anomaly": is_anomaly,
            "risk_category": risk,
            "explanation": note,
        }


if __name__ == "__main__":
    model = JammingTrustModel()
    model.fit()
    model.save()

    # Test queries
    test_cases = [
        {"name": "Fujairah Anchorage (Sept 28, 2026)", "lat": 25.15, "lon": 56.45, "date": "2026-09-28"},
        {"name": "Strait of Hormuz (Oct 15, 2026)", "lat": 26.50, "lon": 56.30, "date": "2026-10-15"},
        {"name": "Sohar Port (Sept 10, 2026)", "lat": 24.50, "lon": 56.65, "date": "2026-09-10"},
    ]
    print("\n--- Test Inferences ---")
    for tc in test_cases:
        res = model.predict(tc["lat"], tc["lon"], tc["date"])
        print(f"\n{tc['name']}:")
        for k, v in res.items():
            print(f"  {k}: {v}")
