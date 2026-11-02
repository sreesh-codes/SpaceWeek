"""Jamming-Trust Map for Gulf waters.

Part of Bill of Orbit (Theme 3 - Seas of Data).
Built on daily GNSS-interference cells (Stanford ADS-B integrity monitor, H3
cells, 05_jamming.py). Gives a jam score for any (lat, lon, date) and an
AIS-trust multiplier, so AIS gaps inside jammed corridors are not read as
evidence of fraud and Sentinel-1 radar takes precedence.

How a query is answered:
  1. If the date has an observation for the H3 cell containing the point, use
     it, shrunk toward the model when few aircraft were seen (a jam score from
     2 aircraft is mostly noise).
  2. Otherwise predict with a gradient-boosted model (a second, forecast
     model without neighbour features handles dates nobody observed) whose features describe
     the cell's own *past* (mean, last value, 7-day mean - never the future)
     plus the same-day mean of neighbouring cells when that day was observed.
     Plain lat/lon/day-of-year features were dropped: day-of-year can't
     extrapolate past the training window and lat/lon alone overfit cells.
  3. Anomaly = today's score well above the cell's own history (robust z > 3).

Backtest (train on earlier dates, test on later ones): backtest_jamming.py.
"""
import os
import pickle

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "jamming", "gulf_jamming_cells.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
JAMMING_MODEL_FILE = os.path.join(MODEL_DIR, "jamming_model.pkl")

CHOKEPOINTS = {
    "hormuz_center": (26.56, 56.25),
    "fujairah_anchorage": (25.13, 56.44),
    "khor_fakkan": (25.36, 56.41),
    "sohar_port": (24.49, 56.66),
    "dubai_offshore": (25.30, 55.10),
}
CELL_RADIUS_KM = 25.0      # H3 res-4 cells: ~22 km edge
NEIGHBOUR_KM = 80.0
MIN_AIRCRAFT = 10          # below this an observed score is shrunk toward the model
JAM_THRESHOLD = 0.25

FEATURES = (["lat", "lon"] + [f"d_{k}" for k in CHOKEPOINTS] +
            ["hist_mean", "hist_n", "last", "days_since_last", "mean_7d", "nbr_mean", "nbr_n", "dow"])
# forecast model: for dates with no same-day observations (e.g. after the file ends)
FORECAST_FEATURES = [f for f in FEATURES if not f.startswith("nbr_")]
MAX_HORIZON = 14


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2.0) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2)
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load_daily(path=DATA_PATH):
    df = pd.read_csv(path)
    d = df[df["hour"] == "daily"].copy()
    if d.empty:  # hourly-only file: aggregate
        d = (df.groupby(["date", "h3", "lat", "lon"], as_index=False)
               [["low_quality", "total_aircraft"]].sum())
        d["jam_score"] = (d.low_quality - 1).clip(lower=0) / d.total_aircraft
    d["date"] = pd.to_datetime(d["date"])
    return d[["date", "h3", "lat", "lon", "jam_score", "low_quality", "total_aircraft"]].reset_index(drop=True)


class _History:
    """Per-cell daily tables used to build leakage-free features."""

    def __init__(self, daily):
        self.daily = daily
        self.cells = daily.drop_duplicates("h3")[["h3", "lat", "lon"]].set_index("h3")
        self.tree = cKDTree(self.cells[["lat", "lon"]].to_numpy())
        self.piv = daily.pivot_table(index="date", columns="h3", values="jam_score")
        self.w = daily.pivot_table(index="date", columns="h3", values="total_aircraft")

    def nearest_cell(self, lat, lon):
        _, i = self.tree.query([lat, lon])
        h3 = self.cells.index[i]
        c = self.cells.iloc[i]
        return h3, float(haversine_km(lat, lon, c.lat, c.lon))

    def features(self, lat, lon, date, h3=None, history_end=None):
        """Feature row for a point/date. Cell history uses only days < date
        (and <= history_end, to simulate a dataset frozen at that day).
        Same-day neighbours never include the point's own cell."""
        date = pd.Timestamp(date)
        cut = date if history_end is None else min(date, pd.Timestamp(history_end) + pd.Timedelta(days=1))
        f = {"lat": lat, "lon": lon, "dow": date.day_of_week}
        for k, (a, b) in CHOKEPOINTS.items():
            f[f"d_{k}"] = haversine_km(lat, lon, a, b)
        f.update(hist_mean=np.nan, hist_n=0, last=np.nan, days_since_last=np.nan, mean_7d=np.nan)
        if h3 is not None and h3 in self.piv.columns:
            s = self.piv[h3][self.piv.index < cut].dropna()
            if len(s):
                f.update(hist_mean=s.mean(), hist_n=len(s), last=s.iloc[-1],
                         days_since_last=(date - s.index[-1]).days,
                         mean_7d=s[s.index >= date - pd.Timedelta(days=7)].mean())
        f["nbr_mean"], f["nbr_n"] = np.nan, 0
        if date in self.piv.index and (history_end is None or date <= pd.Timestamp(history_end)):
            today = self.piv.loc[date].dropna()
            today = today.drop(h3, errors="ignore")
            if len(today):
                cc = self.cells.loc[today.index]
                d = haversine_km(lat, lon, cc.lat.to_numpy(), cc.lon.to_numpy())
                m = d <= NEIGHBOUR_KM
                if m.any():
                    wts = 1.0 / (d[m] + 5.0)
                    f["nbr_mean"] = float(np.sum(today.to_numpy()[m] * wts) / wts.sum())
                    f["nbr_n"] = int(m.sum())
        return f

    def frame(self, rows, **kw):
        return pd.DataFrame([self.features(r.lat, r.lon, r.date, r.h3, **kw)
                             for r in rows.itertuples()])[FEATURES]

    def forecast_frame(self, rows, seed=0, history_end=None):
        """Training rows for the forecast model: each row sees history only up
        to a random 1..MAX_HORIZON days earlier, like a query past the data."""
        rng = np.random.default_rng(seed)
        out = []
        for r in rows.itertuples():
            end = r.date - pd.Timedelta(days=int(rng.integers(1, MAX_HORIZON + 1)))
            if history_end is not None:
                end = min(end, pd.Timestamp(history_end))
            out.append(self.features(r.lat, r.lon, r.date, r.h3, history_end=end))
        return pd.DataFrame(out)[FORECAST_FEATURES]


def fit_regressor(X, y, w):
    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_depth=4, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=20, random_state=42)
    model.fit(X, y, sample_weight=w)
    return model


def aircraft_weight(total_aircraft):
    """Noisy low-traffic cells count less when fitting and scoring."""
    return np.clip(np.asarray(total_aircraft, float), 1, 200)


class JammingTrustModel:
    def __init__(self):
        self.regressor = None
        self.forecaster = None
        self.hist = None
        self.is_trained = False

    def fit(self, data_path=DATA_PATH):
        daily = load_daily(data_path)
        print(f"{len(daily):,} daily cell records, {daily.h3.nunique()} cells, "
              f"{daily.date.min():%Y-%m-%d} -> {daily.date.max():%Y-%m-%d}")
        self.hist = _History(daily)
        train = daily[daily.date > daily.date.min()]  # first day has no history
        y, w = train.jam_score.to_numpy(), aircraft_weight(train.total_aircraft)
        self.regressor = fit_regressor(self.hist.frame(train), y, w)
        self.forecaster = fit_regressor(self.hist.forecast_frame(train), y, w)
        self.is_trained = True

    def save(self, filepath=JAMMING_MODEL_FILE):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({"regressor": self.regressor, "forecaster": self.forecaster,
                         "daily": self.hist.daily}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath=JAMMING_MODEL_FILE):
        inst = cls()
        # Only load model files this repo produced; pickle can run code from untrusted files.
        with open(filepath, "rb") as f:
            b = pickle.load(f)
        if "daily" not in b:
            raise ValueError(f"{filepath} is in the old format; retrain with `python jamming_model.py`")
        inst.regressor, inst.forecaster = b["regressor"], b["forecaster"]
        inst.hist, inst.is_trained = _History(b["daily"]), True
        return inst

    def predict(self, lat, lon, date_str):
        date = pd.Timestamp(date_str)
        h3, dist_km = self.hist.nearest_cell(lat, lon)
        cell = h3 if dist_km <= CELL_RADIUS_KM else None
        f = pd.DataFrame([self.hist.features(lat, lon, date, cell)])
        if f.nbr_n.iloc[0] > 0:
            pred = self.regressor.predict(f[FEATURES])[0]
        else:
            pred = self.forecaster.predict(f[FORECAST_FEATURES])[0]
        pred = float(np.clip(pred, 0, 1))

        obs = n_ac = None
        if cell is not None and date in self.hist.piv.index:
            v = self.hist.piv.at[date, cell]
            if pd.notna(v):
                obs, n_ac = float(v), int(self.hist.w.at[date, cell])
        if obs is not None:
            k = min(1.0, n_ac / MIN_AIRCRAFT)
            score, source = k * obs + (1 - k) * pred, f"observed ({n_ac} aircraft)"
        else:
            score, source = pred, "model estimate (no observation for this cell/date)"
        score = float(np.clip(score, 0, 1))

        is_anomaly = False
        if cell is not None:
            s = self.hist.piv[cell][self.hist.piv.index < date].dropna()
            if len(s) >= 7:
                mad = 1.4826 * np.median(np.abs(s - s.median())) + 0.02
                is_anomaly = bool((score - s.median()) / mad > 3)

        ais_trust = float(np.clip(1.0 / (1.0 + np.exp(8.0 * (score - 0.35))), 0.05, 1.0))
        if score < 0.10:
            risk, note = "LOW", f"Clear GNSS environment (jam score {score:.3f}). AIS transmissions reliable."
        elif score < 0.30:
            risk, note = "MEDIUM", f"Moderate GNSS interference (jam score {score:.3f}). AIS reliability slightly impaired."
        elif score < 0.60:
            risk, note = "HIGH", f"Severe GNSS jamming (jam score {score:.3f}). AIS positions unreliable; SAR verification mandatory."
        else:
            risk, note = "CRITICAL", f"Intense GPS denial zone (jam score {score:.3f}). AIS signals likely spoofed or absent."
        return {
            "lat": lat, "lon": lon, "date": date.strftime("%Y-%m-%d"),
            "jam_score": round(score, 4),
            "is_jammed": score >= JAM_THRESHOLD,
            "ais_trust_factor": round(ais_trust, 4),
            "is_anomaly": is_anomaly,
            "risk_category": risk,
            "source": source,
            "explanation": note,
        }


if __name__ == "__main__":
    model = JammingTrustModel()
    model.fit()
    model.save()
    for name, lat, lon, d in [("Fujairah Anchorage", 25.15, 56.45, "2026-09-28"),
                              ("Strait of Hormuz", 26.50, 56.30, "2026-10-15"),
                              ("Sohar Port", 24.50, 56.65, "2026-09-10"),
                              ("Fujairah, after data ends", 25.15, 56.45, "2026-11-05")]:
        print(f"\n{name} {d}:")
        for k, v in model.predict(lat, lon, d).items():
            print(f"  {k}: {v}")
