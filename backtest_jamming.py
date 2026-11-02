"""Out-of-time backtest of the jamming model.

Train on days before SPLIT, test on days from SPLIT on. Two settings:
  A. gap-fill  - point's own observation hidden, same-day neighbours and the
                 cell's past (up to the day before) available. This is what
                 predict() faces for a location between observed cells.
  B. forecast  - data frozen at SPLIT-1; no same-day neighbours. This is a
                 query for a date after the jamming file ends.
Compared against simple baselines and the previous model (lat/lon/day-of-year
features, random split). Errors are weighted by aircraft count (clipped at
200), since low-traffic jam scores are mostly noise; AUC uses only rows with
>= 10 aircraft. Writes reports/jamming_backtest.json.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

from jamming_model import (CHOKEPOINTS, FEATURES, FORECAST_FEATURES, JAM_THRESHOLD, _History, aircraft_weight,
                           fit_regressor, haversine_km, load_daily)

SPLIT = pd.Timestamp("2026-10-16")
REPO = os.path.dirname(os.path.abspath(__file__))


def old_features(df):
    """Feature set of the previous model (for comparison)."""
    doy, dow = df.date.dt.dayofyear.to_numpy(), df.date.dt.day_of_week.to_numpy()
    f = [df.lat, df.lon, doy, dow, np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
         np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)]
    f += [haversine_km(df.lat.to_numpy(), df.lon.to_numpy(), a, b) for a, b in CHOKEPOINTS.values()]
    return np.column_stack(f)


def metrics(y, p, w, n_aircraft):
    p = np.clip(p, 0, 1)
    ok = n_aircraft >= 10
    lab = y[ok] >= JAM_THRESHOLD
    return dict(mae=float(np.average(np.abs(y - p), weights=w)),
                rmse=float(np.sqrt(np.average((y - p) ** 2, weights=w))),
                auc_jammed=float(roc_auc_score(lab, p[ok])) if 0 < lab.sum() < len(lab) else None,
                n=int(len(y)))


def main():
    daily = load_daily()
    train = daily[(daily.date < SPLIT) & (daily.date > daily.date.min())]
    test = daily[daily.date >= SPLIT]
    print(f"train {train.date.min():%m-%d}..{train.date.max():%m-%d} ({len(train)} rows), "
          f"test {test.date.min():%m-%d}..{test.date.max():%m-%d} ({len(test)} rows)")
    y, w, n_ac = test.jam_score.to_numpy(), aircraft_weight(test.total_aircraft), test.total_aircraft.to_numpy()
    res = {}

    # baselines (training period only)
    gmean = np.average(train.jam_score, weights=aircraft_weight(train.total_aircraft))
    cmean = train.groupby("h3").jam_score.mean()
    clast = train.sort_values("date").groupby("h3").jam_score.last()
    res["baseline_global_mean"] = metrics(y, np.full(len(y), gmean), w, n_ac)
    res["baseline_cell_mean"] = metrics(y, test.h3.map(cmean).fillna(gmean).to_numpy(), w, n_ac)
    res["baseline_last_value"] = metrics(y, test.h3.map(clast).fillna(gmean).to_numpy(), w, n_ac)

    # previous model: lat/lon/day-of-year, trained on the training period
    old = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_depth=8, random_state=42)
    old.fit(old_features(train), train.jam_score)
    res["previous_model"] = metrics(y, old.predict(old_features(test)), w, n_ac)
    # how it looked under its own random split (in-sample-like optimism)
    from sklearn.model_selection import train_test_split
    a, b = train_test_split(daily, test_size=0.2, random_state=42)
    old2 = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_depth=8, random_state=42)
    old2.fit(old_features(a), a.jam_score)
    res["previous_model_random_split_(optimistic)"] = metrics(
        b.jam_score.to_numpy(), old2.predict(old_features(b)), aircraft_weight(b.total_aircraft), b.total_aircraft.to_numpy())

    # new model
    hist = _History(daily)
    print("building features...")
    Xtr = hist.frame(train, history_end=SPLIT - pd.Timedelta(days=1))
    model = fit_regressor(Xtr, train.jam_score.to_numpy(), aircraft_weight(train.total_aircraft))
    XA = hist.frame(test)                                            # gap-fill
    XB = hist.frame(test, history_end=SPLIT - pd.Timedelta(days=1))  # forecast
    res["new_model_gapfill"] = metrics(y, model.predict(XA), w, n_ac)
    fc = fit_regressor(hist.forecast_frame(train, history_end=SPLIT - pd.Timedelta(days=1)),
                       train.jam_score.to_numpy(), aircraft_weight(train.total_aircraft))
    res["new_model_forecast"] = metrics(y, fc.predict(XB[FORECAST_FEATURES]), w, n_ac)
    # baseline for gap-fill: plain neighbour mean
    nb = XA.nbr_mean.fillna(pd.Series(XA.hist_mean)).fillna(gmean).to_numpy()
    res["baseline_neighbour_mean_gapfill"] = metrics(y, nb, w, n_ac)

    # robustness: score per test week
    weekly = {}
    for wk, idx in test.groupby(test.date.dt.isocalendar().week).groups.items():
        i = test.index.get_indexer(idx)
        weekly[int(wk)] = dict(gapfill_mae=float(np.average(np.abs(y[i] - np.clip(model.predict(XA.iloc[i]), 0, 1)), weights=w[i])),
                               cell_mean_mae=float(np.average(np.abs(y[i] - test.h3.map(cmean).fillna(gmean).to_numpy()[i]), weights=w[i])))
    res["weekly"] = weekly

    os.makedirs(os.path.join(REPO, "reports"), exist_ok=True)
    json.dump(dict(split=str(SPLIT.date()), results=res), open(os.path.join(REPO, "reports", "jamming_backtest.json"), "w"), indent=1)
    print(f"\n{'method':42s}   MAE    RMSE   AUC(jam>={JAM_THRESHOLD})")
    for k, v in res.items():
        if k != "weekly":
            auc = f"{v['auc_jammed']:.3f}" if v["auc_jammed"] is not None else "  -  "
            print(f"{k:42s} {v['mae']:.4f} {v['rmse']:.4f}  {auc}")
    print("weekly:", json.dumps(weekly))


if __name__ == "__main__":
    main()
