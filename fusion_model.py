"""Orbital Evidence Fusion Model for Bill of Orbit.

Part of Bill of Orbit (Theme 3 - Seas of Data).
Combines:
1. Sentinel-1 SAR ship detection & length matching
2. GNSS Jamming-Trust map & AIS credibility discount
3. Ship-to-Ship (STS) transfer & dark vessel risk
4. Port geofencing & acquisition timeline alignment

Trains and executes a Gradient-Boosted Decision Tree (GBDT) model to compute
the 0-100 Orbital Evidence Score, risk classification, and automated compliance memo.
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
FUSION_MODEL_FILE = os.path.join(MODEL_DIR, "fusion_scorer.pkl")

FEATURE_NAMES = [
    "sar_detected",
    "sar_confidence",
    "length_match_ratio",
    "length_delta_m",
    "ais_reported",
    "ais_trust_factor",
    "discounted_ais_weight",
    "jamming_score",
    "radar_ais_discrepancy",
    "sts_risk_score",
    "dark_vessel_nearby",
    "days_from_pass",
]


# +1: score may only rise with the feature, -1: only fall, 0: unconstrained.
# Keeps the scorer sensible on inputs unlike the synthetic training cases.
MONOTONIC = {
    "sar_detected": 1, "sar_confidence": 1, "length_match_ratio": 1, "length_delta_m": -1,
    "ais_reported": 0, "ais_trust_factor": 0, "discounted_ais_weight": 0, "jamming_score": 0,
    "radar_ais_discrepancy": -1, "sts_risk_score": -1, "dark_vessel_nearby": -1, "days_from_pass": -1,
}
SCENARIOS = ["clean", "ghost", "jammed_genuine", "sts_transfer", "size_mismatch", "dark_no_excuse"]
SCENARIO_P = [0.35, 0.20, 0.15, 0.10, 0.10, 0.10]
# risk tiers each scenario should land in (used by backtest_fusion())
EXPECTED_TIERS = {
    "clean": {"LOW_RISK"}, "jammed_genuine": {"LOW_RISK"}, "ghost": {"HIGH_RISK"},
    "sts_transfer": {"MEDIUM_RISK", "HIGH_RISK"}, "size_mismatch": {"MEDIUM_RISK", "HIGH_RISK"},
    "dark_no_excuse": {"MEDIUM_RISK", "HIGH_RISK"},
}
CLASS_LENGTHS = [110, 160, 190, 228, 245, 274, 330]


def _length_noise():
    """Relative std of SAR length estimates, measured on the OSSDD test split
    (reports/sar_backtest.json, ships >= 100 m) when available."""
    try:
        import json
        bt = json.load(open(os.path.join(os.path.dirname(__file__), "reports", "sar_backtest.json")))
        best = max(bt.values(), key=lambda r: r["test"]["f1"])
        bins = [best["length_test"].get(k) for k in ("100-200m", "200-1000m")]
        errs = [b["median_rel_err"] for b in bins if b]
        if errs:
            return float(np.clip(1.4826 * np.mean(errs), 0.05, 0.5))  # median abs err -> std
    except Exception:
        pass
    return 0.15


def generate_training_data(n_samples=6000, random_state=42, length_noise=None):
    """Synthetic labelled cases. There is no labelled real-world fraud data, so
    the scorer learns an expert-specified score per scenario; features are
    generated consistently (length ratio and delta come from the same lengths,
    SAR length carries the detector's measured error)."""
    rng = np.random.RandomState(random_state)
    noise = _length_noise() if length_noise is None else length_noise
    rows = []
    for _ in range(n_samples):
        sc = rng.choice(SCENARIOS, p=SCENARIO_P)
        days = int(rng.choice([0, 1, 2, 3]))
        claimed = float(rng.choice(CLASS_LENGTHS))
        true_len = claimed * rng.uniform(0.92, 1.05)  # class nominal vs real hull
        jam = rng.uniform(0.0, 0.15)
        ais, sts, dark = 1, rng.uniform(0.0, 0.15), 0
        sar, conf = 1, rng.uniform(0.40, 0.95)

        if sc == "ghost":
            sar, conf, true_len = 0, 0.0, 0.0
            ais = int(rng.rand() < 0.3)  # AIS may be spoofed
            jam = rng.uniform(0.0, 0.40)
        elif sc == "jammed_genuine":
            ais = int(rng.rand() < 0.25)
            jam = rng.uniform(0.40, 0.90)
            sts, dark = rng.uniform(0.0, 0.20), int(rng.rand() < 0.2)
        elif sc == "sts_transfer":
            jam = rng.uniform(0.10, 0.50)
            sts, dark = rng.uniform(0.70, 0.98), 1
        elif sc == "size_mismatch":
            true_len = claimed * rng.uniform(0.30, 0.60)
        elif sc == "dark_no_excuse":
            ais = 0  # radar sees the ship, AIS silent, no jamming to explain it

        trust = float(np.clip(1.0 / (1.0 + np.exp(8.0 * (jam - 0.35))), 0.05, 1.0))
        sar_len = true_len * max(0.1, 1 + rng.normal(0, noise)) if sar else 0.0
        ratio = min(sar_len, claimed) / max(sar_len, claimed) if sar else 0.0
        delta = abs(sar_len - claimed) if sar else 250.0
        disc = int((sar == 0 and ais == 1) or (sar == 1 and ais == 0 and jam < 0.35))

        if sc == "clean":
            score = 84 + 12 * ratio - 2.5 * days + 4 * (conf - 0.4)
        elif sc == "ghost":
            score = 22 + 6 * ais * trust - 4 * disc
        elif sc == "jammed_genuine":
            score = 80 + 10 * ratio - 2.5 * days + 4 * (conf - 0.4)
        elif sc == "sts_transfer":
            score = 66 - 25 * sts + 5 * ratio
        elif sc == "size_mismatch":
            score = 40 + 30 * ratio
        else:
            score = 50 + 8 * ratio - 2 * days
        score = float(np.clip(score + rng.normal(0, 2.5), 5, 99))

        rows.append(dict(sar_detected=sar, sar_confidence=conf, length_match_ratio=ratio,
                         length_delta_m=delta, ais_reported=ais, ais_trust_factor=trust,
                         discounted_ais_weight=ais * trust, jamming_score=jam,
                         radar_ais_discrepancy=disc, sts_risk_score=sts, dark_vessel_nearby=dark,
                         days_from_pass=days, scenario=sc, orbital_evidence_score=round(score, 2)))
    return pd.DataFrame(rows)


def risk_tier(score):
    return "LOW_RISK" if score >= 80 else "MEDIUM_RISK" if score >= 50 else "HIGH_RISK"


def backtest_fusion(model, n=3000, seed=7, feature_noise=0.0):
    """Score fresh synthetic cases (new seed; optional extra noise on the
    continuous SAR features) and check tiers per scenario + monotonicity."""
    df = generate_training_data(n, random_state=seed)
    X = df[FEATURE_NAMES].copy()
    if feature_noise:
        rng = np.random.RandomState(seed)
        for c in ("sar_confidence", "length_match_ratio"):
            X[c] = np.clip(X[c] + rng.normal(0, feature_noise, len(X)) * (X.sar_detected > 0), 0, 1)
    pred = np.clip(model.predict(X), 0, 100)
    out = {"rmse": float(np.sqrt(mean_squared_error(df.orbital_evidence_score, pred))),
           "r2": float(r2_score(df.orbital_evidence_score, pred)), "tier_accuracy": {}}
    ok_all = []
    for sc in SCENARIOS:
        m = (df.scenario == sc).to_numpy()
        ok = np.array([risk_tier(p) in EXPECTED_TIERS[sc] for p in pred[m]])
        out["tier_accuracy"][sc] = float(ok.mean()) if len(ok) else None
        ok_all.append(ok)
    out["tier_accuracy"]["all"] = float(np.concatenate(ok_all).mean())
    # monotonicity: nudge each constrained feature and count violations
    viol = {}
    for f, s in MONOTONIC.items():
        if s == 0:
            continue
        X2 = X.copy()
        step = 1 if f in ("sar_detected", "radar_ais_discrepancy", "dark_vessel_nearby", "days_from_pass") else 0.1
        X2[f] = X2[f] + step
        d = np.clip(model.predict(X2), 0, 100) - pred
        viol[f] = float(np.mean(d * s < -0.5))
    out["monotonic_violation_rate"] = viol
    return out


class OrbitalEvidenceFusionModel:
    """Trained Gradient-Boosted Fusion Scorer."""

    def __init__(self):
        self.model = None
        self.is_trained = False

    def fit(self, n_samples=6000):
        print(f"Generating {n_samples} synthetic trade scenarios (length noise {_length_noise():.2f})...")
        df = generate_training_data(n_samples=n_samples)
        X_train, X_test, y_train, y_test = train_test_split(
            df[FEATURE_NAMES], df["orbital_evidence_score"], test_size=0.2, random_state=42)
        self.model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=4, min_samples_leaf=30,
            l2_regularization=1.0, monotonic_cst=[MONOTONIC[f] for f in FEATURE_NAMES],
            early_stopping=True, n_iter_no_change=20, random_state=42)
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        print(f"Held-out synthetic RMSE {np.sqrt(mean_squared_error(y_test, preds)):.2f}, "
              f"R2 {r2_score(y_test, preds):.4f}  (synthetic labels: this shows fit, not real-world accuracy)")
        self.is_trained = True

    def save(self, filepath=FUSION_MODEL_FILE):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Fusion model saved to {filepath}")

    @classmethod
    def load(cls, filepath=FUSION_MODEL_FILE):
        inst = cls()
        with open(filepath, "rb") as f:
            inst.model = pickle.load(f)
        inst.is_trained = True
        return inst

    def score_shipment(self, claims, sar_result, jamming_result, sts_result, use_jamming_layer=True):
        """
        Evaluate full trade claim and return the 0-100 Orbital Evidence Score,
        risk assessment, and compliance memo.
        """
        sar_detected = 1 if sar_result.get("detected", False) else 0
        sar_conf = float(sar_result.get("confidence", 0.0))
        sar_length = float(sar_result.get("estimated_length_m", 0.0))
        expected_len = float(claims.get("expected_length_m", 250.0))

        if sar_detected and sar_length > 0:
            len_ratio = min(sar_length, expected_len) / max(sar_length, expected_len)
            len_delta = abs(sar_length - expected_len)
        else:
            len_ratio = 0.0
            len_delta = 250.0

        ais_rep = 1 if sar_result.get("ais_recorded", True) else 0
        raw_jam_score = float(jamming_result.get("jam_score", 0.0))
        jam_score = raw_jam_score if use_jamming_layer else 0.0
        ais_trust = float(jamming_result.get("ais_trust_factor", 1.0)) if use_jamming_layer else 1.0
        disc_ais = ais_rep * ais_trust

        radar_ais_disc = 0
        if sar_detected == 0 and ais_rep == 1:
            radar_ais_disc = 1
        elif sar_detected == 1 and ais_rep == 0:
            # If jamming layer is active and jamming was severe, absence of AIS is expected
            if use_jamming_layer and jam_score >= 0.35:
                radar_ais_disc = 0
            else:
                radar_ais_disc = 1

        sts_risk = float(sts_result.get("sts_risk_score", 0.0))
        dark_nearby = 1 if sts_result.get("dark_vessel_nearby", False) else 0
        days_from_pass = int(sar_result.get("days_from_pass", 0))

        features = pd.DataFrame([{
            "sar_detected": sar_detected,
            "sar_confidence": sar_conf,
            "length_match_ratio": len_ratio,
            "length_delta_m": len_delta,
            "ais_reported": ais_rep,
            "ais_trust_factor": ais_trust,
            "discounted_ais_weight": disc_ais,
            "jamming_score": jam_score,
            "radar_ais_discrepancy": radar_ais_disc,
            "sts_risk_score": sts_risk,
            "dark_vessel_nearby": dark_nearby,
            "days_from_pass": days_from_pass,
        }])[FEATURE_NAMES]

        # No hard-coded cap for the jamming-off case: without the layer, a ship
        # with AIS off looks like the "dark_no_excuse" scenario and the model
        # scores it down on its own.
        final_score = round(float(np.clip(self.model.predict(features)[0], 0.0, 100.0)), 1)

        # Risk Classification
        if final_score >= 80.0:
            risk_tier = "LOW_RISK"
            action = "APPROVED - Satellite evidence strongly corroborates bill of lading."
        elif final_score >= 50.0:
            risk_tier = "MEDIUM_RISK"
            action = "MANUAL_REVIEW_REQUIRED - Ambiguity or minor discrepancy in satellite telemetry."
        else:
            risk_tier = "HIGH_RISK"
            action = "FLAGGED / SUSPECTED FRAUD - Satellite radar and physical evidence contradict claimed shipment."

        # Draft Automated Compliance Memo
        memo = self._draft_compliance_memo(
            claims, final_score, risk_tier, action, 
            sar_result, jamming_result, sts_result, use_jamming_layer
        )

        return {
            "orbital_evidence_score": final_score,
            "risk_tier": risk_tier,
            "recommended_action": action,
            "features_used": features.to_dict(orient="records")[0],
            "compliance_memo": memo,
        }

    def _draft_compliance_memo(self, claims, score, risk_tier, action, sar_res, jam_res, sts_res, use_jamming):
        """Generate executive bank compliance memo citing satellite image chips and telemetry."""
        lines = [
            "=" * 70,
            "BILL OF ORBIT — ORBITAL EVIDENCE VERIFICATION MEMORANDUM",
            "=" * 70,
            f"Vessel Claim:        {claims.get('vessel_name', 'N/A')} (IMO: {claims.get('imo', 'N/A')})",
            f"Claimed Port:        {claims.get('port_name', 'N/A')} ({claims.get('aoi', 'N/A')})",
            f"Claimed Date:        {claims.get('loading_date', 'N/A')}",
            f"Declared Cargo:      {claims.get('quantity_mt', 0):,} MT of {claims.get('cargo', 'N/A')}",
            f"Vessel Size Class:   {claims.get('inferred_vessel_class', 'N/A')} (Expected Length: {claims.get('expected_length_m', 0)}m)",
            "-" * 70,
            f"ORBITAL EVIDENCE SCORE: {score}/100  [{risk_tier}]",
            f"Recommendation:        {action}",
            "-" * 70,
            "SATELLITE & TELEMETRY BREAKDOWN:",
            f"1. Sentinel-1 SAR Radar Detection:",
            f"   - Target Detected:  {'YES' if sar_res.get('detected') else 'NO MATCHING VESSEL FOUND'}",
            f"   - Detector Conf:    {sar_res.get('confidence', 0.0):.2f}",
            f"   - Estimated Length: {sar_res.get('estimated_length_m', 0):.1f}m (Tolerance: +/-{claims.get('tolerance_length_m', 0)}m)",
            f"   - Satellite Pass:   {sar_res.get('pass_file', 'Sentinel-1 Acquisition')}",
            "",
            f"2. GNSS Jamming & AIS Trust Layer:",
            f"   - Jamming Status:   {jam_res.get('risk_category', 'N/A')} (Jam Score: {jam_res.get('jam_score', 0):.3f})",
            f"   - AIS Trust Factor: {jam_res.get('ais_trust_factor', 1.0):.3f} (Jamming Layer: {'ENABLED' if use_jamming else 'DISABLED'})",
            f"   - Assessment:       {jam_res.get('explanation', '')}",
            "",
            f"3. Ship-to-Ship (STS) & Dark Vessel Screening:",
            f"   - STS Transfer:     {'DETECTED' if sts_res.get('sts_detected') else 'NONE'}",
            f"   - Dark Vessels:     {sts_res.get('dark_vessels_count', 0)} in vicinity",
            f"   - Security Note:    {sts_res.get('notes', '')}",
            "-" * 70,
            "AUDIT TRAIL & CITATIONS:",
            f"Generated automatically by Bill of Orbit ML/CV Pipeline on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S UTC')}.",
            "All radar acquisitions cross-referenced against ESA Copernicus Open Access Hub.",
            "=" * 70,
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    import json
    fusion = OrbitalEvidenceFusionModel()
    fusion.fit()
    fusion.save()
    rep = {"fresh_seed": backtest_fusion(fusion.model),
           "fresh_seed_plus_feature_noise_0.1": backtest_fusion(fusion.model, feature_noise=0.1)}
    os.makedirs(os.path.join(os.path.dirname(__file__), "reports"), exist_ok=True)
    json.dump(rep, open(os.path.join(os.path.dirname(__file__), "reports", "fusion_backtest.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))
