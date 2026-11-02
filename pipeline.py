"""Bill of Orbit - Full ML/CV Orchestration Pipeline & Verification Suite.

Theme 3 - Seas of Data
"Your bank checks the paper. We check the planet."

Integrates:
1. Document Claim Extractor (NLP/regex)
2. Fine-tuned YOLO SAR Ship Detector (Computer Vision)
3. Spatio-Temporal Jamming-Trust Model (Anomaly Detection & ML)
4. Ship-to-Ship (STS) & Dark Vessel Screening (Global Fishing Watch & Radar)
5. Gradient-Boosted Orbital Evidence Fusion Model (LightGBM/HistGB)
"""
import os
import json
import pandas as pd

from document_extractor import DocumentClaimExtractor
from jamming_model import JammingTrustModel
from sts_detector import STSDetector
from fusion_model import OrbitalEvidenceFusionModel

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
ACQUISITIONS_CSV = os.path.join(REPO_DIR, "data", "sentinel", "acquisitions.csv")
# The .tif clips are git-ignored (rebuild with 01_gee_sentinel.py); point
# BOO_SENTINEL_DIR at them if they live elsewhere.
SENTINEL_DIR = os.environ.get("BOO_SENTINEL_DIR", os.path.join(REPO_DIR, "data", "sentinel"))
SAR_BACKTEST = os.path.join(REPO_DIR, "reports", "sar_backtest.json")
GFW_BACKTEST = os.path.join(REPO_DIR, "reports", "gfw_backtest.json")
MIN_LENGTH_RATIO = 0.8  # a detection within ~20% of the claimed length counts as a match


class BillOfOrbitPipeline:
    """Full end-to-end verification pipeline."""

    def __init__(self):
        print("Initializing Bill of Orbit ML/CV Pipeline...")
        self.doc_extractor = DocumentClaimExtractor()
        self.jamming_model = JammingTrustModel.load()
        self.sts_detector = STSDetector()
        self.fusion_model = OrbitalEvidenceFusionModel.load()
        self.acquisitions = pd.read_csv(ACQUISITIONS_CSV) if os.path.exists(ACQUISITIONS_CSV) else None
        self.yolo_model = None
        self.det_conf = 0.25
        self._scene_cache = {}
        self._init_yolo()

    def _init_yolo(self):
        try:
            from sar_inference import load_model
            self.yolo_model = load_model()
            name = os.path.basename(os.path.dirname(os.path.dirname(self.yolo_model.ckpt_path)))
            # Full scenes: threshold chosen against GFW on September passes
            # (backtest_gfw.py); fallback: the chip threshold chosen on OSSDD val.
            gfw = json.load(open(GFW_BACKTEST)) if os.path.exists(GFW_BACKTEST) else {}
            if gfw.get("scene_conf") is not None and os.path.basename(
                    os.path.dirname(os.path.dirname(gfw.get("weights", "")))) == name:
                self.det_conf = gfw["scene_conf"]
            elif os.path.exists(SAR_BACKTEST):
                self.det_conf = json.load(open(SAR_BACKTEST)).get(name, {}).get("conf_threshold_from_val", 0.25)
            print(f"Loaded SAR ship detector {self.yolo_model.ckpt_path} (conf {self.det_conf})")
        except Exception as e:
            print(f"Notice: SAR detector unavailable ({e}); only scripted SAR results can be used.")

    def find_sentinel_pass(self, aoi, target_date, max_window_days=3):
        """Closest Sentinel-1 pass for the AOI; flags passes outside the window."""
        if self.acquisitions is None:
            return None
        df = self.acquisitions[(self.acquisitions["aoi"] == aoi) & (self.acquisitions["sensor"] == "s1")
                               & self.acquisitions["file"].notna()].copy()
        if df.empty:
            return None
        df["dt"] = pd.to_datetime(df["utc_time"].str.rstrip("Z"))
        target = pd.to_datetime(target_date)
        df["diff_days"] = (df["dt"].dt.normalize() - target).abs().dt.days
        c = df.sort_values(["diff_days", "dt"]).iloc[0]
        path = os.path.join(SENTINEL_DIR, c["file"])
        return {
            "file": c["file"],
            "path": path,
            "pass_time_utc": c["utc_time"],
            "pass_date": c["dt"].strftime("%Y-%m-%d"),
            "orbit_pass": c.get("orbit_pass"),
            "days_diff": int(c["diff_days"]),
            "within_window": bool(c["diff_days"] <= max_window_days),
            "image_available": os.path.exists(path),
        }

    def scene_detections(self, path):
        if path not in self._scene_cache:
            from sar_inference import detect_scene
            self._scene_cache[path] = detect_scene(self.yolo_model, path, conf=self.det_conf)
        return self._scene_cache[path]

    def sar_evidence(self, claims, s1_pass, ais_recorded=True):
        """Run the detector on the matched pass and pick the detection that best
        matches the claimed vessel size. Radar cannot identify *which* ship it
        is, so this shows whether a ship of the claimed size was there."""
        if s1_pass is None or not s1_pass["image_available"] or self.yolo_model is None:
            why = ("no Sentinel-1 pass listed for this AOI" if s1_pass is None else
                   f"image {s1_pass['file']} not found in {SENTINEL_DIR}" if not s1_pass["image_available"]
                   else "detector weights missing")
            raise FileNotFoundError(f"Cannot run SAR detection: {why}")
        dets = self.scene_detections(s1_pass["path"])
        exp = float(claims["expected_length_m"])
        best = None
        for d in dets:
            ratio = min(d["length_m"], exp) / max(d["length_m"], exp)
            key = (ratio >= MIN_LENGTH_RATIO, ratio * d["conf"])
            if best is None or key > best[0]:
                best = (key, d, ratio)
        result = {
            "detected": best is not None,
            "confidence": round(best[1]["conf"], 3) if best else 0.0,
            "estimated_length_m": best[1]["length_m"] if best else 0.0,
            "length_match": bool(best and best[2] >= MIN_LENGTH_RATIO),
            "vessel_lat": best[1]["lat"] if best else None,
            "vessel_lon": best[1]["lon"] if best else None,
            "ships_in_scene": len(dets),
            "size_matched_ships": sum(min(d["length_m"], exp) / max(d["length_m"], exp) >= MIN_LENGTH_RATIO for d in dets),
            "pass_file": s1_pass["file"],
            "days_from_pass": s1_pass["days_diff"],
            "ais_recorded": ais_recorded,
            "source": "Sentinel-1 detector",
        }
        return result, dets

    def verify_document(self, doc_text, simulated_sar_override=None, use_jamming_layer=True, ais_recorded=True):
        """Process a Bill of Lading and compute verification evidence.

        simulated_sar_override: scripted SAR result (demo only). By default the
        detector runs on the matched Sentinel-1 pass. ais_recorded: whether AIS
        placed the vessel there (no Gulf AIS feed is wired in yet, so the
        caller supplies it).
        """
        claims = self.doc_extractor.extract_claims(doc_text)
        s1_pass = self.find_sentinel_pass(claims["aoi"], claims["loading_date"])

        dets = None
        if simulated_sar_override is not None:
            sar_result = dict(simulated_sar_override, source="scripted override (demo)")
        else:
            sar_result, dets = self.sar_evidence(claims, s1_pass, ais_recorded)
            if s1_pass and not s1_pass["within_window"]:
                sar_result["note"] = f"Nearest pass is {s1_pass['days_diff']} days from the claimed date"

        jamming_result = self.jamming_model.predict(claims["lat"], claims["lon"], claims["loading_date"])

        # look for STS around the matched vessel if radar located it, else the port
        lat = sar_result.get("vessel_lat") or claims["lat"]
        lon = sar_result.get("vessel_lon") or claims["lon"]
        sts_result = self.sts_detector.detect_sts_transfer(lat, lon, claims["loading_date"], sar_detections=dets)

        final_result = self.fusion_model.score_shipment(
            claims=claims,
            sar_result=sar_result,
            jamming_result=jamming_result,
            sts_result=sts_result,
            use_jamming_layer=use_jamming_layer
        )

        return {
            "claims": claims,
            "satellite_pass": s1_pass,
            "sar_detection": sar_result,
            "jamming_analysis": jamming_result,
            "sts_analysis": sts_result,
            "evaluation": final_result,
        }


def run_synthetic_benchmarks():
    """
    Run the 3 synthetic document packets described in the 5-day plan:
    1. Clean shipment (real ship present, radar and AIS match)
    2. Phantom / Ghost ship (no ship on radar, paper fraud)
    3. Jamming Corridor test (AIS suppressed by jamming, radar proves presence)
    4. Undisclosed STS Transfer (moored side-by-side with dark tanker)
    """
    pipeline = BillOfOrbitPipeline()

    scenarios = [
        {
            "name": "Scenario 1: Clean Verified Crude Shipment (Fujairah Anchorage)",
            "text": """
            BILL OF LADING FOR OCEAN TRANSPORT
            B/L Number: BL-DXB-2026-9812
            Vessel: MT SEA EMPRESS
            IMO No: 9412345
            Port of Loading: Fujairah Anchorage, UAE
            Shipped on Board Date: 2026-09-28
            Description of Goods: Light Arabian Crude Oil in Bulk
            Gross Weight: 135,000 MT
            Notify Party: Emirates NBD Trade Operations
            """,
            "sar_override": {
                "detected": True,
                "confidence": 0.94,
                "estimated_length_m": 268.0,  # Matches Suezmax ~274m
                "pass_file": "fujairah_anchorage_s1_20260928T141550.tif",
                "days_from_pass": 0,
                "ais_recorded": True,
            },
        },
        {
            "name": "Scenario 2: Ghost Ship / Fraudulent Claim (No Ship on Radar)",
            "text": """
            BILL OF LADING FOR OCEAN TRANSPORT
            B/L Number: BL-DXB-2026-4401
            Vessel: MT PHANTOM VOYAGER
            IMO No: 9811234
            Port of Loading: Khor Fakkan Anchorage, UAE
            Shipped on Board Date: 2026-10-14
            Description of Goods: High Sulfur Fuel Oil (HSFO 380)
            Gross Weight: 140,000 MT
            Notify Party: Dubai Islamic Bank Trade Credit
            """,
            "sar_override": {
                "detected": False,
                "confidence": 0.0,
                "estimated_length_m": 0.0,
                "pass_file": "khor_fakkan_s1_20261014T141550.tif",
                "days_from_pass": 0,
                "ais_recorded": False,
            },
        },
        {
            "name": "Scenario 3: Hormuz GPS Jamming Zone (Proving Jamming-Trust Layer)",
            "text": """
            BILL OF LADING FOR OCEAN TRANSPORT
            B/L Number: BL-SOH-2026-7731
            Vessel: MT ARABIAN SHIELD
            IMO No: 9355412
            Port of Loading: Sohar Port, Oman
            Shipped on Board Date: 2026-09-10
            Description of Goods: Murban Crude Oil
            Gross Weight: 260,000 MT
            Notify Party: Abu Dhabi Commercial Bank Sanctions Compliance
            """,
            "sar_override": {
                "detected": True,
                "confidence": 0.92,
                "estimated_length_m": 322.0,  # VLCC match (~330m)
                "pass_file": "sohar_s1_20260910T020640.tif",
                "days_from_pass": 0,
                "ais_recorded": False,  # AIS was jammed/off!
            },
            "test_jamming_toggle": True,
        },
    ]

    print("\n" + "=" * 80)
    print("RUNNING BILL OF ORBIT BENCHMARK SUITE")
    print("=" * 80)

    for sc in scenarios:
        print(f"\n>>> EVALUATING: {sc['name']}")
        res = pipeline.verify_document(sc["text"], simulated_sar_override=sc["sar_override"])
        print(res["evaluation"]["compliance_memo"])

        # same document, but with the real detector on the matched Sentinel-1 pass
        try:
            real = pipeline.verify_document(sc["text"], ais_recorded=sc["sar_override"]["ais_recorded"])
            sd = real["sar_detection"]
            print(f"REAL SAR on {sd['pass_file']} (+/-{sd['days_from_pass']} d): {sd['ships_in_scene']} ships, "
                  f"{sd['size_matched_ships']} of claimed size; best {sd['estimated_length_m']:.0f} m "
                  f"conf {sd['confidence']:.2f} -> score {real['evaluation']['orbital_evidence_score']}/100 "
                  f"[{real['evaluation']['risk_tier']}]")
        except FileNotFoundError as e:
            print(f"REAL SAR skipped: {e}")

        if sc.get("test_jamming_toggle"):
            res_no_jam = pipeline.verify_document(
                sc["text"], simulated_sar_override=sc["sar_override"], use_jamming_layer=False
            )
            print("\n*** JAMMING LAYER TOGGLE ***")
            print(f"Jamming-Trust layer ON:  score {res['evaluation']['orbital_evidence_score']}/100")
            print(f"Jamming-Trust layer OFF: score {res_no_jam['evaluation']['orbital_evidence_score']}/100")


if __name__ == "__main__":
    run_synthetic_benchmarks()
