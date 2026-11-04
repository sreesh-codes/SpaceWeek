"""Document Claim Extractor for Trade Finance (Bills of Lading & Letters of Credit).

Part of Bill of Orbit (Theme 3 - Seas of Data).
Reads unstructured shipping documents, bills of lading, and letters of credit,
and extracts verified structured claims:
- Vessel Name & IMO/MMSI
- Port/Anchorage of Loading
- Shipment Date
- Cargo Type & Quantity (MT / BBL)
- Claimed Vessel Class & Expected Length (m)
"""
import re
import json
from datetime import datetime

# Port/Anchorage coordinates and lookup mapping
PORT_DATABASE = {
    "fujairah": {
        "name": "Fujairah Anchorage / Port",
        "aoi": "fujairah_anchorage",
        "lat": 25.13,
        "lon": 56.44,
        "country": "UAE",
    },
    "khor fakkan": {
        "name": "Khor Fakkan Anchorage",
        "aoi": "khor_fakkan",
        "lat": 25.36,
        "lon": 56.41,
        "country": "UAE",
    },
    "sohar": {
        "name": "Sohar Port & Anchorage",
        "aoi": "sohar",
        "lat": 24.49,
        "lon": 56.66,
        "country": "Oman",
    },
    "dubai": {
        "name": "Dubai Offshore / Jebel Ali",
        "aoi": "dubai",
        "lat": 25.02,
        "lon": 55.06,
        "country": "UAE",
    },
}

# Standard tanker size categories and expected lengths
VESSEL_CLASSES = {
    "VLCC": {"length_m": 330, "beam_m": 60, "dwt_range": (200000, 320000)},
    "Suezmax": {"length_m": 274, "beam_m": 48, "dwt_range": (120000, 200000)},
    "Aframax": {"length_m": 245, "beam_m": 42, "dwt_range": (80000, 120000)},
    "Panamax": {"length_m": 228, "beam_m": 32, "dwt_range": (60000, 80000)},
    "Handymax": {"length_m": 190, "beam_m": 28, "dwt_range": (35000, 60000)},
    "Handysize": {"length_m": 160, "beam_m": 24, "dwt_range": (10000, 35000)},
    "General Cargo / Coastal": {"length_m": 110, "beam_m": 16, "dwt_range": (2000, 10000)},
}


class DocumentClaimExtractor:
    """Extracts structured trade claims from Bill of Lading (BL) text."""

    def __init__(self):
        pass

    def extract_claims(self, text):
        """
        Extract structured trade claims from document text.
        """
        lines = text.splitlines()

        # 1. Vessel Name
        vessel_match = re.search(
            r"(?:vessel(?:\s+name)?|m\.?v\.?|m\.?t\.?|tanker)\s*[:\-]?\s*([A-Z0-9\s\.\-]{3,30})",
            text, re.IGNORECASE
        )
        vessel_name = vessel_match.group(1).strip() if vessel_match else "UNKNOWN VESSEL"
        # Clean up trailing words
        vessel_name = re.sub(r"\s+(?:imo|flag|voyage|port|date|shipped|loaded|as|of).*", "", vessel_name, flags=re.IGNORECASE).strip()

        # 2. IMO number
        imo_match = re.search(r"\bIMO\s*(?:NO\.?)?\s*[:\-]?\s*([0-9]{7})\b", text, re.IGNORECASE)
        imo = imo_match.group(1) if imo_match else None

        # 3. Port of Loading
        port_match = re.search(r"(?:port\s+of\s+loading|loading\s+port|anchorage)\s*[:\-]?\s*([A-Za-z\s,]{3,30})", text, re.IGNORECASE)
        port_str = port_match.group(1).strip().lower() if port_match else text.lower()
        
        matched_port = None
        for key, pinfo in PORT_DATABASE.items():
            if key in port_str or key in text.lower():
                matched_port = pinfo
                break
        if not matched_port:
            matched_port = PORT_DATABASE["fujairah"]  # default fallback

        # 4. Loading Date
        date_patterns = [
            r"\b(202[0-9]-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01]))\b",  # YYYY-MM-DD
            r"\b((?:0[1-9]|[12][0-9]|3[01])\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+202[0-9])\b",
            r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(?:0[1-9]|[12][0-9]|3[01]),?\s+202[0-9])\b",
        ]
        parsed_date = "2026-10-15"
        for pat in date_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                raw_d = m.group(1)
                try:
                    for fmt in ("%Y-%m-%d", "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y", "%b %d %Y"):
                        try:
                            parsed_date = datetime.strptime(raw_d.replace(",", ""), fmt).strftime("%Y-%m-%d")
                            break
                        except ValueError:
                            pass
                except Exception:
                    pass
                break

        # 5. Cargo and Quantity
        cargo_match = re.search(r"(?:cargo(?:\s+description)?|description\s+of\s+goods|goods|commodity)\s*[:\-]?\s*([A-Za-z0-9\s\.\-]{3,40})", text, re.IGNORECASE)
        cargo_desc = cargo_match.group(1).strip() if cargo_match else "Crude Oil"
        cargo_desc = re.sub(r"^(?:of\s+goods|description\s+of\s+goods)\s*[:\-]?\s*", "", cargo_desc, flags=re.IGNORECASE)
        cargo_desc = re.sub(r"\s+(?:gross|net|weight|quantity|date).*", "", cargo_desc, flags=re.IGNORECASE).strip()

        qty_match = re.search(r"([0-9,]+(?:\.[0-9]+)?)\s*(?:MT|METRIC\s*TONS?|BBLS?|BARRELS?)", text, re.IGNORECASE)
        qty = float(qty_match.group(1).replace(",", "")) if qty_match else 80000.0

        # Infer vessel size class from quantity
        if qty > 180000:
            v_class = "VLCC"
        elif qty > 110000:
            v_class = "Suezmax"
        elif qty > 70000:
            v_class = "Aframax"
        elif qty > 45000:
            v_class = "Panamax"
        elif qty > 25000:
            v_class = "Handymax"
        else:
            v_class = "Handysize"

        expected_length = VESSEL_CLASSES[v_class]["length_m"]

        return {
            "vessel_name": vessel_name.upper(),
            "imo": imo,
            "port_name": matched_port["name"],
            "aoi": matched_port["aoi"],
            "lat": matched_port["lat"],
            "lon": matched_port["lon"],
            "loading_date": parsed_date,
            "cargo": cargo_desc,
            "quantity_mt": qty,
            "inferred_vessel_class": v_class,
            "expected_length_m": expected_length,
            "tolerance_length_m": round(expected_length * 0.20, 1),
        }


if __name__ == "__main__":
    extractor = DocumentClaimExtractor()
    sample_bl = """
    BILL OF LADING FOR OCEAN TRANSPORT
    B/L Number: BL-DXB-2026-9812
    Vessel: MT SEA EMPRESS
    IMO No: 9412345
    Port of Loading: Fujairah Anchorage, UAE
    Shipped on Board Date: 2026-09-28
    Description of Goods: Light Arabian Crude Oil in Bulk
    Gross Weight: 135,000 MT
    Notify Party: National Bank of Dubai Trade Finance Dept
    """
    claims = extractor.extract_claims(sample_bl)
    print("--- Extracted Claims ---")
    for k, v in claims.items():
        print(f"  {k}: {v}")
