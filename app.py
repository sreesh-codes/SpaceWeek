"""Bill of Orbit - Interactive Satellite Trade Finance Dashboard.

Theme 3 - Seas of Data
"Your bank checks the paper. We check the planet."

Run with:
    streamlit run app.py
"""
import os
import streamlit as st
import pandas as pd
import numpy as np

from pipeline import BillOfOrbitPipeline

st.set_page_config(
    page_title="Bill of Orbit | Satellite Second Opinion for Trade Finance",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for fintech terminal look
st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .stMetric { background-color: #161b22; padding: 12px; border-radius: 8px; border: 1px solid #30363d; }
    .risk-low { color: #2ea043; font-weight: bold; font-size: 1.2rem; }
    .risk-med { color: #d29922; font-weight: bold; font-size: 1.2rem; }
    .risk-high { color: #f85149; font-weight: bold; font-size: 1.2rem; }
    .memo-box { background-color: #161b22; color: #e6edf3; font-family: monospace; padding: 16px; border-radius: 8px; border-left: 4px solid #58a6ff; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    return BillOfOrbitPipeline()


pipeline = get_pipeline()

# Header
st.title("🛰️ Bill of Orbit — Satellite Second Opinion for Trade Finance")
st.caption("**Theme 3: Seas of Data** · *'Your bank checks the paper. We check the planet.'*")

# Preset documents
PRESETS = {
    "Scenario 1: Clean Verified Shipment (Fujairah)": {
        "text": """BILL OF LADING FOR OCEAN TRANSPORT
B/L Number: BL-DXB-2026-9812
Vessel: MT SEA EMPRESS
IMO No: 9412345
Port of Loading: Fujairah Anchorage, UAE
Shipped on Board Date: 2026-09-28
Description of Goods: Light Arabian Crude Oil in Bulk
Gross Weight: 135,000 MT
Notify Party: Emirates NBD Trade Operations""",
        "sar_override": {
            "detected": True,
            "confidence": 0.94,
            "estimated_length_m": 268.0,
            "pass_file": "fujairah_anchorage_s1_20260928T141550.tif",
            "days_from_pass": 0,
            "ais_recorded": True,
        }
    },
    "Scenario 2: Ghost Ship / Fake Shipment (Khor Fakkan)": {
        "text": """BILL OF LADING FOR OCEAN TRANSPORT
B/L Number: BL-DXB-2026-4401
Vessel: MT PHANTOM VOYAGER
IMO No: 9811234
Port of Loading: Khor Fakkan Anchorage, UAE
Shipped on Board Date: 2026-10-14
Description of Goods: High Sulfur Fuel Oil (HSFO 380)
Gross Weight: 140,000 MT
Notify Party: Dubai Islamic Bank Trade Credit""",
        "sar_override": {
            "detected": False,
            "confidence": 0.0,
            "estimated_length_m": 0.0,
            "pass_file": "khor_fakkan_s1_20261014T141550.tif",
            "days_from_pass": 0,
            "ais_recorded": False,
        }
    },
    "Scenario 3: Hormuz GPS Jamming Zone (AIS Disabled)": {
        "text": """BILL OF LADING FOR OCEAN TRANSPORT
B/L Number: BL-SOH-2026-7731
Vessel: MT ARABIAN SHIELD
IMO No: 9355412
Port of Loading: Sohar Port, Oman
Shipped on Board Date: 2026-10-27
Description of Goods: Murban Crude Oil
Gross Weight: 260,000 MT
Notify Party: Abu Dhabi Commercial Bank Sanctions Compliance""",
        "sar_override": {
            "detected": True,
            "confidence": 0.93,
            "estimated_length_m": 324.0,
            "pass_file": "sohar_s1_20261027T020640.tif",
            "days_from_pass": 0,
            "ais_recorded": False,  # AIS lost due to severe jamming
        }
    },
    "Scenario 4: Undisclosed Ship-to-Ship (STS) Transfer": {
        "text": """BILL OF LADING FOR OCEAN TRANSPORT
B/L Number: BL-FUJ-2026-1188
Vessel: MT CASPIAN LEADER
IMO No: 9284711
Port of Loading: Fujairah Anchorage, UAE
Shipped on Board Date: 2026-10-16
Description of Goods: Heavy Basra Crude Oil
Gross Weight: 120,000 MT
Notify Party: First Abu Dhabi Bank Trade Desk""",
        "sar_override": {
            "detected": True,
            "confidence": 0.91,
            "estimated_length_m": 255.0,
            "pass_file": "fujairah_anchorage_s1_20261016T141550.tif",
            "days_from_pass": 0,
            "ais_recorded": True,
        }
    }
}

# Sidebar
st.sidebar.header("⚙️ Inspection Controls")
preset_name = st.sidebar.selectbox("Load Test Scenario", list(PRESETS.keys()))
use_jamming_layer = st.sidebar.toggle("Enable Jamming-Trust Layer", value=True, help="Accounting for GNSS jamming in the Gulf prevents false alarms when ships disable AIS due to electronic interference.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🗺️ Gulf Strategic Anchorages")
st.sidebar.markdown("- **Fujairah Anchorage:** `25.13°N, 56.44°E`\n- **Khor Fakkan:** `25.36°N, 56.41°E`\n- **Sohar Port:** `24.49°N, 56.66°E`\n- **Strait of Hormuz:** `26.56°N, 56.25°E`")

doc_input = st.text_area("Bill of Lading / Letter of Credit Document Text", PRESETS[preset_name]["text"], height=190)

if st.button("🚀 Run Orbital Verification", type="primary"):
    sar_override = PRESETS[preset_name]["sar_override"]
    res = pipeline.verify_document(doc_input, simulated_sar_override=sar_override, use_jamming_layer=use_jamming_layer)

    claims = res["claims"]
    eval_res = res["evaluation"]
    score = eval_res["orbital_evidence_score"]
    risk = eval_res["risk_tier"]

    # Top metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Orbital Evidence Score", f"{score}/100", delta=f"{risk}")
    with m2:
        st.metric("Vessel Identified", claims["vessel_name"], f"IMO: {claims['imo']}")
    with m3:
        st.metric("Declared Cargo", f"{claims['quantity_mt']:,.0f} MT", claims["cargo"])
    with m4:
        st.metric("Sentinel-1 SAR Match", "VERIFIED (10m Res)" if res["sar_detection"]["detected"] else "NO SHIP FOUND", 
                  f"Len: {res['sar_detection']['estimated_length_m']:.0f}m / Claim: {claims['expected_length_m']}m")

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("🛰️ Satellite Telemetry & Multi-Model Breakdown")

        tab1, tab2, tab3 = st.tabs(["Radar CV Detector", "Jamming Trust Map", "STS & Dark Vessels"])

        with tab1:
            st.markdown(f"**Fine-tuned YOLO11n-SAR Detector:**")
            st.write(f"- Detection Status: `{'Vessel Located' if res['sar_detection']['detected'] else 'Empty Sea'}`")
            st.write(f"- Detection Confidence: `{res['sar_detection']['confidence']:.2f}`")
            st.write(f"- Physical Radar Length: `{res['sar_detection']['estimated_length_m']:.1f} meters`")
            st.write(f"- Claimed Tanker Class: `{claims['inferred_vessel_class']}` (Expected `{claims['expected_length_m']}m`)")
            st.write(f"- Satellite Pass File: `{res['sar_detection']['pass_file']}`")

        with tab2:
            st.markdown(f"**Spatio-Temporal GNSS Interference Anomaly Model:**")
            st.write(f"- Jamming Intensity at Port: `{res['jamming_analysis']['jam_score']:.3f}`")
            st.write(f"- Anomaly Status: `{'INTERFERENCE DETECTED' if res['jamming_analysis']['is_jammed'] else 'NORMAL GNSS'}`")
            st.write(f"- AIS Trust Multiplier: `{res['jamming_analysis']['ais_trust_factor']:.3f}`")
            st.info(res['jamming_analysis']['explanation'])

            if not use_jamming_layer:
                st.warning("⚠️ Jamming layer is currently DISABLED. The model is evaluating transponders with naive assumptions.")

        with tab3:
            st.markdown(f"**Ship-to-Ship (STS) & Dark Vessel Screening:**")
            st.write(f"- Offshore STS Mooring Flag: `{'CRITICAL: STS DETECTED' if res['sts_analysis']['sts_detected'] else 'CLEAN / NO STS'}`")
            st.write(f"- Dark Vessels in 1.5km Radius: `{res['sts_analysis']['dark_vessels_count']}`")
            st.write(f"- Min Distance to Nearby Vessels: `{res['sts_analysis']['min_distance_m']}m`")
            st.write(f"- Analysis Note: {res['sts_analysis']['notes']}")

        # Map visualization
        st.subheader("📍 Geospatial Anchorage Position")
        map_data = pd.DataFrame([{
            "lat": claims["lat"],
            "lon": claims["lon"],
            "vessel": claims["vessel_name"]
        }])
        st.map(map_data, zoom=9)

    with col_right:
        st.subheader("📑 Bank Compliance Evidence Memo")
        st.markdown(f"<div class='memo-box'><pre>{eval_res['compliance_memo']}</pre></div>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Download Compliance Memorandum (TXT)",
            data=eval_res["compliance_memo"],
            file_name=f"Orbital_Evidence_Memo_{claims['vessel_name'].replace(' ', '_')}.txt",
            mime="text/plain"
        )
