# Bill of Orbit — data scripts

Run order (start 04 first, it can't backfill):

| Script | Source | Output |
|---|---|---|
| `04_ais_recorder.py` | aisstream.io live AIS | `data/ais/YYYY-MM-DD.jsonl` |
| `01_gee_sentinel.py` | Google Earth Engine S1 + S2 | `data/sentinel/*.tif`, `acquisitions.csv` |
| `02_ossdd_download.py` | Hugging Face OSSDD | `data/ossdd/` |
| `03_gfw_sar.py` | Global Fishing Watch 4Wings | `data/gfw/*_sar_{all,dark}.csv` |
| `05_jamming.py` | Stanford GNSS interference monitor | `data/jamming/gulf_jamming_cells.csv` |

Secrets via env vars: `GEE_PROJECT`, `GFW_TOKEN`, `AISSTREAM_API_KEY`.
Edit AOI boxes and dates in `config.py`.

`acquisitions.csv` is the key file: write the synthetic bills of lading so their
dates line up with real Sentinel-1 pass times listed there.
