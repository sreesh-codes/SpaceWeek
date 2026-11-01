"""Record live AIS from aisstream.io for the Fujairah / Khor Fakkan / Sohar coast.

aisstream has NO history: whatever you don't record, you never get. Run this on
a machine that stays on (not a laptop that sleeps), e.g. inside tmux/screen:

    export AISSTREAM_API_KEY=...
    python 04_ais_recorder.py

Writes one JSON line per message to data/ais/YYYY-MM-DD.jsonl (UTC day).
Reconnects automatically if the socket drops.
"""
import asyncio
import datetime as dt
import json
import os
import time

import websockets

from config import AIS_BOXES, AISSTREAM_API_KEY

URL = "wss://stream.aisstream.io/v0/stream"
OUT = "data/ais"
TYPES = ["PositionReport", "StandardClassBPositionReport", "ShipStaticData"]


async def run_once(stats):
    async with websockets.connect(URL, ping_interval=20, ping_timeout=20) as ws:
        # must be sent within 3 seconds of connecting
        await ws.send(json.dumps({
            "APIKey": AISSTREAM_API_KEY,
            "BoundingBoxes": AIS_BOXES,
            "FilterMessageTypes": TYPES,
        }))
        print(f"[{dt.datetime.utcnow():%H:%M:%S}Z] subscribed")
        async for raw in ws:
            msg = json.loads(raw)
            if "error" in msg:
                raise RuntimeError(msg["error"])
            now = dt.datetime.utcnow()
            msg["received_utc"] = now.isoformat() + "Z"
            with open(os.path.join(OUT, f"{now:%Y-%m-%d}.jsonl"), "a") as f:
                f.write(json.dumps(msg) + "\n")
            stats["n"] += 1
            mmsi = msg.get("MetaData", {}).get("MMSI")
            if mmsi:
                stats["ships"].add(mmsi)
            if time.time() - stats["last"] > 60:
                stats["last"] = time.time()
                print(f"[{now:%H:%M:%S}Z] {stats['n']} msgs, {len(stats['ships'])} unique ships")


async def main():
    if not AISSTREAM_API_KEY:
        raise SystemExit("Set AISSTREAM_API_KEY env var")
    os.makedirs(OUT, exist_ok=True)
    stats = {"n": 0, "ships": set(), "last": time.time()}
    backoff = 5
    while True:
        try:
            await run_once(stats)
            backoff = 5
        except Exception as e:
            print(f"connection lost ({e}); reconnecting in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 300)


if __name__ == "__main__":
    asyncio.run(main())
