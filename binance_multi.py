"""Bulk Binance spot klines (public). Caches data/MULTI_<SYM>_<interval>.csv. python binance_multi.py"""
import sys
import time
from pathlib import Path

import pandas as pd
import requests

SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "TRXUSDT", "LINKUSDT", "PAXGUSDT"]
SEC = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
DAYS = {"15m": 730, "1h": 1460, "4h": 1460, "1d": 1460}
URL = "https://api.binance.com/api/v3/klines"


def fetch(sym, interval, days):
    f = Path("data") / f"MULTI_{sym}_{interval}.csv"
    if f.exists(): return pd.read_csv(f)
    sec = SEC[interval]; end = int(time.time()) // sec * sec; t = end - days * 86400; rows = []
    while t < end:
        for attempt in range(5):
            try:
                r = requests.get(URL, params={"symbol": sym, "interval": interval, "startTime": t * 1000, "limit": 1000}, timeout=20)
                if r.status_code == 200: break
                time.sleep(2 ** attempt)
            except requests.RequestException: time.sleep(2 ** attempt)
        data = r.json() if r.status_code == 200 else []
        if not data: break
        rows += [x[:6] for x in data]; t = int(data[-1][0] // 1000) + sec; time.sleep(0.25)
    d = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"]).astype(float); d["t"] = (d.t // 1000).astype("int64")
    d = d.drop_duplicates("t").sort_values("t"); d = d[d.t < end].reset_index(drop=True); d.to_csv(f, index=False); return d


if __name__ == "__main__":
    for interval in ("1d", "4h", "1h", "15m"):
        for s in SYMS:
            t0 = time.time(); d = fetch(s, interval, DAYS[interval])
            print(f"{interval:>3} {s:9} {len(d):6d} bars {pd.to_datetime(d.t.iloc[0], unit='s').date()} -> {pd.to_datetime(d.t.iloc[-1], unit='s').date()} ({time.time()-t0:.0f}s)", flush=True)
