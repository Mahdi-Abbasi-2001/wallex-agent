"""Binance spot 1m klines (public, no key), cached to data/BINANCE_<SYMBOL>_1m.csv"""
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

URL = "https://api.binance.com/api/v3/klines"


def _chunk(args):
    sym, start = args
    for a in range(4):
        try:
            r = requests.get(URL, params={"symbol": sym, "interval": "1m", "startTime": start * 1000, "limit": 1000}, timeout=20)
            if r.status_code == 200: return r.json()
            time.sleep(2 ** a)
        except requests.RequestException:
            time.sleep(2 ** a)
    return []


def load(sym="BTCUSDT", days=190, refresh=False):
    f = Path("data") / f"BINANCE_{sym}_1m.csv"
    if f.exists() and not refresh: return pd.read_csv(f)
    end = int(time.time()) // 86400 * 86400; start = end - days * 86400
    starts = [(sym, s) for s in range(start, end, 60000)]
    with ThreadPoolExecutor(4) as ex: parts = list(ex.map(_chunk, starts))
    rows = [r[:6] for p in parts for r in p]
    d = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"]).astype(float); d["t"] = (d.t // 1000).astype("int64")
    d = d.drop_duplicates("t").sort_values("t"); d = d[d.t < end].reset_index(drop=True); d.to_csv(f, index=False); return d


if __name__ == "__main__":
    t0 = time.time(); d = load(); print(len(d), "candles", pd.to_datetime(d.t.iloc[0], unit="s"), "->", pd.to_datetime(d.t.iloc[-1], unit="s"), f"{time.time()-t0:.0f}s")
    exp = (d.t.iloc[-1] - d.t.iloc[0]) // 60 + 1; print("expected", exp, "missing", exp - len(d), "| flat candles", f"{((d.o==d.h)&(d.h==d.l)&(d.l==d.c)).mean()*100:.1f}%")
