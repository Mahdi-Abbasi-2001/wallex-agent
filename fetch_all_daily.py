"""Wallex USDT spot coin list -> Binance daily candles (where the coin exists on Binance). Cache data/ALLD_<SYM>.csv"""
import time
from pathlib import Path

import pandas as pd
import requests

STABLE = {"USDT", "USDC", "FDUSD", "TUSD", "USDS", "DAI", "BUSD", "USDP", "USDE", "PYUSD", "USD1", "USDSUSDT", "XUSD", "BFUSD", "AEUR", "EURI"}


def wallex_bases():
    m = requests.get("https://api.wallex.ir/hector/web/v1/markets", timeout=20).json()["result"]["markets"]
    return sorted({x["base_asset"] for x in m if x.get("is_spot") and x["quote_asset"] == "USDT" and x["base_asset"] not in STABLE}), {x["base_asset"]: x for x in m if x.get("is_spot") and x["quote_asset"] == "USDT"}


def fetch(sym, days=1460):
    f = Path("data") / f"ALLD_{sym}.csv"
    if f.exists(): return pd.read_csv(f)
    end = int(time.time()) // 86400 * 86400; t = end - days * 86400; rows = []
    while t < end:
        r = requests.get("https://api.binance.com/api/v3/klines", params={"symbol": sym, "interval": "1d", "startTime": t * 1000, "limit": 1000}, timeout=20)
        if r.status_code != 200: return None
        data = r.json()
        if not data: break
        rows += [[x[0], x[1], x[2], x[3], x[4], x[5], x[7]] for x in data]; t = int(data[-1][0] // 1000) + 86400; time.sleep(0.2)
    if not rows: return None
    d = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v", "qv"]).astype(float); d["t"] = (d.t // 1000).astype("int64")
    d = d.drop_duplicates("t").sort_values("t"); d = d[d.t < end].reset_index(drop=True); d.to_csv(f, index=False); return d


if __name__ == "__main__":
    bases, info = wallex_bases(); print(f"Wallex USDT spot bases (excl. stablecoins): {len(bases)}", flush=True)
    ok, missing = [], []
    for b in bases:
        d = fetch(b + "USDT")
        if d is None or len(d) < 60: missing.append(b)
        else: ok.append(b)
    pd.to_pickle(dict(ok=ok, missing=missing, wallex_vol={b: float(info[b].get("quote_volume_24h") or 0) for b in bases}), "data/ALLD_meta.pkl")
    print(f"on Binance with >=60 daily bars: {len(ok)} | not on Binance / too short: {len(missing)}\nmissing: {missing}")
