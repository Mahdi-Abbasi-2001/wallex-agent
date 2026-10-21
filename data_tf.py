"""1-minute candles (API gives 1 day per request at 1m) cached once, resampled to any timeframe."""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from agent.wallex_client import Wallex

CACHE = Path(__file__).parent / "data"
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "TRXUSDT"]


def _fetch_day(args):
    sym, start = args
    w = Wallex()
    for _ in range(3):
        try:
            c = w.candles(sym, "1", start, start + 86400 - 1)
            return list(zip(c["t"], c["o"], c["h"], c["l"], c["c"], c["v"])) if c.get("s") == "ok" else []
        except Exception:
            time.sleep(1)
    return []


def load_1m(sym, days=180, refresh=False):
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{sym}_1m.csv"
    if f.exists() and not refresh:
        return pd.read_csv(f)
    end = int(time.time()) // 86400 * 86400
    with ThreadPoolExecutor(6) as ex:
        parts = list(ex.map(_fetch_day, [(sym, end - d * 86400) for d in range(days, 0, -1)]))
    d = pd.DataFrame([r for p in parts for r in p], columns=["t", "o", "h", "l", "c", "v"]).astype(float)
    d = d.drop_duplicates("t").sort_values("t").reset_index(drop=True)
    d["t"] = d["t"].astype("int64")
    d.to_csv(f, index=False)
    return d


def resample(d1, minutes):
    """Build `minutes`-minute candles from 1m. Timestamps are bar-open, unix seconds."""
    if minutes == 1:
        return d1.copy()
    b = (d1["t"] // (60 * minutes)) * (60 * minutes)
    g = d1.groupby(b)
    r = pd.DataFrame({"o": g["o"].first(), "h": g["h"].max(), "l": g["l"].min(), "c": g["c"].last(), "v": g["v"].sum()})
    r.insert(0, "t", r.index.astype("int64"))
    return r.reset_index(drop=True)


if __name__ == "__main__":
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 180
    for s in SYMS:
        t0 = time.time(); d = load_1m(s, days)
        print(s, len(d), "1m bars", pd.to_datetime(d.t.iloc[0], unit="s").date(), "->", pd.to_datetime(d.t.iloc[-1], unit="s").date(),
              f"| 5m bars {len(resample(d, 5))} | {time.time()-t0:.0f}s", flush=True)
