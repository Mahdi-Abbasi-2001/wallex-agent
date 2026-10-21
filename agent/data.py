"""Candle download + CSV cache."""
import time
from pathlib import Path

import pandas as pd

CACHE = Path(__file__).resolve().parent.parent / "data"


def load_candles(client, symbol, resolution="60", days=30, refresh=False):
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{symbol}_{resolution}.csv"
    if f.exists() and not refresh:
        return _clean(pd.read_csv(f))
    end = int(time.time())
    start = end - days * 86400
    rows = []
    step = 86400 * 5  # chunk to stay under response limits
    t = start
    while t < end:
        c = client.candles(symbol, resolution, t, min(t + step, end))
        if c.get("s") == "ok":
            rows += list(zip(c["t"], c["o"], c["h"], c["l"], c["c"], c["v"]))
        t += step
    df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"]).drop_duplicates("t").sort_values("t")
    df = _clean(df)
    df.to_csv(f, index=False)
    return df


def _clean(df):
    df = df.astype({k: float for k in ("o", "h", "l", "c", "v")})
    return df.sort_values("t").reset_index(drop=True)
