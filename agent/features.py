"""Compact market summary for strategies / the LLM (keeps tokens low)."""
import pandas as pd


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, 1e-12))


def summarize(df):
    """df: candles up to and including 'now'. Returns a dict of plain numbers."""
    c = df["c"]
    last = float(c.iloc[-1])
    out = {"price": last}
    for n in (1, 6, 24):
        if len(c) > n:
            out[f"ret_{n}"] = round(last / float(c.iloc[-1 - n]) - 1, 4)
    if len(c) >= 50:
        out["sma_10"] = round(float(c.rolling(10).mean().iloc[-1]), 6)
        out["sma_50"] = round(float(c.rolling(50).mean().iloc[-1]), 6)
        out["rsi_14"] = round(float(rsi(c).iloc[-1]), 1)
        rng = (df["h"] - df["l"]).rolling(14).mean().iloc[-1]
        out["atr_pct"] = round(float(rng) / last, 4)
    return out
