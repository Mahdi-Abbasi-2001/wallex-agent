"""Outside-Wallex market data for the AI: Binance (the global reference market; Wallex prices follow it with a lag) and the USDT/TMN rate.
Per market it adds: Binance returns (15m/1h/24h), the same-window Wallex 15m return, and the Wallex premium vs Binance converted at USDT/TMN.
LiveExternal fetches on demand; HistExternal (bot/replay.py) serves the same fields from cached history, never past the simulated time."""
import time

import numpy as np
import pandas as pd
import requests

BASES = ("https://data-api.binance.vision", "https://api.binance.com")      # public market-data mirror first (often reachable where api.binance.com is not)
NO_BINANCE = {"USOON", "COPXON", "SLVON"}                                    # tokenised commodities with no Binance spot pair
ALIAS = {"XAUT": "PAXG"}                                                     # gold proxy on Binance


def bn_symbol(market):
    base = market[:-3]
    if base in NO_BINANCE: return None
    return ALIAS.get(base, base) + "USDT"


def klines(sym, interval, start=None, end=None, limit=1000):
    p = {"symbol": sym, "interval": interval, "limit": limit}
    if start: p["startTime"] = int(start * 1000)
    if end: p["endTime"] = int(end * 1000)
    for b in BASES:
        try:
            r = requests.get(b + "/api/v3/klines", params=p, timeout=10)
            if r.status_code == 200:
                d = r.json()
                return pd.DataFrame({"t": [int(x[0]) // 1000 for x in d], "c": [float(x[4]) for x in d], "qv": [float(x[7]) for x in d]})
        except requests.RequestException:
            continue
    return None


def _ret(t, c, now, secs):
    """Return over the last `secs` using closes of bars that finished by `now` (t = bar open time, 1m bars)."""
    if t is None or len(t) == 0: return None
    i = np.searchsorted(t, now - 60, side="right") - 1; j = np.searchsorted(t, now - 60 - secs, side="right") - 1
    if i < 0 or j < 0 or c[j] <= 0: return None
    return round(float(c[i] / c[j] - 1), 4)


def build(now, markets, bn1m, wx1m, usdt_1m):
    """bn1m/wx1m: {market: (t, c)} 1-minute series; usdt_1m: (t, c) USDTTMN. Returns ({market: fields}, usdt_context)."""
    out = {}
    ut, uc = usdt_1m if usdt_1m else (None, None)
    iu = np.searchsorted(ut, now - 60, side="right") - 1 if ut is not None else -1
    usdt = float(uc[iu]) if iu >= 0 else None
    for m in markets:
        b = bn1m.get(m); w = wx1m.get(m)
        if not b: continue
        bt, bc = b; ib = np.searchsorted(bt, now - 60, side="right") - 1
        f = {"bn_ret_15m": _ret(bt, bc, now, 900), "bn_ret_1h": _ret(bt, bc, now, 3600), "bn_ret_24h": _ret(bt, bc, now, 86400)}
        if w:
            wt, wc = w; iw = np.searchsorted(wt, now - 60, side="right") - 1
            f["wx_ret_15m"] = _ret(wt, wc, now, 900)
            if usdt and ib >= 0 and iw >= 0 and m[:-3] not in ALIAS:
                f["premium_vs_binance"] = round(float(wc[iw] / (bc[ib] * usdt) - 1), 4)
        out[m] = f
    ctx = {"usdt_tmn": usdt, "usdt_tmn_ret_24h": _ret(ut, uc, now, 86400) if ut is not None else None}
    return out, ctx


class LiveExternal:
    def __init__(self, w): self.w = w
    def get(self, now, markets):
        bn, wx = {}, {}
        for m in markets:
            s = bn_symbol(m)
            if s:
                d = klines(s, "1m", limit=120); hrs = klines(s, "1h", limit=27)
                if d is not None and len(d):
                    if hrs is not None and len(hrs):                         # older history as hourly closes, stamped at their closing minute
                        hh = hrs[hrs.t + 3600 <= d.t.iloc[0]]; d = pd.concat([pd.DataFrame({"t": hh.t + 3540, "c": hh.c}), d[["t", "c"]]])
                    bn[m] = (d.t.to_numpy(dtype=int), d.c.to_numpy(dtype=float))
            try:
                c = self.w.candles(m, "1", int(now) - 1800, int(now))
                if c.get("s") == "ok": wx[m] = (np.array(c["t"], dtype=int), np.array(c["c"], dtype=float))
            except Exception: pass
        try:
            c = self.w.candles("USDTTMN", "1", int(now) - 86400 - 600, int(now)); u = (np.array(c["t"], dtype=int), np.array(c["c"], dtype=float))
        except Exception: u = None
        return build(now, markets, bn, wx, u)
