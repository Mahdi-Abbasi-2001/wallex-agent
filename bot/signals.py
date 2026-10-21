"""Deterministic signal engine: features per market + BTC regime + ranked candidates (this is what the LLM chooses from)."""
import time

import numpy as np
import pandas as pd

from agent.wallex_client import Wallex


class Candles:
    def __init__(self, w: Wallex, universe, refresh=900):
        self.w, self.universe, self.refresh, self.data, self.t = w, list(universe), refresh, {}, 0

    def update(self, now=None):
        now = now or time.time()
        if now - self.t < self.refresh and self.data: return self.data
        for s in self.universe:
            try:
                c = self.w.candles(s, "60", int(now) - 120 * 3600, int(now))
                if c.get("s") == "ok": self.data[s] = pd.DataFrame({k: np.array(c[k], dtype=float) for k in "tohlcv"})
            except Exception: pass
        self.t = now
        return self.data


def features(df):
    c, h, l = df["c"], df["h"], df["l"]
    if len(c) < 60: return None
    last = float(c.iloc[-1]); atr = float((h - l).rolling(14).mean().iloc[-1]) / last
    f = {"price": last, "ret_6h": last / float(c.iloc[-7]) - 1, "ret_24h": last / float(c.iloc[-25]) - 1, "ret_72h": last / float(c.iloc[-73]) - 1 if len(c) > 73 else None,
         "sma20_gap": last / float(c.rolling(20).mean().iloc[-1]) - 1, "sma50_gap": last / float(c.rolling(50).mean().iloc[-1]) - 1, "atr_1h": atr}
    return {k: (round(v, 4) if v is not None else None) for k, v in f.items()}


def regime(feats):
    b = feats.get("BTCTMN")
    if not b: return "unknown"
    if b["ret_24h"] < -0.01 and b["sma20_gap"] < 0: return "down"
    if b["ret_24h"] > 0.01 and b["sma20_gap"] > 0: return "up"
    return "range"


def candidates(feats, quotes, cfg, min_score=1.0, ext=None):
    """Spot long-only: strongest relative momentum above its trend; nothing (cash) in a down regime."""
    reg = regime(feats)
    rows = [(s, f) for s, f in feats.items() if f and s in quotes]
    if len(rows) < 4: return reg, []
    b = feats.get("BTCTMN")
    if cfg.regime_gate == "strict" and not (b and b["sma50_gap"] > 0 and b["ret_24h"] > 0): return reg, []
    mom = {s: 0.5 * f["ret_24h"] + 0.5 * (f["ret_72h"] if f["ret_72h"] is not None else f["ret_24h"]) for s, f in rows}
    mu, sd = np.mean(list(mom.values())), np.std(list(mom.values())) or 1e-9
    out = []
    for s, f in rows:
        bid, ask = quotes[s]; spread = (ask - bid) / ((ask + bid) / 2)
        if spread > 0.005: continue
        z = (mom[s] - mu) / sd
        stop = float(np.clip(1.8 * f["atr_1h"] * 2, cfg.min_stop_pct, cfg.max_stop_pct))   # ~1.8 x 2h ATR
        e = (ext or {}).get(s, {})
        if cfg.entry_mode == "pullback":      # uptrend (above 50h avg, up over 72h) that dipped back to its 20h average in the last hours
            ok = reg in ("up", "range") and f["sma50_gap"] > 0 and (f["ret_72h"] or 0) > 0 and -0.015 <= f["sma20_gap"] <= 0.003 and f["ret_6h"] < 0
        elif cfg.entry_mode == "lead":        # Binance already moved up in the last 15m and Wallex has not followed yet
            lead = (e.get("bn_ret_15m") or 0) - (e.get("wx_ret_15m") or 0)
            ok = reg != "down" and lead >= 0.006 and (e.get("bn_ret_1h") or 0) > 0; z = lead * 100 if ok else z
        else:                                 # momentum: strongest relative momentum above its trend
            ok = z >= min_score and f["sma20_gap"] > 0 and f["sma50_gap"] > -0.02 and reg in ("up", "range")
        for side, ok in (("long", ok),):   # spot: long only, cash in a down regime
            if ok: out.append({"market": s, "side": side, "z": round(abs(z), 2), "entry": ask if side == "long" else bid, "stop_pct": round(stop, 4),
                               "take_pct": round(stop * cfg.tp_mult, 4), "spread": round(spread, 4), **f})
    out.sort(key=lambda r: -r["z"])
    return reg, out[:5]
