"""SP2L spike logic on higher timeframes (5m/15m resampled from Binance 1m, 1h/4h native Binance klines, 3 years)."""
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import sp2l as S
from data_tf import resample

URL = "https://api.binance.com/api/v3/klines"


def load_native(sym, interval, days):
    f = Path("data") / f"BINANCE_{sym}_{interval}.csv"
    if f.exists(): return pd.read_csv(f)
    ms = {"1h": 3600, "4h": 14400}[interval]; end = int(time.time()) // ms * ms; start = end - days * 86400; rows = []
    t = start
    while t < end:
        r = requests.get(URL, params={"symbol": sym, "interval": interval, "startTime": t * 1000, "limit": 1000}, timeout=20).json()
        if not r: break
        rows += [x[:6] for x in r]; t = int(r[-1][0] // 1000) + ms; time.sleep(0.1)
    d = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"]).astype(float); d["t"] = (d.t // 1000).astype("int64")
    d = d.drop_duplicates("t").sort_values("t"); d = d[d.t < end].reset_index(drop=True); d.to_csv(f, index=False); return d


def prep_df(sym, d):
    t, o, h, l, c = (d[k].to_numpy() for k in ("t", "o", "h", "l", "c")); pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    return dict(sym=sym, t=t, o=o, h=h, l=l, c=c, atr=pd.Series(tr).rolling(14).mean().to_numpy(), ema=pd.Series(c).ewm(span=60, adjust=False).mean().to_numpy(), hour=(t // 3600) % 24)


CFG = {"5m": dict(cancel=30, max_hold=240, max_risk=0.02), "15m": dict(cancel=30, max_hold=240, max_risk=0.03),
       "1h": dict(cancel=12, max_hold=72, max_risk=0.06), "4h": dict(cancel=12, max_hold=60, max_risk=0.10)}

if __name__ == "__main__":
    print("n trades | win% | median stop% | avg net R [zero | Wallex | cheap] | H1/H2 zero-cost R (first/second half of the period)")
    for sym in ("PAXGUSDT", "BTCUSDT"):
        base = pd.read_csv(f"data/BINANCE_{sym}_1m.csv")
        frames = {"5m": resample(base, 5), "15m": resample(base, 15), "1h": load_native(sym, "1h", 1095), "4h": load_native(sym, "4h", 1095)}
        for tf, df in frames.items():
            D = S.prep_like = prep_df(sym, df); days = len(df) * {"5m": 5, "15m": 15, "1h": 60, "4h": 240}[tf] / 1440
            c = CFG[tf]
            for tp in (1.0, 2.0):
                cand = S.candidates(D, 1, tp, **c) + S.candidates(D, -1, tp, **c)
                print(f"\n== {sym} {tf} | TP {tp:g}R | {len(df)} bars ({days:.0f} days) | {len(cand)} raw setups ==")
                for levels, session in itertools.product((False, True), (False, True) if tf in ("5m", "15m") else (False,)):
                    acc = S.select(S.apply_filters(cand, D, levels, session), D, "one"); s = S.summarize(acc)
                    tag = f"levels={'Y' if levels else 'n'} session={'Y' if session else 'n'}"
                    if s is None or s["n"] < 5: print(f"  {tag}: {0 if s is None else s['n']} trades"); continue
                    half = D["t"][0] + (D["t"][-1] - D["t"][0]) / 2
                    h1 = [t for t in acc if D["t"][t["sig"]] < half]; h2 = [t for t in acc if D["t"][t["sig"]] >= half]
                    z = lambda x: np.mean([S.net_r(t, S.COSTS["zero"]) for t in x]) if x else float("nan")
                    print(f"  {tag}: n={s['n']:5d} ({s['n']/days:5.2f}/d) win {s['win']:3.0f}% stop {s['risk']:.2f}% | R {s['zero']:+.2f} | {s['wallex']:+.2f} | {s['cheap']:+.2f} | H1 {z(h1):+.2f} H2 {z(h2):+.2f}")
