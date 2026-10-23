"""Weekly-map reaction/break-pullback + 3-candle-gap limit entry, run day by day on Binance 1m.
 python fvg_backtest.py"""
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

sys.argv = [sys.argv[0], "BTCUSDT"]          # fvg_day reads argv at import
import fvg_day as F
from binance_data import load

d = load(); d["day"] = d.t // 86400
days = sorted(d.day.unique())
day_idx = {k: g.reset_index(drop=True) for k, g in d.groupby("day")}


def week_levels(day):
    dt = datetime.fromtimestamp(day * 86400, timezone.utc).date(); mon = dt - timedelta(days=dt.weekday())
    a, b = int(datetime.combine(mon - timedelta(days=7), datetime.min.time(), timezone.utc).timestamp()) // 86400, int(datetime.combine(mon, datetime.min.time(), timezone.utc).timestamp()) // 86400
    wk = d[(d.day >= a) & (d.day < b)]
    if len(wk) < 7 * 1440 * 0.95: return None
    hi, lo = wk.h.max(), wk.l.min()
    return {"W-high": hi, "75%": lo + .75 * (hi - lo), "50%": lo + .5 * (hi - lo), "25%": lo + .25 * (hi - lo), "W-low": lo}


def run_all(label, **params):
    for k, v in params.items(): setattr(F, k, v)
    rows = []
    for day in days:
        lv = week_levels(day); g = day_idx[day]
        if lv is None or len(g) < 1400: continue
        for tr in F.run(g, lv):
            tr["day"] = day; tr["t_fill"] = int(g.t[tr["fill_i"]]); rows.append(tr)
    T = pd.DataFrame(rows); ndays = sum(1 for dd in days if week_levels(dd) is not None)
    return T, ndays


def report(label, T, ndays):
    print(f"\n=== {label} ===")
    if T.empty: print("no trades"); return
    T = T.copy(); T["r_net"] = T.net / T.risk; T["r_gross"] = T.gross / T.risk
    eq = (1 + T.net).cumprod()
    print(f"days tested {ndays} | trades {len(T)} ({len(T)/ndays:.1f}/day) | win rate {(T.net>0).mean()*100:.0f}% | median stop distance {T.risk.median()*100:.3f}%")
    print(f"avg net per trade {T.net.mean()*100:+.3f}%  (before fees {T.gross.mean()*100:+.3f}%)  | avg net R {T.r_net.mean():+.2f}  (gross R {T.r_gross.mean():+.2f})  | planned RR median {T.rr_planned.median():.1f}")
    print(f"compounded all-in: {(eq.iloc[-1]-1)*100:+.1f}%  maxDD {(1-eq/eq.cummax()).max()*100:.1f}%  | exits: {T.why.value_counts().to_dict()}")
    h = len(T) // 2; print(f"first half avg net {T.net.iloc[:h].mean()*100:+.3f}% | second half {T.net.iloc[h:].mean()*100:+.3f}%")
    for col in ("kind", "side", "level"):
        print(f"  by {col}: " + " | ".join(f"{k}: n={len(g)} net {g.net.mean()*100:+.3f}% win {(g.net>0).mean()*100:.0f}%" for k, g in T.groupby(col)))


if __name__ == "__main__":
    base = dict(MIN_GAP=0.0001, MIN_RISK=0.0)
    T, n = run_all("as specified", **base); report("AS SPECIFIED (min gap 0.01% of price, no stop-distance filter)", T, n)
    T.to_pickle("data/fvg_binance_trades.pkl")
    # costs-free view and sensitivity (these deviate from your rules; shown only to see if anything works)
    for lab, p in [("gap >= 0.03%", dict(MIN_GAP=0.0003, MIN_RISK=0.0)), ("gap >= 0.05%", dict(MIN_GAP=0.0005, MIN_RISK=0.0)),
                   ("stop distance >= 0.15%", dict(MIN_GAP=0.0001, MIN_RISK=0.0015)), ("stop >= 0.25% and gap >= 0.03%", dict(MIN_GAP=0.0003, MIN_RISK=0.0025))]:
        T2, n2 = run_all(lab, **p); report(f"VARIANT: {lab}", T2, n2)
