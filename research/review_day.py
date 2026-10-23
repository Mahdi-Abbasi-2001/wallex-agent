"""Per-trade review of the weekly-map 3-candle-gap strategy on one local day (PAXGUSDT, Binance 1m, UTC+3:30)."""
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

SYM = "PAXGUSDT"; TZ = 210
sys.argv = ["x", SYM]
import fvg_day as F
from binance_data import load

now = datetime.now(timezone.utc); day = (now + timedelta(minutes=TZ)).date() - timedelta(days=1)
start = int((datetime.combine(day, datetime.min.time(), timezone.utc) - timedelta(minutes=TZ)).timestamp()); end = start + 86400
pre = start - 6 * 3600     # extra history for EMA warm-up
rows = []
for t in range(pre, end, 60000):
    r = requests.get("https://api.binance.com/api/v3/klines", params={"symbol": SYM, "interval": "1m", "startTime": t * 1000, "endTime": (min(t + 60000, end) - 1) * 1000, "limit": 1000}, timeout=20).json(); rows += [x[:6] for x in r]
full = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"]).astype(float); full["t"] = (full.t // 1000).astype("int64"); full = full.drop_duplicates("t").sort_values("t").reset_index(drop=True)
full["ema60"] = full.c.ewm(span=60, adjust=False).mean(); d = full[full.t >= start].reset_index(drop=True)
monday = day - timedelta(days=day.weekday()); w0 = int(datetime.combine(monday - timedelta(days=7), datetime.min.time(), timezone.utc).timestamp()); cc = load(SYM); wk = cc[(cc.t >= w0) & (cc.t < w0 + 7 * 86400)]
hi, lo = wk.h.max(), wk.l.min(); levels = {"W-high": hi, "75%": lo + .75 * (hi - lo), "50%": lo + .5 * (hi - lo), "25%": lo + .25 * (hi - lo), "W-low": lo}
F.MAKER, F.TAKER, F.SLIP = 0.0010, 0.0013, 0.0005; F.TARGET_R = None; F.MIN_GAP = 0.0001; F.MIN_RISK = 0.0; F.MID_MIN_ATR, F.PREV_N, F.PREV_X = 1.0, 5, 1.0
tr = F.run(d, levels)
pc = d.c.shift(); atr = pd.Series(np.maximum(d.h - d.l, np.maximum((d.h - pc).abs(), (d.l - pc).abs()))).rolling(14).mean().to_numpy()
loc = lambda i: datetime.fromtimestamp(int(d.t[i]) + TZ * 60, timezone.utc).strftime("%H:%M")
print(f"{SYM} {day} (UTC+3:30) | filters ON: middle candle >= 1 ATR, formation > previous 5 candles | {len(tr)} trades | price {d.o.iloc[0]:.1f} -> {d.c.iloc[-1]:.1f}\n")
R_ = []
for n, x in enumerate(tr, 1):
    s = x["side"] == "LONG"; sg = 1 if s else -1; i0, e = x["pat_i"], x["pat_i"] + 2; R = abs(x["entry"] - x["stop"])
    seg = d.iloc[x["fill_i"]:x["exit_i"] + 1]
    mfe = ((seg.h.max() - x["entry"]) if s else (x["entry"] - seg.l.min())) / R
    held = d.c[x["touch_i"]:e + 1]; L = x["level_px"]
    worst_close = (held.min() - L) if s else (L - held.max())      # >0 = held the right side, <0 = closed through the level
    ema = full.ema60[full.t == d.t[e]].iloc[0]; with_trend = (d.c[e] > ema) == s
    stop_c = d.iloc[x["exit_i"]]; pierce = ((x["stop"] - stop_c.l) if s else (stop_c.h - x["stop"])) if x["why"] == "stop" else 0
    j = min(len(d) - 1, x["exit_i"] + 5); recovered = (d.c[j] > x["entry"]) if s else (d.c[j] < x["entry"])
    aft30 = sg * (d.c[min(len(d) - 1, x["exit_i"] + 30)] - x["exit"]) / R; aft120 = sg * (d.c[min(len(d) - 1, x["exit_i"] + 120)] - x["exit"]) / R
    R_.append(dict(n=n, side=x["side"], wait=x["fill_i"] - x["order_i"], held=worst_close, mid_atr=(d.h[i0 + 1] - d.l[i0 + 1]) / atr[i0], form=d.h[i0:e + 1].max() - d.l[i0:e + 1].min(), gap=abs(d.l[e] - d.h[i0]) if s else abs(d.l[i0] - d.h[e]),
                   risk=R, risk_atr=R / atr[e], mfe=mfe, dur=x["exit_i"] - x["fill_i"], trend=with_trend, pierce=pierce, recov=recovered, aft30=aft30, aft120=aft120, why=x["why"], net=x["net"] * 100, lvl=x["level"], kind=x["kind"]))
    print(f"#{n} {x['side']:5} {x['kind']:14} @{x['level']:4} {L:,.2f} | touch {loc(x['touch_i'])} gap {loc(i0)}-{loc(e)} fill {loc(x['fill_i'])} @ {x['entry']:,.2f} stop {x['stop']:,.2f} exit {loc(x['exit_i'])} ({x['why']}) net {x['net']*100:+.2f}%")
    print(f"     level-hold: worst close {worst_close:+.2f}$ vs level | gap ${R_[-1]['gap']:.2f}, formation ${R_[-1]['form']:.2f}, middle candle {R_[-1]['mid_atr']:.1f} ATR | order waited {R_[-1]['wait']} min | stop ${R:.2f} = {R_[-1]['risk_atr']:.1f} ATR")
    print(f"     in trade {R_[-1]['dur']} min, best +{mfe:.1f}R | with EMA60 trend: {'yes' if with_trend else 'NO'} | stop pierced by ${pierce:.2f}, recovered within 5 min: {'yes' if recovered else 'no'} | after exit (in trade direction): 30m {aft30:+.1f}R, 2h {aft120:+.1f}R")
T = pd.DataFrame(R_)
print("\nSUMMARY")
print(f"  closed through the level during the setup (worst close < 0): {(T.held < 0).sum()} of {len(T)}")
print(f"  against EMA60 trend: {(~T.trend).sum()} of {len(T)}  (longs {((~T.trend)&(T.side=='LONG')).sum()}/{(T.side=='LONG').sum()}, shorts {((~T.trend)&(T.side=='SHORT')).sum()}/{(T.side=='SHORT').sum()})")
print(f"  order waited > 5 min before filling: {(T.wait > 5).sum()} of {len(T)}")
print(f"  stop distance median ${T.risk.median():.2f} = {T.risk_atr.median():.1f} ATR | median gap ${T.gap.median():.2f}")
print(f"  reached +1R at some point before stop: {(T.mfe >= 1).sum()} of {len(T)} | reached +2R: {(T.mfe >= 2).sum()}")
print(f"  stopped by a wick and recovered within 5 min: {((T.why=='stop')&T.recov).sum()} of {(T.why=='stop').sum()} stops")
print(f"  after exit price continued IN trade direction by 2h: {(T.aft120 > 0).sum()} of {len(T)} | against: {(T.aft120 < 0).sum()}")
