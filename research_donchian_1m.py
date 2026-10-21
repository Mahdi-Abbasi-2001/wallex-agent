"""Long-only Donchian breakout on 1m candles (180 days), equal-weight portfolio of 8 markets."""
import numpy as np
import pandas as pd

from data_tf import SYMS, load_1m

CASES = {"zero": 0.0, "maker .10%": 0.0010, "taker+slip .18%": 0.0018}   # per side
BPY = 525600


def positions(c, hi, lo):
    pos = np.zeros(len(c)); p = 0
    for i in range(len(c)):
        if p == 0:
            if c[i] > hi[i]: p = 1
        elif c[i] < lo[i]:
            p = 0
        pos[i] = p
    return pos


raw = {s: load_1m(s) for s in SYMS}
t0, t1 = [pd.to_datetime(max(r.t.iloc[0] for r in raw.values()) if k == 0 else min(r.t.iloc[-1] for r in raw.values()), unit="s") for k in (0, 1)]
print(f"1m data {t0} -> {t1}  ({len(next(iter(raw.values())))} bars/coin)")
rets = {s: r["c"].pct_change().fillna(0).to_numpy() for s, r in raw.items()}
hold = np.mean([r for r in rets.values()], axis=0) if len({len(r) for r in rets.values()}) == 1 else None
if hold is None:
    n = min(len(r) for r in rets.values()); hold = np.mean([r[-n:] for r in rets.values()], axis=0)
    print("(series trimmed to common length)")
eq = np.cumprod(1 + hold)
print(f"equal-weight HOLD: {(eq[-1]-1)*100:.1f}%  maxDD={(1-eq/np.maximum.accumulate(eq)).max()*100:.1f}%\n")
print(f"{'N/M (min)':>11}{'trades/coin/day':>17}{'avg hold (h)':>14}" + "".join(f"{k:>20}" for k in CASES))
for N in (60, 240, 720, 1440):
    M = N // 2; per = {c: [] for c in CASES}; tr = []; hold_bars = []
    for s, r in raw.items():
        c = r["c"]; hi = r["h"].rolling(N).max().shift(1).to_numpy(); lo = r["l"].rolling(M).min().shift(1).to_numpy()
        pos = positions(c.to_numpy(), hi, lo); p1 = np.r_[0, pos[:-1]]
        flips = np.abs(np.diff(np.r_[0, p1])); n = int(flips.sum() / 2)
        tr.append(n / (len(c) / 1440)); hold_bars.append(p1.sum() / max(n, 1))
        for k, cost in CASES.items():
            per[k].append(p1 * rets[s] - flips * cost)
    out = []
    for k in CASES:
        m = min(len(x) for x in per[k]); port = np.mean([x[-m:] for x in per[k]], axis=0)
        e = np.cumprod(1 + port); out.append(f"{(e[-1]-1)*100:+8.1f}% dd{(1-e/np.maximum.accumulate(e)).max()*100:5.1f}%")
    print(f"{N:>5}/{M:<5}{np.mean(tr):17.2f}{np.mean(hold_bars)/60:14.1f}" + "".join(f"{o:>20}" for o in out))
