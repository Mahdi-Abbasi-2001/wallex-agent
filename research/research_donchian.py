"""Long-only Donchian breakout, equal-weight portfolio over 8 USDT markets, 3 years of 4h data (+ daily resample).
 enter: close > highest high of last N bars; exit: close < lowest low of last M bars. python research_donchian.py [per_side_cost]"""
import sys

import numpy as np
import pandas as pd

from agent.data import load_candles
from agent.wallex_client import Wallex

SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "TRXUSDT"]
COST = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0013 + 0.0005    # per side: taker fee + slippage


def daily(d):
    g = d.groupby(d["t"] // 86400)
    r = pd.DataFrame({"o": g["o"].first(), "h": g["h"].max(), "l": g["l"].min(), "c": g["c"].last()})
    r.insert(0, "t", r.index * 86400); return r.reset_index(drop=True)


def signal(d, N, M, trend):
    c = d["c"]; hi = d["h"].rolling(N).max().shift(1); lo = d["l"].rolling(M).min().shift(1)
    ok = (c > c.rolling(trend).mean()) if trend else pd.Series(True, index=d.index)
    pos, p = [], 0.0
    for ci, h, l, o in zip(c, hi, lo, ok):
        if p == 0 and ci > h and o: p = 1.0
        elif p == 1 and ci < l: p = 0.0
        pos.append(p)
    return pd.Series(pos, index=d.index)


def sym_returns(d, N, M, trend):
    pos = signal(d, N, M, trend).shift(1).fillna(0)
    r = pos * d["c"].pct_change().fillna(0) - pos.diff().abs().fillna(0) * COST
    return pd.Series(r.to_numpy(), index=d["t"].to_numpy()), int((pos.diff().abs() > 0).sum() / 2)


def stats(r, bpy):
    eq = (1 + r).cumprod(); yrs = len(r) / bpy
    return dict(total=eq.iloc[-1] - 1, sharpe=r.mean() / r.std() * np.sqrt(bpy) if r.std() > 0 else 0, mdd=(1 - eq / eq.cummax()).max())


if __name__ == "__main__":
    w = Wallex(); d4 = {s: load_candles(w, s, "240", 1095) for s in SYMS}
    frames = {"4h": (d4, 2190), "1d": ({s: daily(d) for s, d in d4.items()}, 365)}
    grids = {"4h": [(30, 15), (60, 30), (120, 60), (240, 120)], "1d": [(10, 5), (20, 10), (40, 20), (80, 40)]}
    print(f"cost per side={COST*100:.2f}%  period: {pd.to_datetime(min(d.t.iloc[0] for d in d4.values()), unit='s').date()} -> {pd.to_datetime(max(d.t.iloc[-1] for d in d4.values()), unit='s').date()}")
    for tf, (data, bpy) in frames.items():
        hold = pd.concat({s: pd.Series(d["c"].pct_change().fillna(0).to_numpy(), index=d["t"].to_numpy()) for s, d in data.items()}, axis=1).mean(axis=1)
        hs = stats(hold, bpy)
        print(f"\n=== {tf}: equal-weight HOLD  total={hs['total']*100:6.1f}%  sharpe={hs['sharpe']:5.2f}  maxDD={hs['mdd']*100:5.1f}% ===")
        print(f"{'N/M':>8}{'trend':>7}{'total%':>9}{'sharpe':>8}{'maxDD%':>8}{'trades/sym/yr':>15}   by year (total%)")
        for N, M in grids[tf]:
            for trend in (0, 200 if tf == "4h" else 50):
                res = {s: sym_returns(d, N, M, trend) for s, d in data.items()}
                port = pd.concat({s: r for s, (r, _) in res.items()}, axis=1).mean(axis=1)
                st = stats(port, bpy); tr = np.mean([n for _, n in res.values()]) / 3
                yrs = {int(y): round(((1 + g).prod() - 1) * 100, 0) for y, g in port.groupby(pd.to_datetime(port.index, unit="s").year)}
                print(f"{N:>4}/{M:<3}{trend or '-':>7}{st['total']*100:9.1f}{st['sharpe']:8.2f}{st['mdd']*100:8.1f}{tr:15.1f}   {yrs}")
