"""Vectorized strategy research. Signals use only past data (position shifted 1 bar).
python research.py [days] [fee]"""
import sys

import numpy as np
import pandas as pd

from agent.data import load_candles
from agent.features import rsi
from agent.wallex_client import Wallex

SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "TRXUSDT"]
DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 365
FEE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0013    # per side
SLIP = 0.0005
RESM = int(sys.argv[3]) if len(sys.argv) > 3 else 60   # candle minutes
BPY = 525600 / RESM                                     # bars per year


def sma(c, n): return c.rolling(n).mean()


# each returns a target position series in [0,1] using info up to and including bar t
def s_hold(df): return pd.Series(1.0, index=df.index)

def s_sma(df, fast=24, slow=120):
    c = df["c"]; return (sma(c, fast) > sma(c, slow)).astype(float)

def s_donchian(df, n=120, m=48):
    c = df["c"]; hi = df["h"].rolling(n).max().shift(1); lo = df["l"].rolling(m).min().shift(1)
    pos, p = [], 0.0
    for ci, h, l in zip(c, hi, lo):
        if ci > h: p = 1.0
        elif ci < l: p = 0.0
        pos.append(p)
    return pd.Series(pos, index=df.index)

def s_rsi_mr(df, lo=25, hi=55, trend=480):
    c = df["c"]; r = rsi(c, 14); up = c > sma(c, trend)
    pos, p = [], 0.0
    for ri, u in zip(r, up):
        if ri < lo and u: p = 1.0
        elif ri > hi: p = 0.0
        pos.append(p)
    return pd.Series(pos, index=df.index)

def s_volscaled_trend(df, slow=240, target=0.02):
    """Trend filter + position sized inversely to recent volatility (hourly)."""
    c = df["c"]; vol = c.pct_change().rolling(72).std() * np.sqrt(24)   # daily vol
    w = (target / vol).clip(0, 1)
    return (w * (c > sma(c, slow))).fillna(0)

def s_mom(df, n=168, thr=0.0):
    return (df["c"].pct_change(n) > thr).astype(float)

STRATS = {"hold": s_hold, "sma_24/120": s_sma, "sma_48/240": lambda d: s_sma(d, 48, 240),
          "donchian_120/48": s_donchian, "donchian_240/96": lambda d: s_donchian(d, 240, 96),
          "rsi_mr": s_rsi_mr, "volscaled": s_volscaled_trend, "mom_7d": s_mom}


def backtest(df, pos, fee=FEE, slip=SLIP):
    pos = pos.shift(1).fillna(0)                       # trade next bar
    ret = df["c"].pct_change().fillna(0)
    cost = pos.diff().abs().fillna(0) * (fee + slip)
    r = pos * ret - cost
    eq = (1 + r).cumprod()
    return r, eq, int((pos.diff().abs() > 0).sum())


def stats(r, eq, trades):
    yrs = len(r) / BPY
    tot = eq.iloc[-1] - 1
    sharpe = r.mean() / r.std() * np.sqrt(BPY) if r.std() > 0 else 0
    mdd = (1 - eq / eq.cummax()).max()
    return tot, sharpe, mdd, trades / max(yrs, 1e-9)


if __name__ == "__main__":
    w = Wallex(); data = {}
    for s in SYMS:
        try:
            data[s] = load_candles(w, s, str(RESM), DAYS)
        except Exception as e:
            print("skip", s, e)
    print({s: len(d) for s, d in data.items()}, f"fee={FEE} slip={SLIP}")
    for part, sel in [("FULL", slice(None)), ("TRAIN(first 60%)", "train"), ("TEST(last 40%)", "test")]:
        print(f"\n=== {part}  (avg across {len(data)} symbols) ===")
        print(f"{'strategy':18}{'return%':>9}{'sharpe':>8}{'maxDD%':>8}{'trades/yr':>11}{'win syms':>9}")
        for name, f in STRATS.items():
            rows = []
            for s, d in data.items():
                k = int(len(d) * 0.6)
                dd = d if sel == slice(None) else (d.iloc[:k] if sel == "train" else d.iloc[k:]).reset_index(drop=True)
                # signals computed on full history for warm-up, then sliced -> no lookahead
                pos = f(d)
                if sel == "train": pos = pos.iloc[:k]
                elif sel == "test": pos = pos.iloc[k:].reset_index(drop=True)
                r, eq, tr = backtest(dd, pos)
                hold_eq = (dd["c"].iloc[-1] / dd["c"].iloc[0]) - 1
                rows.append(stats(r, eq, tr) + (stats(r, eq, tr)[0] > hold_eq,))
            a = np.array(rows, dtype=float)
            print(f"{name:18}{a[:,0].mean()*100:9.1f}{a[:,1].mean():8.2f}{a[:,2].mean()*100:8.1f}{a[:,3].mean():11.0f}{int(a[:,4].sum()):>6}/{len(rows)}")
