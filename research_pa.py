"""Price-action strategy suite across timeframes. Costs applied after the fact on gross trade returns.
python research_pa.py [tp_r]"""
import sys

import numpy as np
import pandas as pd

from data_tf import SYMS, load_1m, resample

TFS = [1, 5, 15, 60, 240]
TP_R = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
MAX_HOLD, MIN_R, MAX_R = 96, 0.0008, 0.03
COSTS = {"zero": 0.0, "maker": 0.0020, "taker": 2 * 0.0013 + 2 * 0.0005}   # round-trip, fraction of notional


class F:  # feature bundle
    def __init__(s, df):
        s.t = df["t"].to_numpy(); s.o, s.h, s.l, s.c = (df[k].to_numpy() for k in "ohlc")
        c = df["c"]; pc = c.shift(1)
        tr = np.maximum(df["h"] - df["l"], np.maximum((df["h"] - pc).abs(), (df["l"] - pc).abs()))
        s.A = tr.rolling(14).mean().to_numpy()
        s.e20, s.e50, s.e200 = (c.ewm(span=n, adjust=False).mean().to_numpy() for n in (20, 50, 200))
        day = s.t // 86400; s.day = day; s.hour = (s.t // 3600) % 24
        dh = pd.Series(s.h).groupby(day).max(); dl = pd.Series(s.l).groupby(day).min()
        s.pdh = pd.Series(day - 1).map(dh).to_numpy(); s.pdl = pd.Series(day - 1).map(dl).to_numpy()
        s.rng = s.h - s.l + 1e-12
        sh = lambda a, k=1: np.r_[np.full(k, np.nan), a[:-k]]
        s.sh = sh


def roll_any(mask, k):  # any True in previous k bars (excluding current)
    return pd.Series(mask.astype(float)).rolling(k).max().shift(1).fillna(0).to_numpy() > 0


# each strategy returns (mask_long, stop_long, mask_short, stop_short)
def pdh_retest(f, K=12):
    tol = 0.5 * f.A
    L = roll_any(f.c > f.pdh, K) & (f.l <= f.pdh + tol) & (f.c > f.pdh) & (f.c > f.o)
    S = roll_any(f.c < f.pdl, K) & (f.h >= f.pdl - tol) & (f.c < f.pdl) & (f.c < f.o)
    return L, np.minimum(f.l, f.pdh) - 0.3 * f.A, S, np.maximum(f.h, f.pdl) + 0.3 * f.A

def sweep_pd(f):
    L = (f.l < f.pdl) & (f.c > f.pdl) & (f.c > f.o)
    S = (f.h > f.pdh) & (f.c < f.pdh) & (f.c < f.o)
    return L, f.l - 0.2 * f.A, S, f.h + 0.2 * f.A

def sweep_n(f, N=48):
    lo = pd.Series(f.l).rolling(N).min().shift(1).to_numpy(); hi = pd.Series(f.h).rolling(N).max().shift(1).to_numpy()
    L = (f.l < lo) & (f.c > lo) & (f.c > f.o)
    S = (f.h > hi) & (f.c < hi) & (f.c < f.o)
    return L, f.l - 0.2 * f.A, S, f.h + 0.2 * f.A

def pin_ema(f):
    lw = (np.minimum(f.o, f.c) - f.l) / f.rng; uw = (f.h - np.maximum(f.o, f.c)) / f.rng
    L = (f.e50 > f.e200) & (f.l <= f.e50 + 0.25 * f.A) & (f.c > f.e50) & (lw >= 0.6) & ((f.c - f.l) / f.rng >= 0.66)
    S = (f.e50 < f.e200) & (f.h >= f.e50 - 0.25 * f.A) & (f.c < f.e50) & (uw >= 0.6) & ((f.h - f.c) / f.rng >= 0.66)
    return L, f.l - 0.2 * f.A, S, f.h + 0.2 * f.A

def engulf_ema(f):
    o1, c1, l1, h1 = f.sh(f.o), f.sh(f.c), f.sh(f.l), f.sh(f.h)
    L = (f.e20 > f.e50) & (f.e50 > f.e200) & (c1 < o1) & (f.c > f.o) & (f.c >= o1) & (f.o <= c1) & (f.l <= f.e20 + 0.5 * f.A)
    S = (f.e20 < f.e50) & (f.e50 < f.e200) & (c1 > o1) & (f.c < f.o) & (f.c <= o1) & (f.o >= c1) & (f.h >= f.e20 - 0.5 * f.A)
    return L, np.minimum(f.l, l1) - 0.2 * f.A, S, np.maximum(f.h, h1) + 0.2 * f.A

def inside_bar(f):
    h1, l1, h2, l2 = f.sh(f.h), f.sh(f.l), f.sh(f.h, 2), f.sh(f.l, 2)
    inside = (h1 < h2) & (l1 > l2)
    L = inside & (f.e50 > f.e200) & (f.c > h2) & (f.c > f.o)
    S = inside & (f.e50 < f.e200) & (f.c < l2) & (f.c < f.o)
    return L, l2, S, h2

def donchian(f, N=48):
    hi = pd.Series(f.h).rolling(N).max().shift(1).to_numpy(); lo = pd.Series(f.l).rolling(N).min().shift(1).to_numpy()
    L = (f.c > hi) & (f.e50 > f.e200); S = (f.c < lo) & (f.e50 < f.e200)
    return L, f.c - 1.5 * f.A, S, f.c + 1.5 * f.A

def asian_break(f):
    asia = (f.hour < 8)
    hi = pd.Series(np.where(asia, f.h, np.nan)).groupby(f.day).transform("max").to_numpy()
    lo = pd.Series(np.where(asia, f.l, np.nan)).groupby(f.day).transform("min").to_numpy()
    act = (f.hour >= 8) & (f.hour < 20)
    cl = act & (f.c > hi) & (f.sh(f.c) <= hi); cs = act & (f.c < lo) & (f.sh(f.c) >= lo)
    first = lambda m: pd.Series(m.astype(int)).groupby(f.day).cumsum().to_numpy() == 1
    mid = (hi + lo) / 2
    return cl & first(cl), mid, cs & first(cs), mid

STRATS = {"pd_level_retest": pdh_retest, "pd_sweep": sweep_pd, "n_bar_sweep": sweep_n, "pin_bar_ema50": pin_ema,
          "engulf_ema20": engulf_ema, "inside_bar_brk": inside_bar, "donchian_48": donchian, "asian_range_brk": asian_break}


def simulate(f, L, sl, S, ss):
    ok = lambda m, st: m & ~np.isnan(st) & ~np.isnan(f.A)
    il = np.flatnonzero(ok(L, sl)); is_ = np.flatnonzero(ok(S, ss))
    sig = sorted([(t, 1, sl[t]) for t in il] + [(t, -1, ss[t]) for t in is_])
    n = len(f.c); last = -1; out = []
    for t, side, stop in sig:
        if t <= last or t >= n - 2:
            continue
        e = f.c[t]; R = (e - stop) if side == 1 else (stop - e)
        if R <= 0: continue
        rp = R / e
        if rp < MIN_R or rp > MAX_R: continue
        tgt = e + side * TP_R * R
        end = min(n, t + 1 + MAX_HOLD); hh, ll = f.h[t + 1:end], f.l[t + 1:end]
        sh_, th_ = (ll <= stop, hh >= tgt) if side == 1 else (hh >= stop, ll <= tgt)
        si = sh_.argmax() if sh_.any() else 10**9; ti = th_.argmax() if th_.any() else 10**9
        if si == 10**9 and ti == 10**9: px, j = f.c[end - 1], end - 1
        elif si <= ti: px, j = stop, t + 1 + si
        else: px, j = tgt, t + 1 + ti
        out.append((f.t[t], side, side * (px / e - 1), rp)); last = j
    return out


if __name__ == "__main__":
    raw = {s: load_1m(s) for s in SYMS}
    allrows = []
    for tf in TFS:
        data = {s: resample(raw[s], tf) for s in SYMS}
        feats = {s: F(d) for s, d in data.items()}
        for name, fn in STRATS.items():
            tr = []
            for s, f in feats.items():
                for r in simulate(f, *fn(f)):
                    tr.append((s,) + r)
            if not tr: continue
            T = pd.DataFrame(tr, columns=["sym", "t", "side", "gross", "risk"]); T["tf"] = tf; T["strat"] = name
            allrows.append(T)
    A = pd.concat(allrows)
    A.to_pickle("data/pa_trades.pkl")
    mid = A.t.median()
    print(f"tp={TP_R}R max_hold={MAX_HOLD} bars | round-trip costs: " + ", ".join(f"{k}={v*100:.2f}%" for k, v in COSTS.items()))
    print(f"{'tf':>4} {'strategy':16}{'n':>7}{'win%':>6}{'gR':>7}{'maker R':>8}{'taker R':>8}{'risk%':>7}{'H1 gR':>7}{'H2 gR':>7}")
    for (tf, st), g in A.groupby(["tf", "strat"], sort=False):
        if len(g) < 100: continue
        R = lambda c, x=g: ((x.gross - c) / x.risk).mean()
        h1, h2 = g[g.t < mid], g[g.t >= mid]
        print(f"{tf:>4} {st:16}{len(g):7d}{(g.gross>0).mean()*100:6.0f}{R(0):7.2f}{R(COSTS['maker']):8.2f}{R(COSTS['taker']):8.2f}"
              f"{g.risk.median()*100:7.2f}{((h1.gross/h1.risk).mean() if len(h1) else float('nan')):7.2f}{((h2.gross/h2.risk).mean() if len(h2) else float('nan')):7.2f}")
