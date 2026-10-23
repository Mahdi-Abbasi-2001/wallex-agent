"""Broad strategy search: families x parameter grids x timeframes x 10 coins (long-only, Binance spot candles, Wallex-style costs).
 Selection on the first 60% of time, judged on the last 40%.   python strat_search.py [tf ...]"""
import itertools
import sys
import time

import numpy as np
import pandas as pd

from binance_multi import SYMS

BPY = {"15m": 35040, "1h": 8760, "4h": 2190, "1d": 365}
COSTS = {"taker": 0.0018, "maker": 0.0010}          # per side: fee (+ slippage for taker)
REBAL = {"15m": 672, "1h": 168, "4h": 42, "1d": 7}  # weekly rebalance for cross-sectional momentum


# ---------- indicators ----------
def sma(x, n): return pd.Series(x).rolling(n).mean().to_numpy()
def ema(x, n): return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()
def sd(x, n): return pd.Series(x).rolling(n).std().to_numpy()
def rsi(c, n):
    s = pd.Series(c); d = s.diff(); up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + up / dn.replace(0, 1e-12))).to_numpy()
def atr(h, l, c, n):
    pc = np.r_[c[0], c[:-1]]; return pd.Series(np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))).rolling(n).mean().to_numpy()
def hold(ent, ext):
    pos = np.zeros(len(ent)); p = 0
    for i in range(len(ent)):
        if p == 0:
            if ent[i]: p = 1
        elif ext[i]: p = 0
        pos[i] = p
    return pos
def rollmax(sig, n): return pd.Series(sig.astype(float)).rolling(n).max().fillna(0).to_numpy()


# ---------- strategy families (each returns position 0/1 decided at bar close) ----------
def f_sma_cross(D, f, s): return (sma(D["c"], f) > sma(D["c"], s)).astype(float)
def f_ema_cross(D, f, s): return (ema(D["c"], f) > ema(D["c"], s)).astype(float)
def f_price_sma(D, n): return (D["c"] > sma(D["c"], n)).astype(float)
def f_donchian(D, N, M):
    hi = pd.Series(D["h"]).rolling(N).max().shift(1).to_numpy(); lo = pd.Series(D["l"]).rolling(M).min().shift(1).to_numpy()
    return hold(D["c"] > hi, D["c"] < lo)
def f_tsmom(D, n): return (D["c"] / pd.Series(D["c"]).shift(n).to_numpy() - 1 > 0).astype(float)
def f_macd(D, a, b, s):
    m = ema(D["c"], a) - ema(D["c"], b); return (m > ema(m, s)).astype(float)
def f_keltner(D, n, k):
    e = ema(D["c"], n); a = atr(D["h"], D["l"], D["c"], 14); return hold(D["c"] > e + k * a, D["c"] < e)
def f_rsi(D, n, lo, hi, trend=False):
    r = rsi(D["c"], n); ent = r < lo
    if trend: ent = ent & (D["c"] > sma(D["c"], 200))
    return hold(ent, r > hi)
def f_bb(D, n, k):
    m = sma(D["c"], n); s = sd(D["c"], n); return hold(D["c"] < m - k * s, D["c"] > m)
def f_squeeze(D, thr):
    c = D["c"]; m = sma(c, 20); s = sd(c, 20); w = pd.Series(4 * s / m); pct = w.rolling(100).rank(pct=True).shift(1).to_numpy()
    return hold((c > m + 2 * s) & (pct <= thr), c < m)
def f_fvg_hold(D, N):
    h, l, o, c = D["h"], D["l"], D["o"], D["c"]; sig = np.zeros(len(c), bool); sig[2:] = (h[:-2] < l[2:]) & (c[1:-1] > o[1:-1]) & (c[2:] > ema(c, 60)[2:]); return rollmax(sig, N)
def f_engulf_hold(D, N):
    o, c = D["o"], D["c"]; sig = np.zeros(len(c), bool); sig[1:] = (c[:-1] < o[:-1]) & (c[1:] > o[1:]) & (c[1:] >= o[:-1]) & (o[1:] <= c[:-1]) & (c[1:] > ema(c, 60)[1:]); return rollmax(sig, N)
def f_inside_hold(D, N):
    h, l, c = D["h"], D["l"], D["c"]; sig = np.zeros(len(c), bool); sig[2:] = (h[1:-1] < h[:-2]) & (l[1:-1] > l[:-2]) & (c[2:] > h[:-2]) & (c[2:] > ema(c, 60)[2:]); return rollmax(sig, N)

FAMILIES = {
    "trend: SMA cross": (f_sma_cross, [dict(f=f, s=s) for f in (5, 10, 20, 50) for s in (20, 50, 100, 200) if f < s]),
    "trend: EMA cross": (f_ema_cross, [dict(f=f, s=s) for f in (5, 10, 20, 50) for s in (20, 50, 100, 200) if f < s]),
    "trend: price > SMA": (f_price_sma, [dict(n=n) for n in (20, 50, 100, 200)]),
    "trend: Donchian breakout": (f_donchian, [dict(N=N, M=N // 2) for N in (10, 20, 40, 80, 160)]),
    "trend: momentum (return>0)": (f_tsmom, [dict(n=n) for n in (10, 20, 40, 80, 160, 320)]),
    "trend: MACD": (f_macd, [dict(a=a, b=b, s=s) for a, b, s in ((12, 26, 9), (24, 52, 18), (6, 13, 5), (48, 104, 36))]),
    "trend: Keltner breakout": (f_keltner, [dict(n=n, k=k) for n in (20, 50) for k in (1, 1.5, 2)]),
    "revert: RSI oversold": (f_rsi, [dict(n=n, lo=lo, hi=hi) for n in (2, 5, 14) for lo in (10, 20, 30) for hi in (50, 70)]),
    "revert: RSI oversold + uptrend": (lambda D, **p: f_rsi(D, trend=True, **p), [dict(n=n, lo=lo, hi=hi) for n in (2, 5, 14) for lo in (10, 20, 30) for hi in (50, 70)]),
    "revert: Bollinger lower band": (f_bb, [dict(n=n, k=k) for n in (20, 50) for k in (1.5, 2, 2.5)]),
    "volatility: squeeze breakout": (f_squeeze, [dict(thr=t) for t in (0.1, 0.2, 0.3)]),
    "price action: bullish gap (FVG) hold": (f_fvg_hold, [dict(N=N) for N in (2, 4, 8, 16)]),
    "price action: engulfing hold": (f_engulf_hold, [dict(N=N) for N in (2, 4, 8, 16)]),
    "price action: inside-bar break hold": (f_inside_hold, [dict(N=N) for N in (2, 4, 8, 16)]),
}
XS_GRID = [dict(L=L, k=k) for L in (10, 20, 40, 80, 160) for k in (3, 5)]


# ---------- data + backtest ----------
def load_tf(tf):
    raw = {s: pd.read_csv(f"data/MULTI_{s}_{tf}.csv") for s in SYMS}
    idx = sorted(set.intersection(*[set(d.t) for d in raw.values()]))
    data = {s: d.set_index("t").loc[idx].reset_index() for s, d in raw.items()}
    return idx, data


def asset_returns(D, pos, cost):
    r = np.r_[0.0, D["c"][1:] / D["c"][:-1] - 1]; p1 = np.r_[0.0, pos[:-1]]; p2 = np.r_[0.0, p1[:-1]]
    return p1 * r - np.abs(p1 - p2) * cost


def xs_returns(C, L, k, rb, cost):
    n, m = C.shape; ret = np.r_[np.zeros((1, m)), C[1:] / C[:-1] - 1]; mom = C / np.vstack([np.full((L, m), np.nan), C[:-L]]) - 1
    W = np.zeros((n, m)); cur = np.zeros(m)
    for i in range(n):
        if i % rb == 0 and not np.isnan(mom[i]).all():
            order = np.argsort(-np.nan_to_num(mom[i], nan=-9)); sel = [j for j in order[:k] if mom[i, j] > 0]
            cur = np.zeros(m); cur[sel] = 1.0 / k
        W[i] = cur
    W1 = np.vstack([np.zeros((1, m)), W[:-1]]); W2 = np.vstack([np.zeros((1, m)), W1[:-1]])
    return (W1 * ret).sum(1) - np.abs(W1 - W2).sum(1) * cost, W1.sum(1)


def stats(r, bpy):
    r = np.nan_to_num(r); eq = np.cumprod(1 + r); sd_ = r.std()
    return dict(ret=eq[-1] - 1, sharpe=(r.mean() / sd_ * np.sqrt(bpy)) if sd_ > 0 else 0.0, mdd=(1 - eq / np.maximum.accumulate(eq)).max())


def run_tf(tf, cost_name="taker"):
    t0 = time.time(); idx, data = load_tf(tf); n = len(idx); cut = int(n * 0.6); bpy = BPY[tf]; cost = COSTS[cost_name]; yrs = n / bpy
    print(f"\n######## {tf}: {n} bars, {len(SYMS)} coins, {pd.to_datetime(idx[0], unit='s').date()} -> {pd.to_datetime(idx[-1], unit='s').date()} | train until {pd.to_datetime(idx[cut], unit='s').date()} ########", flush=True)
    Dd = {s: {k: d[k].to_numpy() for k in ("o", "h", "l", "c")} for s, d in data.items()}; C = np.column_stack([Dd[s]["c"] for s in SYMS])
    rows = []
    bench = np.mean([np.r_[0.0, Dd[s]["c"][1:] / Dd[s]["c"][:-1] - 1] for s in SYMS], axis=0)
    rows.append(dict(tf=tf, family="BENCHMARK: hold all 10 equally", params={}, trades_yr=0, expo=1.0, tr=stats(bench[:cut], bpy), te=stats(bench[cut:], bpy)))
    for fam, (fn, grid) in FAMILIES.items():
        for p in grid:
            rs = []; tr_ = []; ex_ = []
            for s in SYMS:
                pos = fn(Dd[s], **p); rs.append(asset_returns(Dd[s], pos, cost)); tr_.append(((np.diff(pos) > 0).sum()) / yrs); ex_.append(pos.mean())
            port = np.mean(rs, axis=0)
            rows.append(dict(tf=tf, family=fam, params=p, trades_yr=np.mean(tr_), expo=np.mean(ex_), tr=stats(port[:cut], bpy), te=stats(port[cut:], bpy), port=port))
    for p in XS_GRID:
        port, expo = xs_returns(C, p["L"], p["k"], REBAL[tf], cost)
        rows.append(dict(tf=tf, family="rotation: top-k momentum (weekly)", params=p, trades_yr=0, expo=expo.mean(), tr=stats(port[:cut], bpy), te=stats(port[cut:], bpy), port=port))
    print(f"  ({len(rows)} configs in {time.time()-t0:.0f}s)", flush=True)
    return rows, idx, cut


def report(all_rows):
    out = []
    for tf in sorted({r["tf"] for r in all_rows}, key=list(BPY).index):
        rows = [r for r in all_rows if r["tf"] == tf]; bench = next(r for r in rows if r["family"].startswith("BENCH"))
        print(f"\n=== {tf}: best config per family (picked on TRAIN Sharpe, judged on TEST) | benchmark hold-all: test Sharpe {bench['te']['sharpe']:+.2f}, return {bench['te']['ret']*100:+.0f}%, maxDD {bench['te']['mdd']*100:.0f}% ===")
        print(f"{'family':38}{'#cfg':>5} {'picked params':26}{'trainSh':>8}{'testSh':>8}{'testRet%':>9}{'testDD%':>8}{'trades/yr':>10}{'expo%':>6}{'median cfg testSh':>18}{'%cfg test>0':>12}")
        for fam in dict.fromkeys(r["family"] for r in rows if not r["family"].startswith("BENCH")):
            g = [r for r in rows if r["family"] == fam]; b = max(g, key=lambda r: r["tr"]["sharpe"])
            med = np.median([r["te"]["sharpe"] for r in g]); pos = np.mean([r["te"]["sharpe"] > 0 for r in g]) * 100
            ps = ",".join(f"{k}={v}" for k, v in b["params"].items())
            print(f"{fam:38}{len(g):5d} {ps:26}{b['tr']['sharpe']:8.2f}{b['te']['sharpe']:8.2f}{b['te']['ret']*100:9.1f}{b['te']['mdd']*100:8.1f}{b['trades_yr']:10.1f}{b['expo']*100:6.0f}{med:18.2f}{pos:11.0f}%")
            out.append(dict(tf=tf, family=fam, params=b["params"], train_sh=b["tr"]["sharpe"], test_sh=b["te"]["sharpe"], test_ret=b["te"]["ret"], test_dd=b["te"]["mdd"], trades_yr=b["trades_yr"], med=med, bench_sh=bench["te"]["sharpe"], bench_ret=bench["te"]["ret"]))
    L = pd.DataFrame(out).sort_values("test_sh", ascending=False)
    print("\n=== LEADERBOARD (train-selected configs, ranked by TEST Sharpe) ===")
    for _, r in L.head(12).iterrows():
        print(f"  {r.tf:>3} {r.family:38} {','.join(f'{k}={v}' for k, v in r.params.items()):24} testSharpe {r.test_sh:+.2f} (hold-all {r.bench_sh:+.2f}) | test return {r.test_ret*100:+6.1f}% (hold-all {r.bench_ret*100:+.0f}%) | maxDD {r.test_dd*100:.0f}% | {r.trades_yr:.0f} trades/yr/coin")
    return L


if __name__ == "__main__":
    tfs = sys.argv[1:] or ["1d", "4h", "1h", "15m"]; all_rows = []
    for tf in tfs: all_rows += run_tf(tf)[0]
    L = report(all_rows); pd.to_pickle([{k: v for k, v in r.items()} for r in all_rows], "data/strat_search_rows.pkl"); L.to_pickle("data/strat_search_leaderboard.pkl")
