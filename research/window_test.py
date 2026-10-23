"""Window-by-window test: how would daily strategies have done over contest-length windows (start in cash, pay entry costs)?"""
import numpy as np
import pandas as pd

import strat_search as S

tf = "1d"; idx, data = S.load_tf(tf); n = len(idx); C = S.COSTS["taker"]
Dd = {s: {k: d[k].to_numpy() for k in ("o", "h", "l", "c")} for s, d in data.items()}; closes = np.column_stack([Dd[s]["c"] for s in S.SYMS])
ret_assets = np.column_stack([np.r_[0.0, Dd[s]["c"][1:] / Dd[s]["c"][:-1] - 1] for s in S.SYMS])
dates = pd.to_datetime(idx, unit="s")

def per_coin(fn, p):
    pos = np.column_stack([fn(Dd[s], **p) for s in S.SYMS]); r = np.column_stack([S.asset_returns(Dd[s], pos[:, j], C) for j, s in enumerate(S.SYMS)])
    return r.mean(1), pos.mean(1)

def ens():
    rules = [(S.f_sma_cross, dict(f=10, s=50)), (S.f_sma_cross, dict(f=20, s=100)), (S.f_sma_cross, dict(f=50, s=200)), (S.f_price_sma, dict(n=50)), (S.f_price_sma, dict(n=100)), (S.f_price_sma, dict(n=200)),
             (S.f_tsmom, dict(n=40)), (S.f_tsmom, dict(n=80)), (S.f_tsmom, dict(n=160)), (S.f_macd, dict(a=12, b=26, s=9)), (S.f_macd, dict(a=24, b=52, s=18)), (S.f_macd, dict(a=48, b=104, s=36)),
             (S.f_donchian, dict(N=20, M=10)), (S.f_donchian, dict(N=40, M=20)), (S.f_donchian, dict(N=80, M=40))]
    pos = np.column_stack([np.mean([fn(Dd[s], **p) for fn, p in rules], axis=0) for s in S.SYMS])
    r = np.column_stack([S.asset_returns(Dd[s], pos[:, j], C) for j, s in enumerate(S.SYMS)]); return r.mean(1), pos.mean(1)

def rotation(L, k, absolute, rb=7):
    m = closes.shape[1]; mom = closes / np.vstack([np.full((L, m), np.nan), closes[:-L]]) - 1; W = np.zeros((n, m)); cur = np.zeros(m)
    for i in range(n):
        if i % rb == 0 and not np.isnan(mom[i]).all():
            order = np.argsort(-np.nan_to_num(mom[i], nan=-9)); sel = [j for j in order[:k] if (mom[i, j] > 0 or not absolute)]; cur = np.zeros(m); cur[sel] = 1.0 / k
        W[i] = cur
    W1 = np.vstack([np.zeros((1, m)), W[:-1]]); W2 = np.vstack([np.zeros((1, m)), W1[:-1]])
    return (W1 * ret_assets).sum(1) - np.abs(W1 - W2).sum(1) * C, W1.sum(1)

strats = {"cash": (np.zeros(n), np.zeros(n)), "HOLD all 10 (equal)": (ret_assets.mean(1), np.ones(n)), "HOLD BTC": (ret_assets[:, 0], np.ones(n)),
          "trend ensemble (15 rules)": ens(), "MACD 24/52/18": per_coin(S.f_macd, dict(a=24, b=52, s=18)), "price > SMA50": per_coin(S.f_price_sma, dict(n=50)),
          "top-3 momentum 20d, always in": rotation(20, 3, False), "top-1 momentum 20d, always in": rotation(20, 1, False), "top-3 momentum, cash if <0": rotation(20, 3, True)}
field = [r for r in S.run_tf("1d")[0] if not r["family"].startswith("BENCH")]; F = np.column_stack([r["port"] for r in field]); Fexpo = np.array([r["expo"] for r in field])
lr = {k: np.concatenate([[0], np.cumsum(np.log1p(v[0][1:]))]) for k, v in strats.items()}; lF = np.vstack([np.zeros((1, F.shape[1])), np.cumsum(np.log1p(F[1:]), axis=0)])
START0 = 250   # after the 200-day indicators have warmed up

def windows(L, first):
    out = {k: [] for k in strats}; pct = {k: [] for k in strats}
    for s in range(max(START0, first), n - L):
        fr = np.exp(lF[s + L] - lF[s]) * (1 - C * Fexpo) - 1
        for k, (r, ex) in strats.items():
            w = np.exp(lr[k][s + L] - lr[k][s]) * (1 - C * ex[s - 1]) - 1; out[k].append(w); pct[k].append((fr < w).mean())
    return {k: np.array(v) for k, v in out.items()}, {k: np.array(v) for k, v in pct.items()}

for label, first in (("ALL windows since indicators warmed up", 0), ("LAST 12 MONTHS only", n - 365 - 30)):
    print(f"\n################ {label}: {dates[max(START0, first)].date()} -> {dates[-1].date()} ################")
    for L in (3, 7, 14, 30):
        W, P = windows(L, first); hold = W["HOLD all 10 (equal)"]; nw = len(hold)
        print(f"\n--- {L}-day windows ({nw} overlapping windows) | return % over the window (starting in cash, incl. entry costs) | rank = percentile among the 119-setting field ---")
        print(f"{'strategy':32}{'median':>8}{'mean':>7}{'p10':>7}{'p90':>7}{'worst':>7}{'best':>7}{'%pos':>6}{'%>hold':>7}{'avg rank':>9}{'top10%':>8}{'top25%':>8}")
        for k in strats:
            w = W[k] * 100; p = P[k]
            print(f"{k:32}{np.median(w):8.2f}{w.mean():7.2f}{np.percentile(w,10):7.1f}{np.percentile(w,90):7.1f}{w.min():7.1f}{w.max():7.1f}{(w>0).mean()*100:6.0f}{(W[k]>hold).mean()*100:7.0f}{p.mean()*100:9.0f}{(p>=0.9).mean()*100:8.0f}{(p>=0.75).mean()*100:8.0f}")
