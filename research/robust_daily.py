"""Robustness of daily trend following (long-only, 10 coins, Binance, Wallex-style costs)."""
import numpy as np
import pandas as pd

import strat_search as S

tf = "1d"; idx, data = S.load_tf(tf); n = len(idx); cut = int(n * 0.6); bpy = S.BPY[tf]
Dd = {s: {k: d[k].to_numpy() for k in ("o", "h", "l", "c")} for s, d in data.items()}; years = pd.to_datetime(idx, unit="s").year.to_numpy()
print(f"daily: {n} bars {pd.to_datetime(idx[0], unit='s').date()} -> {pd.to_datetime(idx[-1], unit='s').date()} | test from {pd.to_datetime(idx[cut], unit='s').date()}")
hold_ret = np.mean([np.r_[0.0, Dd[s]["c"][1:] / Dd[s]["c"][:-1] - 1] for s in S.SYMS], axis=0)

def port(fn, p, cost): return np.mean([S.asset_returns(Dd[s], fn(Dd[s], **p), cost) for s in S.SYMS], axis=0)
def yr(r): return {int(y): round(((1 + r[years == y]).prod() - 1) * 100) for y in sorted(set(years))}
def line(name, r):
    a, b, f = S.stats(r[:cut], bpy), S.stats(r[cut:], bpy), S.stats(r, bpy)
    tt = r[cut:].mean() / (r[cut:].std() / np.sqrt(len(r[cut:])) + 1e-12)
    print(f"  {name:34} train Sh {a['sharpe']:+.2f} | test Sh {b['sharpe']:+.2f} (t={tt:+.1f}) ret {b['ret']*100:+6.1f}% DD {b['mdd']*100:4.1f}% | full Sh {f['sharpe']:+.2f} ret {f['ret']*100:+7.1f}% DD {f['mdd']*100:4.1f}%")
    return b

print("\n== 1. the four most-picked daily rules + hold, taker costs (0.18%/side) ==")
cands = [("MACD 24/52/18", S.f_macd, dict(a=24, b=52, s=18)), ("price > SMA50", S.f_price_sma, dict(n=50)), ("SMA cross 10/50", S.f_sma_cross, dict(f=10, s=50)), ("momentum 40d>0", S.f_tsmom, dict(n=40)), ("Donchian 40/20", S.f_donchian, dict(N=40, M=20))]
line("HOLD all 10 (equal weight)", hold_ret)
for nm, fn, p in cands:
    r = port(fn, p, S.COSTS["taker"]); line(nm, r); print(f"      by calendar year %: {yr(r)}   (hold: {yr(hold_ret)})" if nm.startswith("MACD") else f"      by calendar year %: {yr(r)}")

print("\n== 2. cost sensitivity (same rules): full-sample Sharpe at maker 0.10% vs taker 0.18% vs zero ==")
for nm, fn, p in cands:
    print(f"  {nm:20}", " | ".join(f"{k} {S.stats(port(fn, p, c), bpy)['sharpe']:+.2f}" for k, c in (("zero", 0.0), ("maker .10%", .001), ("taker .18%", .0018))))

print("\n== 3. per coin, test period only (MACD 24/52/18 and price>SMA50, taker costs): return % vs hold ==")
for nm, fn, p in cands[:2]:
    row = {}
    for s in S.SYMS:
        r = S.asset_returns(Dd[s], fn(Dd[s], **p), S.COSTS["taker"])[cut:]; h = Dd[s]["c"][-1] / Dd[s]["c"][cut] - 1; row[s[:-4]] = f"{((1+r).prod()-1)*100:+.0f}/{h*100:+.0f}"
    print(f"  {nm:16} (strategy/hold):", row)

print("\n== 4. a-priori ensemble: equal vote of 15 generic trend rules (position = share of rules that are long), taker costs ==")
rules = [(S.f_sma_cross, dict(f=10, s=50)), (S.f_sma_cross, dict(f=20, s=100)), (S.f_sma_cross, dict(f=50, s=200)), (S.f_price_sma, dict(n=50)), (S.f_price_sma, dict(n=100)), (S.f_price_sma, dict(n=200)),
         (S.f_tsmom, dict(n=40)), (S.f_tsmom, dict(n=80)), (S.f_tsmom, dict(n=160)), (S.f_macd, dict(a=12, b=26, s=9)), (S.f_macd, dict(a=24, b=52, s=18)), (S.f_macd, dict(a=48, b=104, s=36)),
         (S.f_donchian, dict(N=20, M=10)), (S.f_donchian, dict(N=40, M=20)), (S.f_donchian, dict(N=80, M=40))]
def ens(cost): return np.mean([S.asset_returns(Dd[s], np.mean([fn(Dd[s], **p) for fn, p in rules], axis=0), cost) for s in S.SYMS], axis=0)
for cn in ("taker", "maker"):
    r = ens(S.COSTS[cn]); b = line(f"ensemble ({cn})", r)
    if cn == "taker": print(f"      by calendar year %: {yr(r)}"); expo = np.mean([np.mean([fn(Dd[s], **p) for fn, p in rules], axis=0).mean() for s in S.SYMS]); print(f"      average exposure {expo*100:.0f}% of capital")
print("\n== 5. how often did single daily trend configs beat hold-all in the TEST period? (119 configs incl. non-trend) ==")
rows = [r for r in S.run_tf("1d")[0] if r["family"].startswith("trend")]; hold_te = S.stats(hold_ret[cut:], bpy)["sharpe"]
print(f"  trend configs: {len(rows)} | test Sharpe > 0: {np.mean([r['te']['sharpe']>0 for r in rows])*100:.0f}% | > hold ({hold_te:+.2f}): {np.mean([r['te']['sharpe']>hold_te for r in rows])*100:.0f}% | median test Sharpe {np.median([r['te']['sharpe'] for r in rows]):+.2f} | median TRAIN Sharpe {np.median([r['tr']['sharpe'] for r in rows]):+.2f}")
