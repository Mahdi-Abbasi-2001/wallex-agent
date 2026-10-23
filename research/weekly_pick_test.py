"""Weekly 'pick the best holdings for next week' test on Wallex USDT coins (daily Binance data). All selection uses only past data."""
import itertools

import numpy as np
import pandas as pd

meta = pd.read_pickle("data/ALLD_meta.pkl"); coins = meta["ok"]; COST = 0.0018; MINVOL = 3e6; HOLD = 7
raw = {c: pd.read_csv(f"data/ALLD_{c}USDT.csv").set_index("t") for c in coins}
idx = sorted(set().union(*[set(d.index) for d in raw.values()])); P = pd.DataFrame({c: raw[c].c for c in coins}).reindex(idx).to_numpy(); QV = pd.DataFrame({c: raw[c].qv for c in coins}).reindex(idx).to_numpy()
T, N = P.shape; dates = pd.to_datetime(idx, unit="s"); avgqv = pd.DataFrame(QV).rolling(30, min_periods=20).mean().to_numpy(); btc = coins.index("BTC")
print(f"{N} coins, {T} days {dates[0].date()} -> {dates[-1].date()} | liquidity filter: trailing 30d Binance volume >= ${MINVOL/1e6:.0f}M/day | weekly hold, cost {COST*100:.2f}%/side on turnover")
R1 = np.vstack([np.full((1, N), np.nan), P[1:] / P[:-1] - 1])


def run(offset):
    out = {}; rng = np.random.default_rng(7); rebs = list(range(100 + offset, T - HOLD, HOLD)); elig_count = []; prev = {}
    def add(name, w_new, nxt, i):
        w_old = prev.get(name, np.zeros(N)); r = np.nansum(w_new * np.nan_to_num(nxt)) - COST * np.abs(w_new - w_old).sum(); prev[name] = w_new; out.setdefault(name, []).append((i, r, np.abs(w_new - w_old).sum(), (w_new > 0).sum()))
    for i in rebs:
        nxt = P[i + HOLD] / P[i] - 1; ok = ~np.isnan(P[i]) & ~np.isnan(P[i - 84]) & (avgqv[i] >= MINVOL) & ~np.isnan(P[i + HOLD]); e = np.flatnonzero(ok); elig_count.append(len(e))
        if len(e) < 12: continue
        wts = lambda sel: (lambda w: (w.__setitem__(sel, 1.0 / len(sel)), w)[1])(np.zeros(N))
        add("benchmark: hold all eligible equally", wts(e), nxt, i); w = np.zeros(N); w[btc] = 1.0; add("benchmark: BTC", w, nxt, i)
        for L in (1, 2, 4, 8, 12):
            mom = P[i] / P[i - 7 * L] - 1
            order_hi = e[np.argsort(-mom[e])]; order_lo = e[np.argsort(mom[e])]
            for k in (1, 3, 5, 10):
                add(f"top-{k} by past {L}w return", wts(order_hi[:k]), nxt, i)
            if L in (1, 2, 4):
                for k in (3, 5, 10): add(f"BOTTOM-{k} by past {L}w return (reversal)", wts(order_lo[:k]), nxt, i)
        # literal combination search: pool of 15 by 4w return, best 3-subset by past 8w Sharpe of the equal-weight daily portfolio
        mom4 = P[i] / P[i - 28] - 1; pool = e[np.argsort(-mom4[e])][:15]; Rd = np.nan_to_num(R1[i - 55:i + 1][:, pool]); combos = np.array(list(itertools.combinations(range(15), 3)))
        pr = Rd[:, combos].mean(axis=2); sh = pr.mean(0) / (pr.std(0) + 1e-9); best = combos[np.argmax(sh)]; add("best 3-combo from top-15 by past-8w Sharpe", wts(pool[best]), nxt, i)
        add("best 3-combo from top-15 by past-8w return", wts(pool[combos[np.argmax(pr.sum(0))]]), nxt, i)
        # hindsight oracle (NOT tradeable): the best 3 coins of the coming week
        add("ORACLE hindsight best 3 (not tradeable)", wts(e[np.argsort(-nxt[e])][:3]), nxt, i)
        for k in (3, 5):
            for s in range(300):
                sel = rng.choice(e, k, replace=False); r = nxt[sel].mean() - COST * 2 * 1.0   # random picks: assume full turnover each week
                out.setdefault(f"__random{k}_{s}", []).append((i, r, 2.0, k))
    return out, rebs, np.mean(elig_count)


def summarize(rs, hold_idx=None):
    r = np.array([x[1] for x in rs]); eq = np.cumprod(1 + r); sd = r.std()
    return dict(mean=r.mean() * 100, hit=(r > 0).mean() * 100, sharpe=r.mean() / sd * np.sqrt(52) if sd > 0 else 0, t=r.mean() / sd * np.sqrt(len(r)) if sd > 0 else 0, total=(eq[-1] - 1) * 100, mdd=(1 - eq / np.maximum.accumulate(eq)).max() * 100,
                turn=np.mean([x[2] for x in rs]) / 2 * 100, hold=np.mean([x[3] for x in rs]), h1=r[:len(r) // 2].mean() * 100, h2=r[len(r) // 2:].mean() * 100, n=len(r))


out, rebs, ne = run(0); print(f"{len(rebs)} weekly decisions, {ne:.0f} eligible coins per week on average\n")
names = [n for n in out if not n.startswith("__")]
rnd = {k: np.array([np.mean([x[1] for x in out[f'__random{k}_{s}']]) * 100 for s in range(300)]) for k in (3, 5)}; rndsh = {k: np.array([summarize(out[f'__random{k}_{s}'])['sharpe'] for s in range(300)]) for k in (3, 5)}
print(f"{'strategy':48}{'wk mean%':>9}{'hit%':>6}{'t-stat':>7}{'Sharpe':>7}{'total%':>10}{'maxDD%':>7}{'turn%':>6}{'hold#':>6}{'1st half':>9}{'2nd half':>9}{'pctile vs random':>17}")
for nm in sorted(names, key=lambda n: -summarize(out[n])["sharpe"]):
    s = summarize(out[nm]); k = 3 if "top-3" in nm or "3-combo" in nm or "BOTTOM-3" in nm or "best 3" in nm else 5 if "-5 " in nm else None
    pct = f"{(rnd[k] < s['mean']).mean()*100:5.0f}% (k={k})" if k else ""
    print(f"{nm:48}{s['mean']:9.2f}{s['hit']:6.0f}{s['t']:7.1f}{s['sharpe']:7.2f}{s['total']:10.0f}{s['mdd']:7.0f}{s['turn']:6.0f}{s['hold']:6.1f}{s['h1']:9.2f}{s['h2']:9.2f}{pct:>17}")
print(f"\nrandom 3-coin picks (300 paths, same costs): weekly mean {np.mean(rnd[3]):.2f}% (5th-95th pct {np.percentile(rnd[3],5):.2f} to {np.percentile(rnd[3],95):.2f}) | Sharpe {np.mean(rndsh[3]):.2f} ({np.percentile(rndsh[3],5):.2f} to {np.percentile(rndsh[3],95):.2f})")
print(f"random 5-coin picks:                      weekly mean {np.mean(rnd[5]):.2f}% (5th-95th pct {np.percentile(rnd[5],5):.2f} to {np.percentile(rnd[5],95):.2f}) | Sharpe {np.mean(rndsh[5]):.2f} ({np.percentile(rndsh[5],5):.2f} to {np.percentile(rndsh[5],95):.2f})")
import os
print("\nrobustness to the weekday of rebalancing (7 offsets), Sharpe min / median / max:")
rob = {}
for off in (range(7) if not os.environ.get("QUICK") else []):
    o, _, _ = run(off)
    for nm in ("benchmark: hold all eligible equally", "top-1 by past 1w return", "top-3 by past 1w return", "top-3 by past 4w return", "top-5 by past 4w return", "top-3 by past 12w return", "BOTTOM-3 by past 1w return (reversal)", "best 3-combo from top-15 by past-8w Sharpe"):
        rob.setdefault(nm, []).append(summarize(o[nm])["sharpe"])
for nm, v in rob.items(): print(f"  {nm:48} {min(v):+.2f} / {np.median(v):+.2f} / {max(v):+.2f}")
wv = meta["wallex_vol"]; s = [r for r in out["top-3 by past 4w return"]]
print("\nWallex reality check: share of the 155 coins whose CURRENT Wallex 24h USDT volume is under $10k:", f"{np.mean([wv[c] < 1e4 for c in coins])*100:.0f}%", "| under $50k:", f"{np.mean([wv[c] < 5e4 for c in coins])*100:.0f}%", "| median:", f"${np.median([wv[c] for c in coins]):,.0f}")

print("\nDISTRIBUTION of single-week outcomes (net of costs), % return:")
print(f"{'strategy':44}{'median':>7}{'p10':>7}{'p90':>7}{'best':>8}{'worst':>7}{'P(>+10%)':>9}{'P(>+25%)':>9}{'P(<-10%)':>9}")
for nm in ("benchmark: BTC", "benchmark: hold all eligible equally", "top-3 by past 4w return", "top-1 by past 4w return", "top-1 by past 12w return", "best 3-combo from top-15 by past-8w Sharpe", "ORACLE hindsight best 3 (not tradeable)"):
    r = np.array([x[1] for x in out[nm]]) * 100
    print(f"{nm:44}{np.median(r):7.1f}{np.percentile(r,10):7.1f}{np.percentile(r,90):7.1f}{r.max():8.0f}{r.min():7.0f}{(r>10).mean()*100:8.0f}%{(r>25).mean()*100:8.0f}%{(r<-10).mean()*100:8.0f}%")
rr = np.array([[x[1] for x in out[f"__random3_{s}"]] for s in range(300)]).ravel() * 100
print(f"{'random 3-coin picks (pooled)':44}{np.median(rr):7.1f}{np.percentile(rr,10):7.1f}{np.percentile(rr,90):7.1f}{rr.max():8.0f}{rr.min():7.0f}{(rr>10).mean()*100:8.0f}%{(rr>25).mean()*100:8.0f}%{(rr<-10).mean()*100:8.0f}%")
