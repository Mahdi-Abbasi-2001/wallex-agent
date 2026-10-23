"""Scenario playbook search: for each of (spot|margin) x (hold|trade-daily) find agents that work, selected on early windows and checked on later ones.
 python playbook_search.py --universe majors|liquid35 --fees wallex|zero"""
import argparse
import multiprocessing as mp
import time

import numpy as np
import pandas as pd

from contest.agents import ActiveGuess, Concentrated, HoldAll, RandomHolder, Rotation, SmaTrader, YOLO
from contest.fills import BidAskFill, MidFill
from contest.sim import SimConfig, run_sim
from contest.universes import load_universe

ap = argparse.ArgumentParser(); ap.add_argument("--universe", default="majors"); ap.add_argument("--fees", default="wallex"); ap.add_argument("--step", type=int, default=2); ap.add_argument("--field", type=int, default=30); ap.add_argument("--length", type=int, default=7)
args = ap.parse_args()
mkt = load_universe(args.universe); L = args.length; WARM = 260; btc = "BTCUSDT"; SPLIT = pd.Timestamp("2025-07-01").timestamp()
starts = list(range(WARM, len(mkt) - L - 1, args.step)); train = np.array([mkt.t[s] < SPLIT for s in starts])
fee = dict(taker=0.0013, maker=0.0010, fill=BidAskFill()) if args.fees == "wallex" else dict(taker=0.0, maker=0.0, fill=MidFill())
CFG = {"spot": SimConfig(start_cash=1000.0, leverage=1.0, allow_short=False, **fee), "margin": SimConfig(start_cash=1000.0, leverage=3.0, allow_short=True, **fee)}


def field_agent(rng, margin):
    u = rng.random(); ss = int(rng.integers(1 << 30))
    if margin:
        if u < 0.25: return ActiveGuess(ss, True, 3.0)
        if u < 0.45: return YOLO(3.0, ss, True)
        if u < 0.65: return RandomHolder(int(rng.integers(1, 4)), ss, weight=float(rng.choice([1.0, 2.0, 3.0])))
        if u < 0.80: return Rotation(int(rng.choice([7, 14, 30])), int(rng.integers(1, 4)), rb=L, lev=2.0)
        return SmaTrader(int(rng.integers(2, 6)), int(rng.choice([20, 50])), ss)
    if u < 0.30: return RandomHolder(int(rng.integers(1, 4)), ss)
    if u < 0.55: return ActiveGuess(ss, False, 1.0)
    if u < 0.75: return SmaTrader(int(rng.integers(2, 6)), int(rng.choice([20, 50])), ss)
    if u < 0.90: return Rotation(int(rng.choice([7, 14, 30])), int(rng.integers(1, 4)), rb=L)
    return HoldAll([btc], weight=float(rng.uniform(0.3, 1.0)))


def grid(cell):
    g = []
    if cell == ("spot", "hold"):
        for l in (7, 14, 30):
            for k in (1, 2, 3):
                for gate in (None, btc): g.append(dict(L=l, k=k, rb=L, gate=gate))
            g.append(dict(L=l, k=1, rb=L, stop=0.15))
    elif cell == ("margin", "hold"):
        for l in (7, 14, 30):
            for k in (1, 2):
                for lev in (2, 3):
                    g.append(dict(L=l, k=k, rb=L, lev=lev))                              # long only, leveraged
                    g.append(dict(L=l, k=k, rb=L, lev=lev, short_k=k))                   # long top-k + short bottom-k (leverage split per side)
                    g.append(dict(L=l, k=k, rb=L, lev=lev, short_only=True))             # short the weakest only
            g.append(dict(L=l, k=1, rb=L, lev=3, short_k=1, stop=0.15)); g.append(dict(L=l, k=1, rb=L, lev=3, stop=0.15))
    elif cell == ("spot", "trade"):
        for l in (1, 3, 7):
            for k in (1, 2):
                for rb in (1, 2, 3): g.append(dict(L=l, k=k, rb=rb))
            for rb in (1, 2, 3): g.append(dict(L=l, k=1, rb=rb, stop=0.05))
    else:
        for l in (1, 3, 7):
            for k in (1, 2):
                for rb in (1, 2, 3):
                    g.append(dict(L=l, k=k, rb=rb, lev=3)); g.append(dict(L=l, k=k, rb=rb, lev=3, short_k=k)); g.append(dict(L=l, k=k, rb=rb, lev=3, short_only=True)); g.append(dict(L=l, k=k, rb=rb, lev=2, short_k=k))
            for rb in (1, 2, 3): g.append(dict(L=l, k=1, rb=rb, lev=3, short_k=1, stop=0.05))
    return g


CELLS = [("spot", "hold"), ("spot", "trade"), ("margin", "hold"), ("margin", "trade")]; JOBS = [(cell, p) for cell in CELLS for p in grid(cell)]


def field_job(scn):
    cfg = CFG[scn]; rng = np.random.default_rng(5 if scn == "spot" else 6)
    return np.array([[run_sim(mkt, field_agent(rng, scn == "margin"), s, s + L, cfg).ret for _ in range(args.field)] for s in starts])


def cfg_job(i):
    (scn, kind), p = JOBS[i]; cfg = CFG[scn]
    return np.array([run_sim(mkt, Concentrated(**p), s, s + L, cfg).ret for s in starts])


def stats(r, F):
    srt = np.sort(F, axis=1)[:, ::-1]; pct = (F < r[:, None]).mean(1)
    return dict(med=np.median(r) * 100, mean=r.mean() * 100, p10=np.percentile(r, 10) * 100, p90=np.percentile(r, 90) * 100, g25=(r > .25).mean() * 100, ruin=(r < -.30).mean() * 100,
                top1=(r > srt[:, 0]).mean() * 100, top3=(r > srt[:, 2]).mean() * 100, pct=pct.mean() * 100, sr=r.mean() / (r.std() + 1e-9))


if __name__ == "__main__":
    t0 = time.time(); ctx = mp.get_context("fork")
    with ctx.Pool(8) as pool:
        Fs = dict(zip(("spot", "margin"), pool.map(field_job, ["spot", "margin"]))); rets = pool.map(cfg_job, range(len(JOBS)), chunksize=2)
    print(f"universe {args.universe} ({len(mkt.symbols)} coins) | fees {args.fees} | {L}-day contests | {len(starts)} windows ({train.sum()} train <2025-07, {(~train).sum()} test) x {args.field} field agents | {len(JOBS)} configs | {time.time()-t0:.0f}s")
    out = []
    for (scn, kind), p in JOBS: pass
    for i, ((scn, kind), p) in enumerate(JOBS):
        F, r = Fs[scn], rets[i]; tr, te = stats(r[train], F[train]), stats(r[~train], F[~train])
        out.append(dict(cell=f"{scn}/{kind}", params=p, name=Concentrated(**p).name, **{f"tr_{k}": v for k, v in tr.items()}, **{f"te_{k}": v for k, v in te.items()}))
    D = pd.DataFrame(out); D.to_pickle(f"data/playbook_{args.universe}_{args.fees}.pkl")
    for scn, kind in CELLS:
        d = D[D.cell == f"{scn}/{kind}"]; fm = np.median(Fs[scn][~train].max(1)) * 100
        print(f"\n===== {scn.upper()} / {kind.upper()} | {len(d)} configs | field median best (test windows) {fm:+.0f}% | all-config median test P(top3) {d.te_top3.median():.0f}%, P(#1) {d.te_top1.median():.1f}% =====")
        print(f"{'config':44}{'trP(top3)':>10}{'tr P(#1)':>9} | {'teP(top3)':>10}{'te P(#1)':>9}{'te med%':>8}{'te p10%':>8}{'te p90%':>8}{'te ruin%':>9}{'te mean%':>9}")
        for _, r in d.sort_values("tr_top3", ascending=False).head(5).iterrows():
            print(f"{r['name']:44}{r.tr_top3:9.0f}%{r.tr_top1:8.1f}% | {r.te_top3:9.0f}%{r.te_top1:8.1f}%{r.te_med:8.1f}{r.te_p10:8.1f}{r.te_p90:8.1f}{r.te_ruin:9.0f}{r.te_mean:9.1f}")
        print("  risk-aware picks (best train mean/std):")
        for _, r in d.sort_values("tr_sr", ascending=False).head(3).iterrows():
            print(f"{r['name']:44}{r.tr_top3:9.0f}%{r.tr_top1:8.1f}% | {r.te_top3:9.0f}%{r.te_top1:8.1f}%{r.te_med:8.1f}{r.te_p10:8.1f}{r.te_p90:8.1f}{r.te_ruin:9.0f}{r.te_mean:9.1f}")
