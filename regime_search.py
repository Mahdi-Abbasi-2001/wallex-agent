"""Regime-switch search (margin: leverage 3, shorting allowed). Train on weeks before 2025-07, test on later weeks, plus the last 8 weeks.
 python regime_search.py [--universe majors|liquid35] [--fees wallex|zero]"""
import argparse
import multiprocessing as mp
import time

import numpy as np
import pandas as pd

from contest.agents import ActiveGuess, Concentrated, HoldAll, RandomHolder, RegimeSwitch, Rotation, SmaTrader, YOLO
from contest.fills import BidAskFill, MidFill
from contest.sim import SimConfig, run_sim
from contest.universes import load_universe

ap = argparse.ArgumentParser(); ap.add_argument("--universe", default="majors"); ap.add_argument("--fees", default="wallex"); ap.add_argument("--step", type=int, default=2); ap.add_argument("--field", type=int, default=30)
args = ap.parse_args(); mkt = load_universe(args.universe); L = 7; WARM = 260; SPLIT = pd.Timestamp("2025-07-01").timestamp(); btc = "BTCUSDT"
starts = list(range(WARM, len(mkt) - L - 1, args.step)); train = np.array([mkt.t[s] < SPLIT for s in starts])
fee = dict(taker=0.0013, maker=0.0010, fill=BidAskFill()) if args.fees == "wallex" else dict(taker=0.0, maker=0.0, fill=MidFill())
cfg = SimConfig(start_cash=1000.0, leverage=3.0, allow_short=True, **fee)


def field_agent(rng):      # same assumed aggressive field as playbook_search
    u = rng.random(); ss = int(rng.integers(1 << 30))
    if u < 0.25: return ActiveGuess(ss, True, 3.0)
    if u < 0.45: return YOLO(3.0, ss, True)
    if u < 0.65: return RandomHolder(int(rng.integers(1, 4)), ss, weight=float(rng.choice([1.0, 2.0, 3.0])))
    if u < 0.80: return Rotation(int(rng.choice([7, 14, 30])), int(rng.integers(1, 4)), rb=L, lev=2.0)
    return SmaTrader(int(rng.integers(2, 6)), int(rng.choice([20, 50])), ss)


GRID = [dict(L=l, k=k, rb=rb, lev=lev, gate_n=gn, down=dn) for l in (7, 14, 30) for k in (1, 2) for rb in (7, 3) for lev in (2.0, 3.0) for gn in (20, 50, 100) for dn in ("short", "flat")]
REF = [("ref: SHORT weakest 14d x3 weekly (playbook)", lambda: Concentrated(L=14, k=1, rb=7, lev=3, short_only=True)), ("ref: long top-1 14d x3 weekly", lambda: Concentrated(L=14, k=1, rb=7, lev=3)),
       ("ref: hold BTC", lambda: HoldAll([btc], name="hold BTC")), ("ref: hold all", lambda: HoldAll(name="hold all"))]


def field_job(_):
    rng = np.random.default_rng(9); return np.array([[run_sim(mkt, field_agent(rng), s, s + L, cfg).ret for _ in range(args.field)] for s in starts])
def job(i):
    mk = (lambda p=GRID[i]: RegimeSwitch(**p)) if i < len(GRID) else REF[i - len(GRID)][1]
    return np.array([run_sim(mkt, mk(), s, s + L, cfg).ret for s in starts])
def stats(r, F):
    srt = np.sort(F, axis=1)[:, ::-1]; return dict(med=np.median(r) * 100, mean=r.mean() * 100, p10=np.percentile(r, 10) * 100, p90=np.percentile(r, 90) * 100, ruin=(r < -.30).mean() * 100, top1=(r > srt[:, 0]).mean() * 100, top3=(r > srt[:, 2]).mean() * 100)
def recent(mk, nw=8):
    T = len(mkt); out = []
    for k in range(nw): end = T - 7 * (nw - 1 - k); out.append(run_sim(mkt, mk(), end - 7, end, cfg).ret * 100)
    return out


if __name__ == "__main__":
    t0 = time.time(); ctx = mp.get_context("fork")
    with ctx.Pool(8) as pool: F = pool.map(field_job, [0])[0]; R = pool.map(job, range(len(GRID) + len(REF)), chunksize=2)
    print(f"universe {args.universe} ({len(mkt.symbols)} coins) | fees {args.fees} | margin x3 | {len(starts)} windows ({train.sum()} train / {(~train).sum()} test) | {len(GRID)} regime configs | {time.time()-t0:.0f}s")
    rows = []
    for i, r in enumerate(R):
        nm = RegimeSwitch(**GRID[i]).name if i < len(GRID) else REF[i - len(GRID)][0]; tr, te = stats(r[train], F[train]), stats(r[~train], F[~train]); rows.append(dict(i=i, name=nm, **{f"tr_{k}": v for k, v in tr.items()}, **{f"te_{k}": v for k, v in te.items()}))
    D = pd.DataFrame(rows); D.to_pickle(f"data/regime_{args.universe}_{args.fees}.pkl")
    G = D[D.i < len(GRID)]; print(f"all {len(G)} regime configs: median test P(top3) {G.te_top3.median():.0f}%, P(#1) {G.te_top1.median():.1f}%, mean {G.te_mean.median():+.1f}%/wk, ruin {G.te_ruin.median():.0f}%  (chance: top3 9.7%, #1 3.2%)")
    print(f"{'config (picked by train P(top3))':76}{'trP3':>5}{'trP1':>6} |{'teP3':>5}{'teP1':>6}{'med%':>7}{'mean%':>7}{'p10%':>7}{'p90%':>7}{'ruin%':>6}")
    for _, r in G.sort_values("tr_top3", ascending=False).head(6).iterrows(): print(f"{r['name']:76}{r.tr_top3:4.0f}%{r.tr_top1:5.1f}% |{r.te_top3:4.0f}%{r.te_top1:5.1f}%{r.te_med:7.1f}{r.te_mean:7.1f}{r.te_p10:7.1f}{r.te_p90:7.1f}{r.te_ruin:6.0f}")
    print("references:")
    for _, r in D[D.i >= len(GRID)].iterrows(): print(f"{r['name']:76}{r.tr_top3:4.0f}%{r.tr_top1:5.1f}% |{r.te_top3:4.0f}%{r.te_top1:5.1f}%{r.te_med:7.1f}{r.te_mean:7.1f}{r.te_p10:7.1f}{r.te_p90:7.1f}{r.te_ruin:6.0f}")
    print("\nlast 8 weeks (consecutive 7-day contests ending 14 Oct), weekly % and compound; top-3 regime configs by train P(top3) vs references:")
    T = len(mkt); heads = [str(pd.to_datetime(mkt.t[T - 7 * (8 - k)], unit="s").date())[5:] for k in range(8)]; print(f"{'':76}" + "".join(f"{h:>7}" for h in heads) + f"{'compound':>10}")
    picks = [(RegimeSwitch(**GRID[int(r.i)]).name, (lambda p=GRID[int(r.i)]: RegimeSwitch(**p))) for _, r in G.sort_values("tr_top3", ascending=False).head(3).iterrows()] + [(n, m) for n, m in REF[:3]]
    for nm, mk in picks:
        w = recent(mk); print(f"{nm:76}" + "".join(f"{x:7.1f}" for x in w) + f"{(np.prod(1 + np.array(w)/100) - 1)*100:10.0f}")
