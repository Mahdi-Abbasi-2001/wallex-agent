"""Score agents over many contest-length windows under different fill models; estimate rank vs a (guessed) field.
 python run_contest.py [--lengths 7 14 30] [--step 4] [--field 40] [--fills mid bidask book] [--cash 1000] [--leverage 1] [--short]"""
import argparse
import time

import numpy as np

import strat_search as S
from contest.agents import Cash, HoldAll, RandomHolder, RandomTrader, Rotation, SmaTrader, TrendEnsemble
from contest.fills import BidAskFill, BookFill, MidFill
from contest.market import Market
from contest.sim import SimConfig, run_sim

ap = argparse.ArgumentParser(); ap.add_argument("--lengths", type=int, nargs="+", default=[7, 14, 30]); ap.add_argument("--step", type=int, default=4); ap.add_argument("--field", type=int, default=40)
ap.add_argument("--fills", nargs="+", default=["mid", "bidask", "book"]); ap.add_argument("--cash", type=float, default=1000.0); ap.add_argument("--leverage", type=float, default=1.0); ap.add_argument("--short", action="store_true")
ap.add_argument("--fee", type=float, default=0.0013); ap.add_argument("--maker", type=float, default=0.0010); args = ap.parse_args()

idx, data = S.load_tf("1d"); mkt = Market(data, 86400); WARM = 260
FILLS = {"mid": (MidFill(), 0.0, 0.0), "bidask": (BidAskFill(), args.fee, args.maker), "book": (BookFill(), args.fee, args.maker)}
AGENTS = {"cash": lambda: Cash(), "hold BTC": lambda: HoldAll(["BTCUSDT"], name="hold BTC"), "hold all 10": lambda: HoldAll(name="hold all 10"), "trend ensemble": lambda: TrendEnsemble(),
          "top-3 momentum 20d": lambda: Rotation(20, 3), "top-1 momentum 20d": lambda: Rotation(20, 1)}
if args.leverage > 1: AGENTS[f"top-1 momentum 20d x{args.leverage:g}"] = lambda: Rotation(20, 1, lev=args.leverage)


def field_agent(rng):
    u = rng.random()                      # ASSUMED composition of the other participants
    if u < 0.40: return RandomHolder(int(rng.integers(1, 4)), int(rng.integers(1 << 30)), weight=float(rng.choice([0.5, 1.0])))
    if u < 0.60: return RandomTrader(int(rng.integers(1, 4)), int(rng.integers(2, 6)), int(rng.integers(1 << 30)))
    if u < 0.80: return SmaTrader(int(rng.integers(2, 6)), int(rng.choice([20, 50, 100])), int(rng.integers(1 << 30)))
    return HoldAll(["BTCUSDT"], weight=float(rng.uniform(0.2, 0.7)))


for fname in args.fills:
    fill, taker, maker = FILLS[fname]; cfg = SimConfig(start_cash=args.cash, taker=taker, maker=maker, fill=fill, leverage=args.leverage, allow_short=args.short)
    for L in args.lengths:
        t0 = time.time(); starts = list(range(WARM, len(mkt) - L - 1, args.step)); res = {k: [] for k in AGENTS}; rank = {k: [] for k in AGENTS}; rank_c = {k: [] for k in AGENTS}; rng = np.random.default_rng(L)
        for s in starts:
            f = [run_sim(mkt, field_agent(rng), s, s + L, cfg) for _ in range(args.field)]; fr = np.array([x.ret for x in f]); fc = np.array([x.calmar for x in f])
            for k, mk in AGENTS.items():
                r = run_sim(mkt, mk(), s, s + L, cfg); res[k].append(r.ret); rank[k].append((fr < r.ret).mean()); rank_c[k].append((fc < r.calmar).mean())
        print(f"\n=== fills: {fname} (taker {taker*100:.2f}%, maker {maker*100:.2f}%) | contest length {L} days | {len(starts)} windows x {args.field} field agents | start cash ${args.cash:g}, leverage {args.leverage:g}{', shorting' if args.short else ''} | {time.time()-t0:.0f}s ===")
        print(f"{'agent':28}{'median%':>8}{'mean%':>7}{'p10%':>7}{'p90%':>7}{'%pos':>6}{'avg pctile':>11}{'P(top10%)':>10}{'P(top25%)':>10}{'P(top10%) by return/DD':>24}")
        for k in AGENTS:
            r = np.array(res[k]) * 100; p = np.array(rank[k]); pc = np.array(rank_c[k])
            print(f"{k:28}{np.median(r):8.2f}{r.mean():7.2f}{np.percentile(r,10):7.1f}{np.percentile(r,90):7.1f}{(r>0).mean()*100:6.0f}{p.mean()*100:11.0f}{(p>=0.9).mean()*100:9.0f}%{(p>=0.75).mean()*100:9.0f}%{(pc>=0.9).mean()*100:23.0f}%")
