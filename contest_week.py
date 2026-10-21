"""One-week, rank-on-profit contest model. python contest_week.py --universe majors|liquid36 --scenario spot|margin3 [--step 2] [--field 30]"""
import argparse
import time

import numpy as np
import pandas as pd

import strat_search as S
from contest.agents import ActiveGuess, Cash, Concentrated, HoldAll, RandomHolder, Rotation, SmaTrader, YOLO
from contest.fills import BidAskFill
from contest.market import Market
from contest.sim import SimConfig, run_sim

ap = argparse.ArgumentParser(); ap.add_argument("--universe", default="majors"); ap.add_argument("--scenario", default="spot"); ap.add_argument("--step", type=int, default=2); ap.add_argument("--field", type=int, default=30); ap.add_argument("--length", type=int, default=7)
args = ap.parse_args()
if args.universe == "majors": mkt = Market(S.load_tf("1d")[1], 86400)
else:
    meta = pd.read_pickle("data/ALLD_meta.pkl"); bars = {}; last = max(int(pd.read_csv(f"data/ALLD_{c}USDT.csv").t.iloc[-1]) for c in meta["ok"])
    for c in meta["ok"]:
        d = pd.read_csv(f"data/ALLD_{c}USDT.csv")
        if int(d.t.iloc[-1]) < last - 3 * 86400: continue          # stopped trading on Binance -> would truncate the common calendar
        if pd.to_datetime(d.t.iloc[0], unit="s") <= pd.Timestamp("2023-09-01") and d.qv.tail(90).mean() >= 5e6:
            cal = np.arange(int(pd.Timestamp("2023-09-01").timestamp()), int(d.t.iloc[-1]) + 1, 86400); dd = d.set_index("t").reindex(cal)
            if dd.c.isna().sum() <= 3: bars[c + "USDT"] = dd.ffill().bfill().reset_index().rename(columns={"index": "t"})[["t", "o", "h", "l", "c", "v"]]
    mkt = Market(bars, 86400)
margin = args.scenario == "margin3"; LEV = 3.0 if margin else 1.0; L = args.length; WARM = 260
cfg = SimConfig(start_cash=1000.0, taker=0.0013, maker=0.0010, fill=BidAskFill(), leverage=LEV, allow_short=margin)
btc = "BTCUSDT"
AG = {"hold BTC": lambda: HoldAll([btc], name="hold BTC"), "top-1 momentum 7d": lambda: Concentrated(7, 1, rb=L), "top-1 momentum 14d": lambda: Concentrated(14, 1, rb=L), "top-1 momentum 30d": lambda: Concentrated(30, 1, rb=L),
      "top-3 momentum 14d": lambda: Concentrated(14, 3, rb=L), "top-1 vol x momentum 14d": lambda: Concentrated(14, 1, rb=L, pick="vol_mom"), "top-1 momentum 14d, BTC>SMA50 gate": lambda: Concentrated(14, 1, rb=L, gate=btc)}
if margin:
    AG.update({"top-1 momentum 14d x3": lambda: Concentrated(14, 1, rb=L, lev=3.0), "top-1 vol x momentum x3": lambda: Concentrated(14, 1, rb=L, lev=3.0, pick="vol_mom"),
               "top-3 momentum x3": lambda: Concentrated(14, 3, rb=L, lev=3.0), "long top-1 / short bottom-1 x3": lambda: Concentrated(14, 1, rb=L, lev=3.0, short_k=1), "cash": lambda: Cash()})


def field_agent(rng):
    u = rng.random(); ss = int(rng.integers(1 << 30))
    if margin:      # ASSUMED aggressive field: leverage and shorting available
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


t0 = time.time(); starts = list(range(WARM, len(mkt) - L - 1, args.step)); rng = np.random.default_rng(11); res = {k: [] for k in AG}; top1 = {k: [] for k in AG}; top3 = {k: [] for k in AG}; pct = {k: [] for k in AG}; fmax = []; fmed = []
for s in starts:
    fr = np.array([run_sim(mkt, field_agent(rng), s, s + args.length, cfg).ret for _ in range(args.field)]); fmax.append(fr.max()); fmed.append(np.median(fr)); srt = np.sort(fr)[::-1]
    for k, mk in AG.items():
        r = run_sim(mkt, mk(), s, s + args.length, cfg).ret; res[k].append(r); top1[k].append(r > srt[0]); top3[k].append(r > srt[2]); pct[k].append((fr < r).mean())
print(f"universe {args.universe} ({len(mkt.symbols)} coins, {pd.to_datetime(mkt.t[WARM], unit='s').date()} -> {pd.to_datetime(mkt.t[-1], unit='s').date()}) | scenario {args.scenario} (leverage {LEV:g}{', shorting' if margin else ''}) | {args.length}-day contests, {len(starts)} windows x {args.field} field agents | fees 0.13%/0.10% + bid/ask | {time.time()-t0:.0f}s")
print(f"FIELD: median participant {np.median(fmed)*100:+.1f}% | average BEST participant {np.mean(fmax)*100:+.1f}% (median best {np.median(fmax)*100:+.1f}%, 90th pct of best {np.percentile(fmax,90)*100:+.1f}%)")
print(f"{'agent':38}{'median%':>8}{'mean%':>7}{'p10%':>7}{'p90%':>7}{'P(>+10%)':>9}{'P(>+25%)':>9}{'P(<-30%)':>9}{'avg pctile':>11}{'P(#1)':>7}{'P(top3)':>8}")
for k in AG:
    r = np.array(res[k]) * 100
    print(f"{k:38}{np.median(r):8.2f}{r.mean():7.2f}{np.percentile(r,10):7.1f}{np.percentile(r,90):7.1f}{(r>10).mean()*100:8.0f}%{(r>25).mean()*100:8.0f}%{(r<-30).mean()*100:8.0f}%{np.mean(pct[k])*100:11.0f}{np.mean(top1[k])*100:6.0f}%{np.mean(top3[k])*100:7.0f}%")
