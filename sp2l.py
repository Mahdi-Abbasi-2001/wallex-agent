"""SP2L (Spike + 2 Legs) from the video, on Binance 1m. Long+short mirrored.
 spike = breakout candle + 3-candle gap (candle1 tail vs candle3 tail), EMA60 trend filter, limit entry at 3rd-candle tail,
 stop at start of the 3-candle formation, TP = TP_R x risk, optional 2nd entry once 1st trade is >= 0.5R in profit.
 Optional filters from the video: round levels, session opens.  python sp2l.py"""
import itertools
import sys

import numpy as np
import pandas as pd

from binance_data import load

STEP = {"PAXGUSDT": 250.0, "BTCUSDT": 1000.0}           # round-level spacing (video: every 250 points on gold)
COSTS = {"zero": (0, 0, 0), "wallex": (0.0010, 0.0013, 0.0005), "cheap": (0.0002, 0.0004, 0.0001)}   # (maker, taker, slip) per side


def prep(sym):
    d = load(sym); t, o, h, l, c = (d[k].to_numpy() for k in ("t", "o", "h", "l", "c"))
    pc = np.r_[c[0], c[:-1]]; tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    return dict(sym=sym, t=t, o=o, h=h, l=l, c=c, atr=pd.Series(tr).rolling(14).mean().to_numpy(),
                ema=pd.Series(c).ewm(span=60, adjust=False).mean().to_numpy(), hour=(t // 3600) % 24)


def candidates(D, side, tp_r, cancel=30, max_hold=240, offset_atr=0.0, stop_buf=0.1, max_spike_atr=4.0, min_risk=0.0002, max_risk=0.01):
    o, h, l, c, atr, ema, n = D["o"], D["h"], D["l"], D["c"], D["atr"], D["ema"], len(D["c"])
    i = np.arange(0, n - 2); m, e = i + 1, i + 2
    if side == 1: cond = (h[i] < l[e]) & (c[m] > h[i]) & (c[m] > o[m]) & (c[e] > ema[e])
    else: cond = (l[i] > h[e]) & (c[m] < l[i]) & (c[m] < o[m]) & (c[e] < ema[e])
    out = []
    for ee in e[cond & ~np.isnan(atr[e])]:
        ii = ee - 2; A = atr[ee]
        ext = l[ii:ee + 1].min() if side == 1 else h[ii:ee + 1].max()
        entry = (l[ee] + offset_atr * A) if side == 1 else (h[ee] - offset_atr * A)
        stop = ext - stop_buf * A if side == 1 else ext + stop_buf * A
        R = (entry - stop) if side == 1 else (stop - entry)
        if R <= 0 or not (min_risk <= R / entry <= max_risk) or R > max_spike_atr * A: continue
        target = entry + side * tp_r * R
        fill = None
        for j in range(ee + 1, min(n, ee + 1 + cancel)):
            if (side == 1 and h[j] >= target) or (side == -1 and l[j] <= target): break
            if (side == 1 and l[j] <= entry) or (side == -1 and h[j] >= entry): fill = j; break
            if (side == 1 and l[j] <= stop) or (side == -1 and h[j] >= stop): break
        if fill is None: continue
        xi, why, px = None, "time", None
        if (side == 1 and l[fill] <= stop) or (side == -1 and h[fill] >= stop): xi, why, px = fill, "stop", stop
        else:
            end = min(n, fill + 1 + max_hold); hh, ll = h[fill + 1:end], l[fill + 1:end]
            sh, th = (ll <= stop, hh >= target) if side == 1 else (hh >= stop, ll <= target)
            si = sh.argmax() if sh.any() else 10**9; ti = th.argmax() if th.any() else 10**9
            if si == 10**9 and ti == 10**9: xi, px = end - 1, c[end - 1]
            elif si <= ti: xi, why, px = fill + 1 + si, "stop", stop
            else: xi, why, px = fill + 1 + ti, "target", target
        out.append(dict(side=side, sig=ee, fill=fill, exit=xi, entry=entry, stop=stop, target=target, exit_px=px, why=why, R=R, risk=R / entry,
                        ext=ext, hour=D["hour"][ee]))
    return out


def select(trades, D, mode):
    """mode 'one': one position at a time. mode 'two': a 2nd same-side entry allowed once the open one is >= 0.5R in profit."""
    trades = sorted(trades, key=lambda x: x["sig"]); acc = []; open_ = []
    for tr in trades:
        open_ = [k for k in open_ if k["exit"] >= tr["sig"]]
        if not open_: acc.append(tr); open_.append(tr); continue
        if mode == "two" and len(open_) == 1 and open_[0]["side"] == tr["side"]:
            k = open_[0]
            if k["fill"] > tr["sig"]: continue          # first order not filled yet
            seg = slice(k["fill"], tr["sig"] + 1)
            mfe = (D["h"][seg].max() - k["entry"]) if k["side"] == 1 else (k["entry"] - D["l"][seg].min())
            if mfe >= 0.5 * k["R"]: acc.append(tr); open_.append(tr)
    return acc


def net_r(tr, cost):
    mk, tk, sl = cost; s = tr["side"]
    gross = s * (tr["exit_px"] / tr["entry"] - 1)
    fee_out, slip = (mk, 0.0) if tr["why"] == "target" else (tk, sl)
    return (gross - slip - mk - fee_out) / tr["risk"]


def summarize(acc):
    if not acc: return None
    r = {k: np.array([net_r(t, c) for t in acc]) for k, c in COSTS.items()}
    return dict(n=len(acc), win=np.mean([t["exit_px"] * t["side"] > t["entry"] * t["side"] for t in acc]) * 100, risk=np.median([t["risk"] for t in acc]) * 100,
                zero=r["zero"].mean(), wallex=r["wallex"].mean(), cheap=r["cheap"].mean(), tgt=np.mean([t["why"] == "target" for t in acc]) * 100)


def apply_filters(trades, D, levels, session):
    out = []
    for t in trades:
        if session and not ((7 <= t["hour"] < 11) or (13 <= t["hour"] < 17)): continue
        if levels:
            step = STEP[D["sym"]]; lvl = round(t["ext"] / step) * step
            if abs(t["ext"] - lvl) > 0.0007 * t["entry"]: continue
        out.append(t)
    return out


if __name__ == "__main__":
    tp_list = [1.0, 2.0]
    print("columns: n trades | win% | median stop% | avg net R  [zero costs | Wallex costs | cheap costs] | target-hit%")
    for sym in ("PAXGUSDT", "BTCUSDT"):
        D = prep(sym); days = len(D["c"]) / 1440
        for tp in tp_list:
            cand = candidates(D, 1, tp) + candidates(D, -1, tp)
            print(f"\n== {sym} | TP {tp:g}R | {len(cand)} raw setups over {days:.0f} days ==")
            for levels, session, mode in itertools.product((False, True), (False, True), ("one", "two")):
                acc = select(apply_filters(cand, D, levels, session), D, mode); s = summarize(acc)
                tag = f"levels={'Y' if levels else 'n'} session={'Y' if session else 'n'} entries={mode:>3}"
                if s is None: print(f"  {tag}: no trades"); continue
                print(f"  {tag}: n={s['n']:5d} ({s['n']/days:4.1f}/d) win {s['win']:3.0f}% stop {s['risk']:.3f}% | R {s['zero']:+.2f} | {s['wallex']:+.2f} | {s['cheap']:+.2f} | tgt {s['tgt']:.0f}%")
