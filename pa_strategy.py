"""Price-action strategy (long-only, hourly candles).
 1. Bias (last 7d): close > SMA(168) and 7d return > 0.
 2. State (last ~24-36h): breakout / range / channel.
 3. Entry on a confirmed reaction at a support level (round, weekly map, swing clusters).
 4. Exit: stop below level, target at next level >=1.5R (else 2R), 96-bar time stop.
 python pa_strategy.py [fee] [slippage]"""
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from agent.data import load_candles
from agent.wallex_client import Wallex

SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "TRXUSDT"]


@dataclass
class P:
    fee: float = 0.0013
    slip: float = 0.0005
    bias_n: int = 168
    er_range: float = 0.30        # efficiency ratio below this = range
    range_win: int = 36
    min_range_pct: float = 0.015
    brk_win: int = 48             # prior bars defining the breakout level
    brk_recent: int = 12          # breakout must have happened within these bars
    sections: int = 5             # weekly map: high/low split into 5 equal sections
    swing_k: int = 6
    swing_win: int = 720          # one month
    swing_touch: int = 3
    swing_tol: float = 0.004
    conf_close_pos: float = 0.6   # close in upper 60% of bar range
    stop_atr: float = 0.3
    min_r_pct: float = 0.003
    max_r_pct: float = 0.05
    tp_r: float = 1.5
    max_hold: int = 96
    bpd: int = 24                 # bars per day
    warm: int = 800
    er_n: int = 24
    week_min: int = 100
    use_range: bool = False       # range setups disabled
    dbl: str = 'strict'           # 'strict' = double bottom/top required, 'prefer' = tag if present, 'off'
    dbl_win: int = 72
    dbl_gap: int = 6
    dbl_bounce: float = 0.008


def atr(h, l, c, n=14):
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    return pd.Series(tr).rolling(n).mean().to_numpy()


def round_levels(price, lo, hi):
    step = 10 ** round(np.log10(price * 0.01))
    a = np.floor(lo / step) * step
    return np.arange(a, hi + step, step)


def swing_pivots(h, l, k):
    hm = pd.Series(h).rolling(2 * k + 1, center=True).max().to_numpy()
    lm = pd.Series(l).rolling(2 * k + 1, center=True).min().to_numpy()
    idx_h = np.where(h == hm)[0]; idx_l = np.where(l == lm)[0]
    idx = np.concatenate([idx_h, idx_l]); px = np.concatenate([h[idx_h], l[idx_l]])
    o = np.argsort(idx)
    return idx[o], px[o]


def cluster_levels(idx, px, t, p):
    """Levels with >= swing_touch pivots in the past month, using pivots confirmed by bar t."""
    m = (idx + p.swing_k <= t) & (idx >= t - p.swing_win)
    pr = np.sort(px[m]); out = []
    i = 0
    while i < len(pr):
        j = i
        while j + 1 < len(pr) and pr[j + 1] <= pr[i] * (1 + p.swing_tol):
            j += 1
        if j - i + 1 >= p.swing_touch:
            out.append(pr[i:j + 1].mean())
        i = j + 1
    return np.array(out)


def run(df, p=P(), side=1):
    """side=+1 long (bull rules), side=-1 short (exact mirror: bear bias, resistance, rallies)."""
    t_s = df["t"].to_numpy(); o = df["o"].to_numpy(); h = df["h"].to_numpy()
    l = df["l"].to_numpy(); c = df["c"].to_numpy()
    n = len(c); A = atr(h, l, c)
    sma = pd.Series(c).rolling(p.bias_n).mean().to_numpy()
    wk = ((t_s // 86400) + 3) // 7
    wdf = pd.DataFrame({"wk": wk, "h": h, "l": l}).groupby("wk").agg(h=("h", "max"), l=("l", "min"), n=("h", "size"))
    pidx, ppx = swing_pivots(h, l, p.swing_k)
    # mirrored views: for shorts, "low" becomes the high, so the same comparisons apply with signs flipped
    sg = side
    touch = l if sg == 1 else h            # the extreme that probes the level
    trades, t, swing_cache, swing_day = [], p.warm, np.array([]), -1
    while t < n - 2:
        day = t // p.bpd
        if day != swing_day:
            swing_cache, swing_day = cluster_levels(pidx, ppx, t, p), day
        bias = (c[t] - sma[t]) * sg > 0 and (c[t] / c[t - p.bias_n] - 1) * sg > 0
        if not bias or np.isnan(A[t]):
            t += 1; continue
        tol = max(0.5 * A[t], 0.0015 * c[t])
        rng = h[t] - l[t] + 1e-12
        if sg == 1:
            confirm = c[t] > o[t] and (c[t] - l[t]) / rng >= p.conf_close_pos
        else:
            confirm = c[t] < o[t] and (h[t] - c[t]) / rng >= p.conf_close_pos
        if not confirm:
            t += 1; continue
        lv = [round_levels(c[t], c[t] * 0.85, c[t] * 1.1), swing_cache]
        w = wk[t] - 1
        if w in wdf.index and wdf.loc[w, "n"] >= p.week_min:
            lo_, hi_ = wdf.loc[w, "l"], wdf.loc[w, "h"]
            lv.append(lo_ + (hi_ - lo_) * np.arange(p.sections + 1) / p.sections)
        levels = np.concatenate([x for x in lv if len(x)])
        seg = c[t - p.er_n:t + 1]
        er24 = abs(seg[-1] - seg[0]) / (np.abs(np.diff(seg)).sum() + 1e-12)
        seg36 = c[t - p.range_win:t + 1]
        er36 = abs(seg36[-1] - seg36[0]) / (np.abs(np.diff(seg36)).sum() + 1e-12)
        setup, level = None, None
        if sg == 1:
            B = h[t - p.brk_win - p.brk_recent:t - p.brk_recent].max()
            if (c[t - p.brk_recent:t] > B).any() and abs(l[t] - B) <= tol and c[t] > B:
                setup, level = "breakout", B
        else:
            B = l[t - p.brk_win - p.brk_recent:t - p.brk_recent].min()
            if (c[t - p.brk_recent:t] < B).any() and abs(h[t] - B) <= tol and c[t] < B:
                setup, level = "breakout", B
        if setup is None and p.use_range and er36 < p.er_range:
            RL = l[t - p.range_win:t].min(); RH = h[t - p.range_win:t].max()
            if (RH - RL) / RL >= p.min_range_pct:
                if sg == 1 and abs(l[t] - RL) <= tol and c[t] > RL:
                    setup, level = "range", RL
                elif sg == -1 and abs(h[t] - RH) <= tol and c[t] < RH:
                    setup, level = "range", RH
        if setup is None and er24 >= p.er_range and (c[t] - c[t - p.er_n]) * sg > 0:
            if sg == 1:
                cand = levels[(abs(levels - l[t]) <= tol) & (levels < c[t])]
                pick = cand[np.argmin(abs(cand - l[t]))] if len(cand) else None
            else:
                cand = levels[(abs(levels - h[t]) <= tol) & (levels > c[t])]
                pick = cand[np.argmin(abs(cand - h[t]))] if len(cand) else None
            if pick is not None:
                setup, level = "channel", pick
        if setup is None:
            t += 1; continue
        if p.dbl != 'off':
            lo_i, hi_i = t - p.dbl_win, t - p.dbl_gap
            bounce = max(p.dbl_bounce * level, 2 * A[t])
            idx = np.where(abs((l if sg == 1 else h)[lo_i:hi_i + 1] - level) <= tol)[0]
            dbl = False
            if len(idx):
                j0 = lo_i + idx[0]
                dbl = (h[j0 + 1:t].max() >= level + bounce) if sg == 1 else (l[j0 + 1:t].min() <= level - bounce)
            if p.dbl == 'strict' and not dbl:
                t += 1; continue
            if dbl:
                setup += "+double"
        if sg == 1:
            entry = c[t] * (1 + p.slip)
            stop = min(l[t], level) - p.stop_atr * A[t]
            R = entry - stop
        else:
            entry = c[t] * (1 - p.slip)
            stop = max(h[t], level) + p.stop_atr * A[t]
            R = stop - entry
        if not (p.min_r_pct <= R / entry <= p.max_r_pct):
            t += 1; continue
        if sg == 1:
            far = levels[levels >= entry + p.tp_r * R]
            target = min(far.min() if len(far) else entry + 2 * R, entry + 4 * R)
        else:
            far = levels[levels <= entry - p.tp_r * R]
            target = max(far.max() if len(far) else entry - 2 * R, entry - 4 * R)
        j, exit_px, why = t + 1, None, "time"
        while j < n:
            if (sg == 1 and l[j] <= stop) or (sg == -1 and h[j] >= stop):
                exit_px, why = stop, "stop"; break
            if (sg == 1 and h[j] >= target) or (sg == -1 and l[j] <= target):
                exit_px, why = target, "target"; break
            if j - t >= p.max_hold:
                exit_px = c[j]; break
            j += 1
        if exit_px is None:
            break
        if sg == 1:
            exit_px *= (1 - p.slip)
            ret = exit_px / entry * (1 - p.fee) / (1 + p.fee) - 1
        else:
            exit_px *= (1 + p.slip)
            ret = (entry - exit_px) / entry - 2 * p.fee   # no borrow/funding cost modelled
        trades.append(dict(i=t, exit_i=j, side=sg, setup=setup, why=why, ret=ret, r=ret / (R / entry), bars=j - t))
        t = j + 1
    return pd.DataFrame(trades)


def summarize(tr, n_bars, hold):
    if tr.empty:
        return dict(n=0, total=0.0, hold=hold)
    eq = (1 + tr["ret"]).cumprod()
    return dict(n=len(tr), win=(tr["ret"] > 0).mean(), avg=tr["ret"].mean() * 100, avgR=tr["r"].mean(),
                total=eq.iloc[-1] - 1, mdd=(1 - eq / eq.cummax()).max(),
                expo=tr["bars"].sum() / n_bars, hold=hold)


if __name__ == "__main__":
    fee = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0013
    slip = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0005
    p = P(fee=fee, slip=slip)
    w = Wallex(); allt = []
    print(f"fee={fee} slip={slip}")
    print(f"{'symbol':9}{'trades':>7}{'win%':>6}{'avg%':>7}{'avgR':>6}{'total%':>8}{'maxDD%':>8}{'expo%':>7}{'hold%':>8}")
    for s in SYMS:
        df = load_candles(w, s, "60", 365)
        tr = run(df, p); tr["sym"] = s; tr["frac"] = tr["i"] / len(df); allt.append(tr)
        hold = df["c"].iloc[-1] / df["c"].iloc[800] - 1
        r = summarize(tr, len(df) - 800, hold)
        if r["n"]:
            print(f"{s:9}{r['n']:7d}{r['win']*100:6.0f}{r['avg']:7.2f}{r['avgR']:6.2f}{r['total']*100:8.1f}{r['mdd']*100:8.1f}{r['expo']*100:7.1f}{hold*100:8.1f}")
        else:
            print(f"{s:9}   no trades   hold {hold*100:.1f}%")
    T = pd.concat(allt)
    print("\nBy setup (all symbols pooled):")
    print(T.groupby("setup").agg(n=("ret", "size"), win=("ret", lambda x: (x > 0).mean()), avg_pct=("ret", lambda x: x.mean() * 100),
                                 avgR=("r", "mean")).round(3))
    print("\nExit reasons:", T["why"].value_counts().to_dict())
    for name, m in [("first 60% of period", T["frac"] < 0.6), ("last 40% of period", T["frac"] >= 0.6)]:
        x = T[m]
        print(f"{name}: n={len(x)} win={(x['ret']>0).mean()*100:.0f}% avg={x['ret'].mean()*100:.2f}% avgR={x['r'].mean():.2f}")


def p4h(**kw):
    """Same durations as the hourly version, in 4h bars."""
    base = dict(bpd=6, warm=200, bias_n=42, er_n=12, range_win=9, brk_win=12, brk_recent=3, swing_k=3,
                swing_win=180, dbl_win=18, dbl_gap=2, max_hold=24, week_min=28, max_r_pct=0.10)
    base.update(kw); return P(**base)
