"""Weekly-map reaction / break-and-pullback + 3-candle gap (FVG) limit entry, single-day test + chart.
 python fvg_day.py [SYMBOL] [YYYY-MM-DD]      (day = a UTC day; weekly map from the previous Mon-Sun UTC week)"""
import sys
from datetime import datetime, timedelta, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agent.wallex_client import Wallex

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
DAY = sys.argv[2] if len(sys.argv) > 2 else (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
TOL, MIN_GAP, W_PATTERN, BREAK_LOOK, CANCEL_AFTER = 0.0005, 0.0001, 15, 120, 30
MID_MIN_ATR, PREV_N, PREV_X = 0.0, 0, 1.0      # size filters: middle candle range >= MID_MIN_ATR x ATR; formation height >= PREV_X x max range of previous PREV_N candles
TARGET_R = None         # None = next weekly-map line; else fixed R multiple
SOURCE, TZ_MIN, TZ_LABEL = "Wallex", 0, "UTC"     # chart labelling only
MIN_RISK = 0.0          # optional filter: skip setups whose stop is closer than this fraction of price
MAKER, TAKER, SLIP = 0.0010, 0.0013, 0.0005


def ts(day): return int(datetime.fromisoformat(str(day)).replace(tzinfo=timezone.utc).timestamp())


def fetch_1m(w, start, end):
    rows = []
    for t0 in range(start, end, 86400):
        c = w.candles(SYMBOL, "1", t0, min(t0 + 86400, end) - 1)
        if c.get("s") == "ok": rows += list(zip(c["t"], c["o"], c["h"], c["l"], c["c"], c["v"]))
    d = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"]).astype(float); d["t"] = d["t"].astype("int64")
    return d.drop_duplicates("t").sort_values("t").reset_index(drop=True)


def weekly_map(w, day):
    d0 = datetime.fromisoformat(str(day)).date(); monday = d0 - timedelta(days=d0.weekday())
    ws, we = ts(monday - timedelta(days=7)), ts(monday)
    wk = fetch_1m(w, ws, we); hi, lo = wk.h.max(), wk.l.min()
    return {"W-high": hi, "75%": lo + .75 * (hi - lo), "50%": lo + .5 * (hi - lo), "25%": lo + .25 * (hi - lo), "W-low": lo}, (monday - timedelta(days=7), monday - timedelta(days=1)), len(wk)


def run(d, levels):
    t, o, h, l, c = (d[k].to_numpy() for k in "tohlc"); n = len(d)
    lv = sorted(levels.items(), key=lambda kv: kv[1]); trades, busy_until = [], -1
    _pc = np.r_[c[0], c[:-1]]; atr = pd.Series(np.maximum(h - l, np.maximum(abs(h - _pc), abs(l - _pc)))).rolling(14).mean().to_numpy()
    for i in range(1, n - 3):
        e = i + 2
        if e <= busy_until: continue
        for side in (1, -1):
            gap = (l[e] - h[i]) if side == 1 else (l[i] - h[e])           # bullish: candle1 high < candle3 low
            if gap < MIN_GAP * c[i]: continue
            if MID_MIN_ATR > 0 or PREV_N > 0:
                if np.isnan(atr[i]) or i < PREV_N + 1: continue
                if MID_MIN_ATR > 0 and (h[i + 1] - l[i + 1]) < MID_MIN_ATR * atr[i]: continue
                if PREV_N > 0 and (h[i:e + 1].max() - l[i:e + 1].min()) < PREV_X * (h[i - PREV_N:i] - l[i - PREV_N:i]).max(): continue
            for name, L in lv:
                tol = TOL * L; hit = None
                for k in range(max(1, i - W_PATTERN), i + 1):             # reaction (touch) bar, at or before candle 1
                    if side == 1 and l[k] <= L + tol and l[k] >= L - 3 * tol and c[k] >= L - tol and c[k - 1] > L: hit = k
                    if side == -1 and h[k] >= L - tol and h[k] <= L + 3 * tol and c[k] <= L + tol and c[k - 1] < L: hit = k
                if hit is None: continue
                seg = slice(hit, e + 1)                                   # level must hold through the formation
                if side == 1 and (c[seg] < L - tol).any(): continue
                if side == -1 and (c[seg] > L + tol).any(): continue
                lb = slice(max(0, hit - BREAK_LOOK), hit)
                broke = (c[lb] > L + tol).any() if side == 1 and False else None
                # break+pullback = price had closed beyond the level on the other side of the pullback within BREAK_LOOK bars
                broke = ((c[lb] > L + tol).any() if side == 1 else (c[lb] < L - tol).any())
                kind = "break+pullback" if broke else "reaction"
                entry = l[e] if side == 1 else h[e]
                stop = min(l[i:e + 1]) if side == 1 else max(h[i:e + 1])
                R = abs(entry - stop)
                if R <= 0 or R / entry < MIN_RISK: continue
                cands = [v for _, v in lv if (v > entry if side == 1 else v < entry)]
                target = (min(cands) if side == 1 else max(cands)) if cands else entry + side * 2 * R
                if TARGET_R is not None: target = entry + side * TARGET_R * R
                fill = None
                for j in range(e + 1, min(n, e + 1 + CANCEL_AFTER)):
                    if (side == 1 and h[j] >= target) or (side == -1 and l[j] <= target): break   # ran away, never filled
                    if (side == 1 and l[j] <= stop and l[j] > entry) or (side == -1 and h[j] >= stop and h[j] < entry): break
                    if (side == 1 and l[j] <= entry) or (side == -1 and h[j] >= entry): fill = j; break
                if fill is None: continue
                ex, why, xi = None, "end of day", n - 1
                for j in range(fill, n):
                    sh = (l[j] <= stop) if side == 1 else (h[j] >= stop)
                    th = (h[j] >= target) if side == 1 else (l[j] <= target)
                    if j == fill: th = False
                    if sh: ex, why, xi = (stop if j == fill else (min(stop, o[j]) if side == 1 else max(stop, o[j]))), "stop", j; break    # a gap through the stop fills at the open
                    if th: ex, why, xi = (max(target, o[j]) if side == 1 else min(target, o[j])), "target", j; break
                if ex is None: ex = c[-1]
                fee_in = MAKER; fee_out, slip = (MAKER, 0) if why == "target" else (TAKER, SLIP)
                px_out = ex * (1 - slip * side)
                gross = side * (px_out / entry - 1); net = gross - fee_in - fee_out
                trades.append(dict(side="LONG" if side == 1 else "SHORT", level=name, level_px=L, kind=kind, touch_i=hit, pat_i=i, order_i=e + 1, fill_i=fill,
                                   exit_i=xi, entry=entry, stop=stop, target=target, exit=ex, why=why, gross=gross, net=net, rr_planned=abs(target - entry) / R, risk=R / entry))
                busy_until = xi; break
            if busy_until >= e: break
    return trades


BIN = None


def draw(d, levels, trades, path, day):
    t = pd.to_datetime(d.t + TZ_MIN * 60, unit="s"); x = np.arange(len(d)); cols = min(4, max(1, len(trades))); rows = max(1, -(-len(trades) // cols)); fig = plt.figure(figsize=(17, 6.0 + 3.1 * rows)); gs = fig.add_gridspec(1 + rows, cols, height_ratios=[2.4] + [1.3] * rows)
    def candles(ax, a, b):
        for k in range(a, b):
            if d.o[k] == d.h[k] == d.l[k] == d.c[k]:
                ax.hlines(d.c[k], k - 0.45, k + 0.45, color="#9e9e9e", lw=0.8 if b - a > 200 else 1.5); continue
            col = "#26a69a" if d.c[k] >= d.o[k] else "#ef5350"
            ax.vlines(k, d.l[k], d.h[k], color=col, lw=0.8); ax.vlines(k, min(d.o[k], d.c[k]), max(d.o[k], d.c[k]) + 1e-9, color=col, lw=2.2 if b - a < 200 else 1)
    ax = fig.add_subplot(gs[0, :]); candles(ax, 0, len(d))
    lo_y, hi_y = d.l.min(), d.h.max(); pad = 0.12 * (hi_y - lo_y); outside = []
    if BIN is not None: ax.plot(BIN.index, BIN.c.to_numpy(), color="#42a5f5", lw=0.9, alpha=.8, label="Binance close (reference)", zorder=1); ax.legend(loc="upper left", fontsize=8)
    for name, v in levels.items():
        if lo_y - pad <= v <= hi_y + pad:
            ax.axhline(v, color="#7e57c2", ls="--", lw=1); ax.text(len(d) * 1.002, v, f"{name} {v:,.0f}", va="center", fontsize=8, color="#5e35b1")
        else: outside.append(f"{name} {v:,.0f}")
    ax.set_ylim(lo_y - pad, hi_y + pad)
    for n_, tr in enumerate(trades, 1):
        c_ = "#1b5e20" if tr["net"] > 0 else "#b71c1c"
        ax.axvspan(tr["pat_i"] - .5, tr["order_i"] - 1.5, color="#ffb300", alpha=.35)
        ax.plot([tr["fill_i"]], [tr["entry"]], "^" if tr["side"] == "LONG" else "v", ms=10, color="blue", zorder=5)
        ax.plot([tr["exit_i"]], [tr["exit"]], "x", ms=10, mew=2, color=c_, zorder=5)
        ax.text(tr["fill_i"], tr["entry"], f" #{n_}", fontsize=10, weight="bold", color="blue")
    ax.set_xlim(-5, len(d) + 90); ax.set_title(f"{SYMBOL} 1m  {day} {TZ_LABEL} | {SOURCE} candles (grey tick = minute with no price change: {int(((d.o==d.h)&(d.h==d.l)&(d.l==d.c)).sum())}/{len(d)}) | weekly map dashed (off-screen: {', '.join(outside) or 'none'}) | gold = gap formation | ^/v fill, x exit", fontsize=10)
    ticks = list(range(0, len(d), 120)); ax.set_xticks(ticks); ax.set_xticklabels([t[k].strftime("%H:%M") for k in ticks]); ax.grid(alpha=.15)
    for n_, tr in enumerate(trades, 1):
        a = fig.add_subplot(gs[1 + (n_ - 1) // cols, (n_ - 1) % cols]); lo_, hi_ = max(0, tr["touch_i"] - 12), min(len(d), tr["exit_i"] + 12); candles(a, lo_, hi_)
        L = tr["level_px"]; a.axhline(L, color="#7e57c2", ls="--", lw=1); a.text(lo_, L, f" {tr['level']} {L:,.0f}", fontsize=8, color="#5e35b1", va="bottom")
        a.axvspan(tr["pat_i"] - .5, tr["pat_i"] + 2.5, color="#ffb300", alpha=.35)
        a.hlines(tr["entry"], tr["order_i"], tr["fill_i"], color="blue", ls=":"); a.plot([tr["fill_i"]], [tr["entry"]], "^" if tr["side"] == "LONG" else "v", ms=10, color="blue")
        a.axhline(tr["stop"], color="red", lw=1); a.axhline(tr["target"], color="green", lw=1); a.plot([tr["exit_i"]], [tr["exit"]], "x", ms=11, mew=2, color="k")
        a.plot([tr["touch_i"]], [d.l[tr["touch_i"]] if tr["side"] == "LONG" else d.h[tr["touch_i"]]], "o", ms=7, mfc="none", mec="#7e57c2", mew=2)
        a.set_xlim(lo_ - 1, hi_ + 1); a.set_xticks([lo_, (lo_ + hi_) // 2, hi_ - 1]); a.set_xticklabels([t[k].strftime("%H:%M") for k in (lo_, (lo_ + hi_) // 2, hi_ - 1)])
        a.set_title(f"#{n_} {tr['side']} {'brk+pb' if tr['kind'].startswith('break') else 'react'} @{tr['level']} | {tr['why']} {tr['net']*100:+.2f}%", fontsize=9); a.grid(alpha=.15)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


if __name__ == "__main__":
    w = Wallex(); day = DAY
    levels, (m0, m1), nwk = weekly_map(w, day)
    d = fetch_1m(w, ts(day), ts(day) + 86400)
    print(f"{SYMBOL} {day} UTC: {len(d)} 1m candles | weekly map from {m0} to {m1} ({nwk} candles)")
    for k, v in levels.items(): print(f"  {k:7} {v:,.2f}")
    tr = run(d, levels); t = pd.to_datetime(d.t, unit="s")
    print(f"\n{len(tr)} trade(s):")
    for n_, x in enumerate(tr, 1):
        print(f" #{n_} {x['side']} {x['kind']} @ {x['level']} {x['level_px']:,.1f} | touch {t[x['touch_i']]:%H:%M}, gap formation {t[x['pat_i']]:%H:%M}-{t[x['pat_i']+2]:%H:%M}, order {t[x['order_i']]:%H:%M}, fill {t[x['fill_i']]:%H:%M} @ {x['entry']:,.2f}\n"
              f"    stop {x['stop']:,.2f} | target {x['target']:,.2f} (planned {x['rr_planned']:.1f}R, risk {x['risk']*100:.2f}%) | exit {t[x['exit_i']]:%H:%M} @ {x['exit']:,.2f} ({x['why']}) | gross {x['gross']*100:+.2f}%  net {x['net']*100:+.2f}%")
    try:
        import requests
        r = []
        for off in (0, 1000):                      # Binance caps each request at 1000 candles
            r += requests.get("https://api.binance.com/api/v3/klines", params={"symbol": SYMBOL, "interval": "1m", "startTime": (ts(day) + off * 60) * 1000, "limit": 440 if off else 1000}, timeout=15).json()
        b = pd.DataFrame(r).iloc[:, :5].astype(float); b.columns = ["t", "o", "h", "l", "c"]; b["t"] = (b.t // 1000).astype("int64")
        pos = {tt: i for i, tt in enumerate(d.t)}; b = b[b.t.isin(pos)]; b.index = [pos[tt] for tt in b.t]; BIN = b
    except Exception as e:
        print("no binance overlay:", e)
    out = f"data/fvg_{SYMBOL}_{day}.png"; draw(d, levels, tr, out, day); print("chart:", out)
