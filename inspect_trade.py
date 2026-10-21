"""Inspect one trade of the weekly-map 3-candle-gap strategy minute by minute. python inspect_trade.py [trade_no]"""
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

NO = int(sys.argv[1]) if len(sys.argv) > 1 else 1
sys.argv = ["x", "PAXGUSDT"]
import fvg_day as F
from binance_data import load

SYM, TZ = "PAXGUSDT", 210
now = datetime.now(timezone.utc); day = (now + timedelta(minutes=TZ)).date() - timedelta(days=1)
start = int((datetime.combine(day, datetime.min.time(), timezone.utc) - timedelta(minutes=TZ)).timestamp()); end = start + 86400
rows = []
for t in range(start, end, 60000):
    r = requests.get("https://api.binance.com/api/v3/klines", params={"symbol": SYM, "interval": "1m", "startTime": t * 1000, "endTime": (min(t + 60000, end) - 1) * 1000, "limit": 1000}, timeout=20).json(); rows += [x[:6] for x in r]
d = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"]).astype(float); d["t"] = (d.t // 1000).astype("int64"); d = d.drop_duplicates("t").sort_values("t").reset_index(drop=True)
monday = day - timedelta(days=day.weekday()); w0 = int(datetime.combine(monday - timedelta(days=7), datetime.min.time(), timezone.utc).timestamp()); c = load(SYM); wk = c[(c.t >= w0) & (c.t < w0 + 7 * 86400)]
hi, lo = wk.h.max(), wk.l.min(); levels = {"W-high": hi, "75%": lo + .75 * (hi - lo), "50%": lo + .5 * (hi - lo), "25%": lo + .25 * (hi - lo), "W-low": lo}
F.MAKER, F.TAKER, F.SLIP = 0.0010, 0.0013, 0.0005; F.TARGET_R = None; F.MIN_GAP = 0.0001; F.MIN_RISK = 0.0
tr = F.run(d, levels); x = tr[NO - 1]
loc = lambda i: datetime.fromtimestamp(int(d.t[i]) + TZ * 60, timezone.utc).strftime("%H:%M")
atr = pd.Series(np.maximum(d.h - d.l, np.maximum((d.h - d.c.shift()).abs(), (d.l - d.c.shift()).abs()))).rolling(14).mean()
print(f"TRADE #{NO}: {x['side']} {x['kind']} at weekly {x['level']} line {x['level_px']:.2f} (local times UTC+3:30)")
print(f"  touch {loc(x['touch_i'])} | gap formation candles {loc(x['pat_i'])},{loc(x['pat_i']+1)},{loc(x['pat_i']+2)} | order from {loc(x['order_i'])} | fill {loc(x['fill_i'])} | exit {loc(x['exit_i'])} ({x['why']})")
print(f"  entry {x['entry']:.2f} | stop {x['stop']:.2f} | target {x['target']:.2f} | risk ${x['entry']-x['stop']:.2f} = {x['risk']*100:.3f}% | planned {x['rr_planned']:.1f}R | ATR(14) at signal ${atr[x['pat_i']+2]:.2f}")
print("\nformation + fill (o h l c):")
for i in range(x["touch_i"] - 1, x["order_i"] + 1):
    tag = []
    if i == x["touch_i"]: tag.append("<- touch of level")
    if i in (x["pat_i"], x["pat_i"] + 1, x["pat_i"] + 2): tag.append(f"<- gap candle {i - x['pat_i'] + 1}")
    print(f"  {loc(i)} {d.o[i]:8.2f} {d.h[i]:8.2f} {d.l[i]:8.2f} {d.c[i]:8.2f}  {' '.join(tag)}")
g1, g3 = d.h[x["pat_i"]], d.l[x["pat_i"] + 2]; print(f"  gap: candle1 high {g1:.2f} < candle3 low {g3:.2f}  (gap ${g3-g1:.2f}, = entry level {x['entry']:.2f}); stop = lowest low of the 3 candles ${d.l[x['pat_i']:x['pat_i']+3].min():.2f} minus 0.1 ATR")
seg = d.iloc[x["fill_i"]:x["exit_i"] + 1]
print(f"\nwhile in the trade ({loc(x['fill_i'])} -> {loc(x['exit_i'])}, {len(seg)} min): high {seg.h.max():.2f} (+${seg.h.max()-x['entry']:.2f} = {(seg.h.max()-x['entry'])/(x['entry']-x['stop']):.1f}R), low {seg.l.min():.2f}")
print(f"  median 1-min range in this window: ${(d.h-d.l).iloc[x['touch_i']-30:x['exit_i']].median():.2f}  vs the stop distance ${x['entry']-x['stop']:.2f}")
print("\nlast minutes before / at the stop:")
for i in range(x["exit_i"] - 6, x["exit_i"] + 6):
    print(f"  {loc(i)} {d.o[i]:8.2f} {d.h[i]:8.2f} {d.l[i]:8.2f} {d.c[i]:8.2f} {'<- STOP HIT (low <= stop)' if i == x['exit_i'] else ''}")
print(f"\nwhat price did after the stop: 30 min later {d.c[x['exit_i']+30]:.2f}, 2h later {d.c[x['exit_i']+120]:.2f}, low of next 3h {d.l[x['exit_i']:x['exit_i']+180].min():.2f}")
R = x["entry"] - x["stop"]; print("\nwhat-ifs with the same entry (no costs):")
for k in (1, 2, 3, 5):
    stop_k = x["entry"] - k * R
    for tp in (1.0, 2.0):
        tgt = x["entry"] + tp * k * R; res = "open"
        for i in range(x["fill_i"] + 1, len(d)):
            if d.l[i] <= stop_k: res = f"STOP at {loc(i)}"; break
            if d.h[i] >= tgt: res = f"TARGET at {loc(i)}"; break
        print(f"  stop {k}x wider (${k*R:.2f}) target {tp:g}R (${tp*k*R:.2f}): {res}")
print("\ncost breakdown of the net result:")
slip = F.SLIP * x["stop"]; print(f"  price move entry->stop: {(x['stop']/x['entry']-1)*100:+.3f}% | assumed stop slippage 0.050% (${slip:.2f}) | fees 0.10% maker in + 0.13% taker out = 0.230% | net {x['net']*100:+.3f}%")

# ---- annotated chart of this one trade ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
a, b = x["touch_i"] - 8, min(len(d), x["exit_i"] + 10)
fig, ax = plt.subplots(figsize=(17, 8.5))
for i in range(a, b):
    col = "#26a69a" if d.c[i] >= d.o[i] else "#ef5350"
    if d.o[i] == d.h[i] == d.l[i] == d.c[i]: ax.hlines(d.c[i], i - .4, i + .4, color="#9e9e9e", lw=1.5); continue
    ax.vlines(i, d.l[i], d.h[i], color=col, lw=1); ax.vlines(i, min(d.o[i], d.c[i]), max(d.o[i], d.c[i]) + 1e-9, color=col, lw=4)
ax.axhline(x["level_px"], color="#7e57c2", ls="--", lw=1.3); ax.text(b, x["level_px"], f" weekly 50% line {x['level_px']:.2f}", color="#5e35b1", va="center")
ax.axhline(x["entry"], color="blue", ls=":", lw=1.2); ax.text(b, x["entry"], f" limit entry {x['entry']:.2f}", color="blue", va="center")
ax.axhline(x["stop"], color="red", lw=1.3); ax.text(b, x["stop"], f" stop {x['stop']:.2f}", color="red", va="center")
ax.axhline(x["entry"] + R, color="green", ls=":", lw=1); ax.text(b, x["entry"] + R, f" 1R target would be {x['entry']+R:.2f}", color="green", va="center")
ax.axvspan(x["pat_i"] - .5, x["pat_i"] + 2.5, color="#ffb300", alpha=.3); ax.text(x["pat_i"] + 1, d.h[a:b].max(), "3-candle gap\nformation", ha="center", va="top", fontsize=9)
ax.annotate("price touches line,\ncloses below it for 4 min", (x["touch_i"], d.l[x["touch_i"] + 1]), (x["touch_i"] - 6, d.l[x["touch_i"] + 1] - 1.2), arrowprops=dict(arrowstyle="->"), fontsize=9)
ax.hlines(x["entry"], x["order_i"], x["fill_i"], color="blue", ls="-", lw=2, alpha=.5); ax.text((x["order_i"] + x["fill_i"]) / 2, x["entry"] + .25, f"order waits {x['fill_i']-x['order_i']} min", ha="center", fontsize=9, color="blue")
ax.plot([x["fill_i"]], [x["entry"]], "^", ms=13, color="blue"); ax.plot([x["exit_i"]], [x["stop"]], "x", ms=14, mew=3, color="red")
hi_i = x["fill_i"] + int(np.argmax(d.h[x["fill_i"]:x["exit_i"] + 1].to_numpy())); ax.annotate(f"best point: +{(d.h[hi_i]-x['entry'])/R:.1f}R", (hi_i, d.h[hi_i]), (hi_i - 3, d.h[hi_i] + 1), arrowprops=dict(arrowstyle="->"), fontsize=9)
ax.annotate("wick pierces stop by $0.37,\nthen price recovers", (x["exit_i"], d.l[x["exit_i"]]), (x["exit_i"] - 14, d.l[x["exit_i"]] - 1.3), arrowprops=dict(arrowstyle="->"), fontsize=9, color="red")
ticks = list(range(a, b, 6)); ax.set_xticks(ticks); ax.set_xticklabels([loc(i) for i in ticks]); ax.set_xlim(a - 1, b + 28); ax.grid(alpha=.15)
ax.set_title(f"Trade #{NO}: {SYM} 1m (Binance), {x['side']} at weekly 50% line | grey tick = no price change that minute | times UTC+3:30", fontsize=11)
out = f"data/trade{NO}_{SYM}_inspect.png"; fig.tight_layout(); fig.savefig(out, dpi=110); print("\nchart:", out)
