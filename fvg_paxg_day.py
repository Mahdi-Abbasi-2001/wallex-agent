"""Weekly-map 3-candle-gap strategy on Binance 1m for one 'local' day. python fvg_paxg_day.py [SYMBOL]"""
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

SYM = sys.argv[1] if len(sys.argv) > 1 else "PAXGUSDT"
MID = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
PREVN = int(sys.argv[3]) if len(sys.argv) > 3 else 0
TGT = float(sys.argv[4]) if len(sys.argv) > 4 else None
sys.argv = ["x", SYM]
import fvg_day as F
from binance_data import load

TZ = 210                                     # UTC+3:30 (the chart's timezone)
now = datetime.now(timezone.utc); local_today = (now + timedelta(minutes=TZ)).date(); day = local_today - timedelta(days=1)
start = int((datetime.combine(day, datetime.min.time(), timezone.utc) - timedelta(minutes=TZ)).timestamp()); end = start + 86400


def klines(a, b):
    rows = []
    for t in range(a, b, 60000):
        r = requests.get("https://api.binance.com/api/v3/klines", params={"symbol": SYM, "interval": "1m", "startTime": t * 1000, "endTime": (min(t + 60000, b) - 1) * 1000, "limit": 1000}, timeout=20).json()
        rows += [x[:6] for x in r]
    d = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"]).astype(float); d["t"] = (d.t // 1000).astype("int64")
    return d.drop_duplicates("t").sort_values("t").reset_index(drop=True)


d = klines(start, end)
monday = day - timedelta(days=day.weekday()); w0 = int(datetime.combine(monday - timedelta(days=7), datetime.min.time(), timezone.utc).timestamp()); w1 = w0 + 7 * 86400
cache = load(SYM); wk = cache[(cache.t >= w0) & (cache.t < w1)]
if len(wk) < 7 * 1440 * 0.95: wk = klines(w0, w1)
hi, lo = wk.h.max(), wk.l.min()
levels = {"W-high": hi, "75%": lo + .75 * (hi - lo), "50%": lo + .5 * (hi - lo), "25%": lo + .25 * (hi - lo), "W-low": lo}
F.SYMBOL, F.SOURCE, F.TZ_MIN, F.TZ_LABEL, F.BIN = SYM, "Binance", TZ, "UTC+3:30", None
loc = lambda t: datetime.fromtimestamp(int(t) + TZ * 60, timezone.utc).strftime("%H:%M")
print(f"{SYM} 1m | local day {day} (UTC+3:30) = {datetime.fromtimestamp(start, timezone.utc):%d %b %H:%M} -> {datetime.fromtimestamp(end, timezone.utc):%d %b %H:%M} UTC | {len(d)} candles")
print(f"price {d.o.iloc[0]:,.1f} -> {d.c.iloc[-1]:,.1f} ({(d.c.iloc[-1]/d.o.iloc[0]-1)*100:+.2f}%), range {d.l.min():,.1f}-{d.h.max():,.1f}")
print(f"weekly map from {monday - timedelta(days=7)} to {monday - timedelta(days=1)} (UTC Mon-Sun, {len(wk)} candles), 3 interior lines:")
for k, v in levels.items(): print(f"  {k:7} {v:,.2f}")
res = []
for lab, costs in (("Wallex costs", (F.MAKER, F.TAKER, F.SLIP)), ("zero costs", (0, 0, 0))):
    F.MAKER, F.TAKER, F.SLIP = costs; F.TARGET_R = TGT; F.MIN_GAP = 0.0001; F.MIN_RISK = 0.0; F.MID_MIN_ATR, F.PREV_N, F.PREV_X = MID, PREVN, 1.0
    tr = F.run(d, levels); res.append((lab, tr))
    print(f"\n{lab}: {len(tr)} trade(s) | sum of net {sum(x['net'] for x in tr)*100:+.2f}%")
    if lab == "Wallex costs":
        for n, x in enumerate(tr, 1):
            print(f" #{n} {x['side']:5} {x['kind']:14} @ {x['level']:6} {x['level_px']:,.1f} | touch {loc(d.t[x['touch_i']])} gap {loc(d.t[x['pat_i']])}-{loc(d.t[x['pat_i']+2])} order {loc(d.t[x['order_i']])} fill {loc(d.t[x['fill_i']])} @ {x['entry']:,.2f}\n"
                  f"    stop {x['stop']:,.2f} ({x['risk']*100:.3f}%) target {x['target']:,.2f} ({x['rr_planned']:.1f}R) | exit {loc(d.t[x['exit_i']])} @ {x['exit']:,.2f} ({x['why']}) | gross {x['gross']*100:+.3f}% net {x['net']*100:+.3f}%")
F.MAKER, F.TAKER, F.SLIP = (0.0010, 0.0013, 0.0005)
tr = res[0][1]; out = f"data/fvg_{SYM}_{day}_binance{'_filtered' if (MID or PREVN) else ''}.png"; F.draw(d, levels, tr, out, day); print("\nchart:", out, "| trades drawn:", len(tr))
