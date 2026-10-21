"""Weekly-map 3-candle-gap strategy: do size filters help? Binance 1m, ~177 days, BTCUSDT + PAXGUSDT."""
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

sys.argv = [sys.argv[0], "BTCUSDT"]
import fvg_day as F
from binance_data import load

CONFIGS = {"no filter": (0.0, 0, 1.0), "A: middle candle >= 1 ATR": (1.0, 0, 1.0), "B: bigger than prev 5 candles": (0.0, 5, 1.0),
           "A+B": (1.0, 5, 1.0), "strict: >=1.5 ATR & bigger than prev 10": (1.5, 10, 1.0)}
TARGETS = {"next weekly line": None, "2R": 2.0, "1R": 1.0}
MAKER, TAKER, SLIP = 0.0010, 0.0013, 0.0005


def run_symbol(sym):
    d = load(sym); d["day"] = d.t // 86400; days = sorted(d.day.unique()); groups = {k: g.reset_index(drop=True) for k, g in d.groupby("day")}
    def levels_for(day):
        dt = datetime.fromtimestamp(day * 86400, timezone.utc).date(); mon = dt - timedelta(days=dt.weekday())
        a = int(datetime.combine(mon - timedelta(days=7), datetime.min.time(), timezone.utc).timestamp()) // 86400; b = a + 7
        wk = d[(d.day >= a) & (d.day < b)]
        if len(wk) < 7 * 1440 * 0.95: return None
        hi, lo = wk.h.max(), wk.l.min()
        return {"W-high": hi, "75%": lo + .75 * (hi - lo), "50%": lo + .5 * (hi - lo), "25%": lo + .25 * (hi - lo), "W-low": lo}
    valid = [(k, levels_for(k)) for k in days]; valid = [(k, lv) for k, lv in valid if lv is not None and len(groups[k]) >= 1400]
    print(f"\n######## {sym}: {len(valid)} days ########", flush=True)
    F.MAKER = F.TAKER = F.SLIP = 0.0; F.MIN_GAP = 0.0001; F.MIN_RISK = 0.0
    out = {}
    for cname, (mid, pn, px) in CONFIGS.items():
        F.MID_MIN_ATR, F.PREV_N, F.PREV_X = mid, pn, px
        for tname, tr_ in TARGETS.items():
            F.TARGET_R = tr_; rows = []; t0 = time.time()
            for k, lv in valid:
                for tr in F.run(groups[k], lv): tr["day"] = k; rows.append(tr)
            T = pd.DataFrame(rows); out[(cname, tname)] = T
            print(f"  done {cname} | {tname}: {len(T)} trades ({time.time()-t0:.0f}s)", flush=True)
    return out, len(valid)


def report(sym, out, ndays):
    for tname in TARGETS:
        print(f"\n=== {sym} | target: {tname} | {ndays} days ===")
        print(f"{'filter':42}{'trades':>7}{'/day':>6}{'win%':>6}{'stop%':>7}{'gross%':>8}{'grossR':>7}{'netR(Wallex)':>13}{'net%':>8}  H1/H2 grossR")
        for cname in CONFIGS:
            T = out[(cname, tname)]
            if len(T) < 10: print(f"{cname:42}{len(T):7d}  (too few)"); continue
            g = T.gross; fee_out = np.where(T.why == "target", MAKER, TAKER); slip = np.where(T.why == "target", 0.0, SLIP)
            net = g - MAKER - fee_out - slip; r_g = g / T.risk; r_n = net / T.risk; half = T.day.median()
            h1, h2 = r_g[T.day <= half].mean(), r_g[T.day > half].mean()
            print(f"{cname:42}{len(T):7d}{len(T)/ndays:6.1f}{(g>0).mean()*100:6.0f}{T.risk.median()*100:7.3f}{g.mean()*100:+8.3f}{r_g.mean():+7.2f}{r_n.mean():+13.2f}{net.mean()*100:+8.3f}  {h1:+.2f}/{h2:+.2f}")


if __name__ == "__main__":
    for sym in ("PAXGUSDT", "BTCUSDT"):
        out, n = run_symbol(sym); report(sym, out, n)
        pd.to_pickle(out, f"data/fvg_filter_{sym}.pkl")
