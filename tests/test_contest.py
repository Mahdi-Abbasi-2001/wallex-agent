"""Sanity tests for the contest simulator (toy data). python tests/test_contest.py"""
import sys
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from contest.agents import Agent, HoldAll
from contest.fills import BidAskFill, BookFill, MidFill
from contest.market import Market
from contest.orders import Order
from contest.sim import SimConfig, run_sim


def mk(prices, spread=0.0, highs=None, lows=None):
    """one asset 'AAA' (+ 'BBB' flat) with open = previous close"""
    p = np.array(prices, float); o = np.r_[p[0], p[:-1]]; h = np.maximum(o, p) * (1 + spread) if highs is None else np.array(highs, float); l = np.minimum(o, p) * (1 - spread) if lows is None else np.array(lows, float)
    d = pd.DataFrame({"t": np.arange(len(p)) * 86400, "o": o, "h": h, "l": l, "c": p, "v": 1.0}); flat = d.copy(); flat[["o", "h", "l", "c"]] = 100.0
    return Market({"AAA": d, "BBB": flat}, 86400)


class Script(Agent):
    def __init__(self, orders_by_bar): self.o = orders_by_bar; self.n = 0
    def reset(self): self.n = 0
    def on_bar(self, ctx): r = self.o.get(self.n, []); self.n += 1; return r


# 1. buy & hold with zero fees = price ratio (enter at first open, mark at last close)
m = mk([100, 110, 121, 133.1]); r = run_sim(m, HoldAll(["AAA"]), 1, 4, SimConfig(taker=0, maker=0))
assert abs(r.ret - 0.331) < 1e-9, r.ret
# 2. fees: round trip costs ~2*fee
m = mk([100, 100, 100, 100]); r = run_sim(m, Script({0: [Order("AAA", "buy", notional=1000)], 1: [Order("AAA", "sell", qty=10)]}), 1, 4, SimConfig(taker=0.001))
assert abs(r.ret + 0.002) < 1e-4 and abs(r.fees - 2.0) < 0.01, (r.ret, r.fees)   # two taker fees of 0.1% on ~$999
# 3. limit buy fills only if low touches; fill at min(limit, open)
m = mk([100, 100, 100, 100], lows=[100, 100, 94, 100], highs=[100, 100, 100, 100]); sc = Script({0: [Order("AAA", "buy", "limit", notional=500, price=95, ttl=5)]})
m2 = mk([100, 100, 100, 100], lows=[100, 100, 94, 100], highs=[100, 100, 100, 100]); r = run_sim(m2, Script({0: [Order("AAA", "buy", "limit", notional=500, price=95, ttl=5)]}), 1, 4, SimConfig(taker=0, maker=0)); assert r.fills == 1, r.fills
m3 = mk([100, 100, 100, 100], lows=[100, 100, 96, 100], highs=[100, 100, 100, 100]); r = run_sim(m3, Script({0: [Order("AAA", "buy", "limit", notional=500, price=95, ttl=5)]}), 1, 4, SimConfig(taker=0, maker=0)); assert r.fills == 0, r.fills
# 4. stop-sell is gap aware: bar opens below the stop -> fills at the open
m = mk([100, 100, 80, 80], lows=[100, 100, 80, 80], highs=[100, 100, 100, 80])   # bar 2 opens at 100 -> low 80 (crash inside bar), stop 90 fills at 90
sc = Script({0: [Order("AAA", "buy", notional=1000)], 1: [Order("AAA", "sell", "stop", qty=10, stop=90, ttl=9)]}); r = run_sim(m, sc, 1, 4, SimConfig(taker=0, maker=0)); assert abs(r.ret - (-0.10)) < 1e-9, r.ret
# 5. spot: no shorting; cash limits buys
m = mk([100] * 5); r = run_sim(m, Script({0: [Order("AAA", "sell", qty=5)], 1: [Order("AAA", "buy", notional=5000)]}), 1, 5, SimConfig(start_cash=1000, taker=0)); assert r.eq[-1] <= 1000.0001 and abs(r.eq[-1] - 1000) < 1e-6, r.eq
# 6. bid/ask fill: buying costs half-spread, selling costs half-spread
f = BidAskFill({"AAA": 100}); assert abs(f.market("AAA", 1, 1.0, 100)[0] - 100.5) < 1e-9 and abs(f.market("AAA", -1, 1.0, 100)[0] - 99.5) < 1e-9
# 7. book fill: small order near touch, big order partial/cancel under the 2% rule
b = BookFill({"AAA": 20}, {"AAA": 1000}, step_bps=5, partial=True); px, q = b.market("AAA", 1, 1.0, 100); assert abs(q - 1.0) < 1e-9 and px > 100.1 - 1e-9
px2, q2 = b.market("AAA", 1, 100.0, 100); assert q2 < 100.0 * 0.2, q2                     # $10k order vs $1k depth -> only a fraction fills
bc = BookFill({"AAA": 20}, {"AAA": 1000}, partial=False); assert bc.market("AAA", 1, 100.0, 100)[1] == 0.0
# 8. margin: 5x long and a 25% drop -> liquidated, equity never negative
m = mk([100, 100, 75, 75]); r = run_sim(m, Script({0: [Order("AAA", "buy", notional=5000)]}), 1, 4, SimConfig(start_cash=1000, taker=0.001, leverage=5.0)); assert r.liquidated and r.eq.min() >= 0, (r.liquidated, r.eq)
# 9. short allowed only when enabled
m = mk([100, 100, 90, 90]); r = run_sim(m, Script({0: [Order("AAA", "sell", notional=500)]}), 1, 4, SimConfig(taker=0, allow_short=True, leverage=2.0)); assert abs(r.ret - 0.05) < 1e-9, r.ret
print("all contest simulator tests passed")
