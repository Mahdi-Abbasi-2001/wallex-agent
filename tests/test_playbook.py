"""Smoke test: every scenario builds an agent that runs through the simulator. python tests/test_playbook.py"""
import itertools
import sys
sys.path.insert(0, ".")
import numpy as np
import strat_search as S
from contest.fills import BidAskFill
from contest.market import Market
from contest.playbook import Rules, make_agent
from contest.sim import SimConfig, run_sim

mkt = Market(S.load_tf("1d")[1], 86400); n = 0
for short, lev, style, obj, fee in itertools.product((False, True), (1.0, 3.0), ("hold", "trade"), ("rank", "safe"), (0.0, 0.13)):
    ag = make_agent(Rules(short, lev, style, obj, fee)); cfg = SimConfig(taker=fee / 100, maker=fee / 100 * 0.8, fill=BidAskFill(), leverage=max(lev, 1.0), allow_short=short)
    for s in (300, 700, 1000):
        r = run_sim(mkt, ag, s, s + 7, cfg); assert np.isfinite(r.ret) and r.ret > -1.0 - 1e-9 and not np.isnan(r.eq).any(), (short, lev, style, obj, fee, r.ret); n += 1
print(f"playbook smoke test passed ({n} simulated contests); example agents:")
for rules in (Rules(), Rules(True, 3.0, "hold"), Rules(True, 3.0, "trade", taker_fee_pct=0.0), Rules(False, 3.0, "trade"), Rules(objective="safe")): print("  ", rules, "->", make_agent(rules).name)

from contest.agents import Basket
assert isinstance(make_agent(Rules(objective="safe", contest_days=2)), Basket) and isinstance(make_agent(Rules(objective="basket")), Basket)
assert not isinstance(make_agent(Rules(objective="safe", contest_days=7)), Basket)
r = run_sim(mkt, make_agent(Rules(objective="basket")), 1000, 1002, SimConfig(taker=0.0013, maker=0.001, fill=BidAskFill())); assert np.isfinite(r.ret)
print("basket checks passed")
