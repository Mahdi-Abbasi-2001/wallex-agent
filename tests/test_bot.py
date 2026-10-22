"""Unit tests for the spot bot: risk guard, paper broker fees/liquidation/stops, end-game phases, full open->close pipeline with a fake LLM.
 python tests/test_bot.py"""
import os, sys, tempfile, time
sys.path.insert(0, ".")
from bot.config import Config
from bot.futures import PaperFutures
from bot.risk import RiskGuard, phase
from bot.runner import Bot
from bot.state import State

tmp = tempfile.mkdtemp()
cfg = Config(log_dir=tmp, state_file=os.path.join(tmp, "s.json"), kill_file=os.path.join(tmp, "KILL"), challenge_end=time.time() + 40 * 3600, max_positions=3, daily_loss_cap=0.02)
now = time.time()

def guard():
    st = State(os.path.join(tmp, "g.json")); g = RiskGuard(cfg, st); g.update_equity(5_000_000, now); return g, st

# --- chosen rule set is the default
d = Config(); assert (d.entry_mode, d.regime_gate, d.max_positions, d.daily_loss_cap) == ("pullback", "strict", 2, 0.01)

# --- phases
assert phase(cfg, now) == "trade"
assert phase(Config(challenge_end=now + 3 * 3600), now) == "trade" and phase(Config(challenge_end=now + 90 * 60), now) == "no_new" and phase(Config(challenge_end=now + 20 * 60), now) == "unwind"
assert phase(Config(challenge_end=now + 4 * 60), now) == "force" and phase(Config(challenge_end=now - 1), now) == "ended"

# --- sizing (spot, long only): risk-based, capped at 33% of equity
g, st = guard(); base = dict(market="BTCTMN", side="long", entry=100.0, stop_pct=0.02, take_pct=0.04, lev=1)
s, why = g.check_open(base, 5_000_000, [], now); assert s and s.risk_coef == 1 and s.collateral == 1_650_000 and abs(s.stop_price - 98.0) < 1e-9 and abs(s.take_price - 104.0) < 1e-9, why
big = g.check_open({**base, "stop_pct": 0.05}, 5_000_000, [], now)[0]; assert big.collateral == 1_500_000                    # 1.5% risk / 5% stop = 30% of equity
assert g.check_open({**base, "stop_pct": 0.08}, 5_000_000, [], now)[0] is None                                                 # stop wider than max
assert g.check_open({**base, "side": "short"}, 5_000_000, [], now)[1] == "spot account: long only"
assert g.check_open({**base, "lev": 10}, 5_000_000, [], now)[0].risk_coef == 1.0                                              # no leverage ever
st["cooldowns"]["BTCTMN"] = now + 100; assert g.check_open(base, 5_000_000, [], now)[1] == "market cooldown"
pos = [dict(market=m, collateral=1_500_000, risk_coef=1) for m in ("A", "B", "C")]; assert g.check_open(base, 5_000_000, pos, now)[0] is None
two = [dict(market=m, collateral=1_650_000, risk_coef=1) for m in ("A", "B")]; s2, _ = g.check_open({**base, "market": "ETHTMN"}, 5_000_000, two, now); assert s2 and s2.collateral <= 5_000_000 * 0.97 - 3_300_000   # cash left is the cap

# --- conviction sizing: AI can only shrink the guard's size
full = g.check_open({**base, "market": "SOLTMN", "conviction": "high"}, 5_000_000, [], now)[0].exposure
assert g.check_open({**base, "market": "SOLTMN", "conviction": "medium"}, 5_000_000, [], now)[0].exposure == 1_230_000 and full == 1_650_000
assert g.check_open({**base, "market": "SOLTMN", "conviction": "low"}, 5_000_000, [], now)[0].exposure == 820_000
assert g.check_open({**base, "market": "SOLTMN", "conviction": "huge"}, 5_000_000, [], now)[0].exposure == 820_000    # unknown -> smallest

# --- close-out mode blocks every new entry
from dataclasses import replace
g2 = RiskGuard(replace(cfg, no_new_entries=True), st); assert g2.check_open({**base, "market": "XRPTMN"}, 5_000_000, [], now)[1] == "new entries disabled (close-out mode)"

# --- picky style: prompts switch and low conviction is refused in code
from bot.agents import Brain, PICKY_TASK, TRADER_TASK
assert Brain(None, None, style="picky").task is PICKY_TASK and Brain(None, None).task is TRADER_TASK

# --- drawdown halt / profit lock / daily loss
g, st = guard(); assert g.update_equity(4_700_000, now)["halt"] and st["halted"] and g.check_open(base, 4_700_000, [], now)[0] is None
g, st = guard(); assert g.update_equity(5_320_000, now)["halt"] and "profit lock" in st["halt_reason"]
g, st = guard(); g.update_equity(4_880_000, now); assert g.check_open(base, 4_880_000, [], now)[1] == "daily loss cap reached"
g, st = guard(); g.update_equity(5_160_000, now); assert g.check_open(base, 5_160_000, [dict(market="X", collateral=1_500_000, risk_coef=1)], now)[0] is None  # +3%: one position only

# --- paper broker: fees, stop, take, liquidation
pb = PaperFutures(cfg)
p = pb.open("BTCTMN", "long", 1_500_000, 2, 100.0, 98.0, 103.0, now); assert pb.cash == 3_500_000
assert pb.sync({"BTCTMN": (99.0, 99.1)}, now) == []                                         # in range
c = pb.sync({"BTCTMN": (97.9, 98.0)}, now); assert c and c[0]["close_reason"] == "stop_loss"
exp = 3_000_000; expect_pnl = exp / 100 * (97.9 - 100) - exp * (cfg.open_fee + cfg.trade_fee) - exp * cfg.trade_fee   # gap through the stop fills at the market (97.9)
assert abs(c[0]["profit"] - expect_pnl) < 1, (c[0]["profit"], expect_pnl)
assert abs(pb.cash - (3_500_000 + 1_500_000 + expect_pnl)) < 1
pb.open("ETHTMN", "short", 1_500_000, 2, 100.0, 101.0, 97.0, now); c = pb.sync({"ETHTMN": (96.0, 96.5)}, now); assert c[0]["close_reason"] == "take_profit" and c[0]["profit"] > 0
pb2 = PaperFutures(cfg); pb2.open("SOLTMN", "long", 1_500_000, 3, 100.0, 50.0, 200.0, now)    # stop far away -> liquidation fires first
assert pb2.sync({"SOLTMN": (70.0, 70.1)}, now)[0]["close_reason"] == "liquidated"
pb3 = PaperFutures(cfg); pb3.open("A", "long", 1_500_000, 2, 100.0, 90.0, 120.0, now); eq0 = pb3.equity({"A": (100.0, 100.0)}); assert eq0 < 5_000_000   # fees visible immediately

# --- full pipeline with a fake LLM: open -> protection -> unwind -> flat
class FakeLLM:
    def decide(self, system, user):
        if "independent risk reviewer" in system: return {"approve": True, "reason": "ok"}
        if "market regime" in system: return {"regime": "up", "summary": "fake"}
        if "tune bounded" in system: return {"min_score": 1.0, "max_lev": 2, "note": "n"}
        import json; ctx = json.loads(user); c = ctx["candidates"][0] if ctx["candidates"] else None
        return {"opens": [{"market": c["market"], "side": c["side"], "lev": 2, "reason": "fake"}] if c else [], "closes": [], "note": ""}
bot = Bot(cfg, llm=FakeLLM()); bot.last_quotes = {"BTCTMN": (99.9, 100.1)}
fake_c = [{"market": "BTCTMN", "side": "long", "z": 2, "entry": 100.1, "stop_pct": 0.02, "take_pct": 0.04}]
bot.st["day_start_equity"] = 5_000_000; bot.st["start_equity"] = 5_000_000; bot.st["peak"] = 5_000_000; bot.st["day"] = int((now + 3.5 * 3600) // 86400)
bot.try_open({"market": "BTCTMN", "side": "long", "lev": 2, "reason": "t"}, fake_c, [], 5_000_000, {}, now)
assert len(bot.broker.positions()) == 1
lines = open(os.path.join(tmp, "decisions.jsonl")).read(); assert "\"kind\": \"open\"" in lines and "\"approve\": true" in lines
bot.cfg.challenge_end = now + 20 * 60; bot.tick  # unwind phase -> positions closed via tick
import bot.runner as R; R.quotes_for = lambda w, u: {"BTCTMN": (99.9, 100.1)}; R.quote_one = lambda w, m: (99.9, 100.1)
bot.tick(now); assert not bot.broker.positions(), "unwind must flatten"
assert '"why": "end_game_unwind"' in open(os.path.join(tmp, "decisions.jsonl")).read()
open(cfg.kill_file, "w").write("x"); bot.broker.open("BTCTMN", "long", 1_500_000, 2, 100.0, 98, 103, now); bot.cfg.challenge_end = now + 40 * 3600; bot.tick(now); assert not bot.broker.positions()

# --- live spot sell-off on a market without MARKET orders: escalating limits, partial fills, never a MARKET order
from bot.futures import SpotLive
class FakeW:
    def __init__(s): s.orders, s.sent, s.n = {}, [], 0
    def markets(s): return {"PEPETMN": {"stepSize": 0, "tickSize": 4, "minNotional": 50000, "isMarketTypeEnable": False, "stats": {"bidPrice": "1.2"}}}
    def depth(s, m): return {"bid": [{"price": 1.2}]}
    def balances(s): return {"PEPE": {"value": "1000000"}}
    def place_order(s, sym, side, typ, qty, price=None, client_id=None):
        s.sent.append((side, typ, qty, price)); s.n += 1
        fill = qty * 0.4 if s.n == 1 else qty                      # first limit fills 40%, the next one fully
        s.orders[client_id] = {"status": "FILLED" if fill == qty else "NEW", "executedQty": fill, "executedPrice": price}; return 201, {}
    def order(s, cid): return {"result": s.orders[cid]}
    def cancel_order(s, cid): s.orders[cid]["status"] = "CANCELED"
fw = FakeW(); sl = SpotLive(fw, cfg, path=os.path.join(tmp, "sp.json"))
q, px = sl.sell_all("PEPETMN", 1_000_000, 1.2)
assert abs(q - 1_000_000) < 1 and all(t == "LIMIT" for _, t, _, _ in fw.sent) and len(fw.sent) == 2, fw.sent
assert fw.sent[1][3] < fw.sent[0][3], "second attempt must be priced lower (more aggressive)"
print("bot tests passed")
