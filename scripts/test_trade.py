"""One minimum-size round trip through the bot's own order code, to verify order/fill fields before any strategy runs.
 python scripts/test_trade.py            -> shows what it would do (no orders)
 python scripts/test_trade.py --execute  -> buys ~60,000 TMN of USDTTMN, sells it back, prints raw order responses"""
import json
import sys
import time

sys.path.insert(0, ".")
from agent.llm import load_env

load_env()
from agent.wallex_client import Wallex
from bot.config import Config
from bot.futures import SpotLive

MARKET, AMOUNT = "USDTTMN", 60_000
w = Wallex(); sl = SpotLive(w, Config(), path="logs/test_trade_positions.json")
st = sl.meta[MARKET]["stats"]; bid, ask = float(st["bidPrice"]), float(st["askPrice"])
print(f"{MARKET} bid {bid} ask {ask} stepSize {sl.meta[MARKET]['stepSize']} minNotional {sl.meta[MARKET]['minNotional']}")
print(f"plan: BUY ~{AMOUNT:,} TMN at {ask} (qty {sl._qty(MARKET, AMOUNT / ask)}), then SELL it back at the bid")
if "--execute" not in sys.argv:
    sys.exit("dry: add --execute to place the two orders")

orig = sl._wait
def logged(cid, secs=12):
    o = orig(cid, secs); print("ORDER", cid, json.dumps(o, ensure_ascii=False)); return o
sl._wait = logged

p = sl.open(MARKET, "long", AMOUNT, 1, ask, 0, 0)
print("POSITION", p)
print("balances after buy", {k: v["value"] for k, v in w.balances().items() if float(v["value"] or 0) > 0})
time.sleep(2)
c = sl.close(p["id"], float(w.markets()[MARKET]["stats"]["bidPrice"]))
print("CLOSED", {k: c[k] for k in ("entry", "qty", "close_price", "profit")})
print("balances after sell", {k: v["value"] for k, v in w.balances().items() if float(v["value"] or 0) > 0})
