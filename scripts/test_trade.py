"""One minimum-size round trip through the bot's own order code, to verify order/fill fields before any strategy runs.
 python scripts/test_trade.py            -> shows what it would do (no orders)
 python scripts/test_trade.py --execute  -> buys the smallest sellable amount of BTCTMN (or --market X), sells it back, prints raw order responses"""
import json
import sys
import time

sys.path.insert(0, ".")
from agent.llm import load_env

load_env()
from agent.wallex_client import Wallex
from bot.config import Config
from bot.futures import SpotLive

MARKET = sys.argv[sys.argv.index("--market") + 1] if "--market" in sys.argv else "BTCTMN"   # USDTTMN is refused for the challenge account
w = Wallex(); sl = SpotLive(w, Config(), path="logs/test_trade_positions.json")
d = w.depth(MARKET); bid, ask = float(d["bid"][0]["price"]), float(d["ask"][0]["price"])   # live book, not the lagging market list
# Smallest buy whose remainder, after a 0.3% fee possibly taken in the coin and rounding down to the step, is still above
# Wallex's minimum order value at the bid, so it can always be sold back.
step = 10 ** -int(sl.meta[MARKET]["stepSize"]); min_notional = float(sl.meta[MARKET]["minNotional"]); q = step
while sl._qty(MARKET, q * (1 - 0.003)) * bid < min_notional * 1.01 or q * ask < min_notional * 1.01: q = round(q + step, 8)
AMOUNT = q * ask
print(f"{MARKET} bid {bid} ask {ask} stepSize {sl.meta[MARKET]['stepSize']} minNotional {sl.meta[MARKET]['minNotional']}")
print(f"plan: BUY ~{AMOUNT:,.0f} TMN at {ask} (qty {sl._qty(MARKET, AMOUNT / ask)}), then SELL it back at the bid")
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
c = sl.close(p["id"], float(w.depth(MARKET)["bid"][0]["price"]))
print("CLOSED", {k: c[k] for k in ("entry", "qty", "close_price", "profit")})
print("balances after sell", {k: v["value"] for k, v in w.balances().items() if float(v["value"] or 0) > 0})
