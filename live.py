"""Live loop: REST candles for trend context + websocket for real-time price/flow/book.
   python live.py BTCUSDT            -> PAPER mode: virtual portfolio, real prices, no orders
   python live.py BTCUSDT --live     -> sends real MARKET orders (needs WALLEX_API_KEY)
Edit STRATEGY / RISK / timings after reading the competition rules."""
import json
import sys
import time

import pandas as pd

from agent.broker import PaperBroker, WallexBroker
from agent.data import _clean
from agent.features import summarize
from agent.risk import Risk, RiskConfig
from agent.strategy import SmaTrend
from agent.wallex_client import Wallex
from agent.ws_feed import PriceFeed

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
LIVE = "--live" in sys.argv
RES = "60"                 # candle resolution for trend context (minutes)
CANDLE_REFRESH = 300       # seconds between REST candle refreshes
DECIDE_EVERY = 30          # seconds between strategy decisions (cost control for LLM)
STALE_AFTER = 60           # skip trading if the websocket feed is older than this
STRATEGY = SmaTrend()      # tonight: LLMStrategy(LLM(provider="boxapi"), every=1)
RISK = Risk(RiskConfig())
START_CASH, FEE, SLIP = 1000.0, 0.0013, 0.0005   # paper only; FEE = USDT-market taker level 1 (wallex.ir/commission)

w = Wallex()
feed = PriceFeed([SYMBOL])
feed.start()
for _ in range(100):            # wait up to 10s for the first websocket message
    if feed.updated:
        break
    time.sleep(0.1)
paper = PaperBroker(START_CASH, FEE, SLIP)
live_broker = WallexBroker(w, SYMBOL, live=True) if LIVE else None
quote, basea = ("USDT", SYMBOL[:-4]) if SYMBOL.endswith("USDT") else ("TMN", SYMBOL[:-3])

df, last_fetch, bar = None, 0, 0
print(f"{'LIVE' if LIVE else 'PAPER'} mode on {SYMBOL}", flush=True)
while True:
    now = time.time()
    if df is None or now - last_fetch > CANDLE_REFRESH:
        c = w.candles(SYMBOL, RES, int(now) - 100 * 3600, int(now))
        df, last_fetch = _clean(pd.DataFrame({k: c[k] for k in "tohlcv"})), now
    micro = feed.snapshot(SYMBOL)
    price = micro.get("last") or float(df["c"].iloc[-1])
    age = micro.get("ws_age_s")
    if age is None or age > STALE_AFTER:
        print(json.dumps({"ts": int(now), "warn": "ws feed stale/not connected, skipping"}), flush=True)
        time.sleep(DECIDE_EVERY)
        continue
    # candles give trend context; replace last close with the live price
    feats = {**summarize(df.assign(c=df["c"].where(df.index < len(df) - 1, price))), **micro}
    if LIVE:
        cash, base = w.free(quote), w.free(basea)
    else:
        cash, base = paper.cash, paper.base
    equity = cash + base * price
    state = {"cash": round(cash, 2), "position_value": round(base * price, 2), "equity": round(equity, 2)}
    act = STRATEGY.decide(feats, state)
    order = RISK.check(act, equity, base, price, bar)
    if order:
        res = live_broker.market(order[0], order[1], price) if LIVE else paper.market(order[0], order[1], price)
    else:
        res = None
    print(json.dumps({"ts": int(now), "price": price, "equity": round(equity, 2), "act": act,
                      "order": order, "result": str(res) if res else None,
                      "micro": {k: micro[k] for k in micro if k != "last"}}), flush=True)
    bar += 1
    time.sleep(DECIDE_EVERY)
