"""Public-endpoint smoke test; no keys needed. Run: python smoke_test.py"""
import time
from agent.wallex_client import Wallex

w = Wallex()
m = w.markets()
print("markets:", len(m), "e.g.", list(m)[:5])
d = w.depth("BTCUSDT")
print("depth bid/ask levels:", len(d["bid"]), len(d["ask"]))
now = int(time.time())
c = w.candles("BTCUSDT", "60", now - 86400 * 3, now)
print("candles:", len(c.get("t", [])), "status", c.get("s"))
