"""python run_backtest.py [SYMBOL] [days]   (public data only, no keys)"""
import sys

from agent.backtest import run
from agent.data import load_candles
from agent.strategy import Hold, SmaTrend
from agent.wallex_client import Wallex

sym = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
df = load_candles(Wallex(), sym, "60", days)
print(f"{sym}: {len(df)} hourly candles")
for name, s in [("hold", Hold()), ("sma_trend", SmaTrend())]:
    print(name, run(df, s))
