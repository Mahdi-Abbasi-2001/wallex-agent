from dataclasses import dataclass


@dataclass
class Order:
    sym: str
    side: str                 # 'buy' | 'sell'
    kind: str = "market"      # 'market' | 'limit' | 'stop' | 'cancel_all'
    qty: float = 0.0          # base units ...
    notional: float = 0.0     # ... or USD (converted at execution price)
    price: float = 0.0        # limit price
    stop: float = 0.0         # stop trigger
    ttl: int = 1              # bars a limit/stop stays alive (stops: use a big number)
    tag: str = ""
