"""Deterministic guardrails between any strategy (incl. LLM) and the broker."""
from dataclasses import dataclass


@dataclass
class RiskConfig:
    max_position_frac: float = 0.5     # max share of equity held in the asset
    max_trade_frac: float = 0.25       # max share of equity per single trade
    max_drawdown: float = 0.15         # kill switch: stop trading after this peak-to-trough loss
    min_notional: float = 5.0          # exchange minimum order value (check /markets)
    amount_precision: int = 6          # decimals allowed in quantity
    cooldown_bars: int = 1             # min bars between trades


class Risk:
    def __init__(self, cfg: RiskConfig = None):
        self.cfg = cfg or RiskConfig()
        self.peak = 0.0
        self.halted = False
        self.last_trade_bar = -10**9

    def check(self, action, equity, base_qty, price, bar):
        """action: dict {action, fraction}. Returns (side, qty) or None. Never trusts input."""
        c = self.cfg
        self.peak = max(self.peak, equity)
        if self.peak and (self.peak - equity) / self.peak >= c.max_drawdown:
            self.halted = True
        a = str(action.get("action", "hold")).lower() if isinstance(action, dict) else "hold"
        try:
            frac = min(max(float(action.get("fraction", 0)), 0.0), 1.0)
        except (TypeError, ValueError):
            return None
        if a not in ("buy", "sell") or frac <= 0 or bar - self.last_trade_bar < c.cooldown_bars:
            return None
        if self.halted and a == "buy":
            return None  # allow exits when halted, block new risk
        if a == "buy":
            room = c.max_position_frac * equity - base_qty * price
            notional = min(frac * c.max_trade_frac * equity, room)
            qty = notional / price
        else:
            qty = base_qty * frac
        qty = round(qty, c.amount_precision)
        if qty * price < c.min_notional:
            return None
        self.last_trade_bar = bar
        return a.upper(), qty
