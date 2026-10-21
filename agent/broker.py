"""Brokers share one interface so strategies/risk never care if it's paper or live."""
from dataclasses import dataclass, field


@dataclass
class Fill:
    side: str
    qty: float
    price: float
    fee: float


class PaperBroker:
    """Long-only spot simulator. Fills at given price +/- slippage, charges fee on quote."""

    def __init__(self, cash=1000.0, fee=0.0013, slippage=0.0005):
        self.cash, self.base = cash, 0.0
        self.fee, self.slip = fee, slippage
        self.fills: list[Fill] = []

    def equity(self, price):
        return self.cash + self.base * price

    def market(self, side, qty, price):
        if qty <= 0:
            return None
        if side == "BUY":
            p = price * (1 + self.slip)
            cost = qty * p
            fee = cost * self.fee
            if cost + fee > self.cash:
                qty = self.cash / (p * (1 + self.fee))
                cost, fee = qty * p, qty * p * self.fee
            if qty <= 0:
                return None
            self.cash -= cost + fee
            self.base += qty
        else:
            qty = min(qty, self.base)
            if qty <= 0:
                return None
            p = price * (1 - self.slip)
            fee = qty * p * self.fee
            self.cash += qty * p - fee
            self.base -= qty
        f = Fill(side, qty, p, fee)
        self.fills.append(f)
        return f


class WallexBroker:
    """Live broker wrapping agent.wallex_client.Wallex. Disabled unless live=True."""

    def __init__(self, client, symbol, live=False):
        self.c, self.symbol, self.live = client, symbol, live

    def market(self, side, qty, price):
        if not self.live:
            raise RuntimeError("WallexBroker is in dry mode; pass live=True to send real orders")
        return self.c.place_order(self.symbol, side, "MARKET", qty)
