import numpy as np


class Account:
    """Cash + signed base quantities. Spot (leverage=1, no shorting) by default; optional margin/shorting."""

    def __init__(self, symbols, cash=1000.0, leverage=1.0, allow_short=False, maint=0.05):
        self.symbols = symbols; self.q = np.zeros(len(symbols)); self.cash = float(cash); self.start_cash = float(cash)
        self.leverage, self.allow_short, self.maint = leverage, allow_short, maint
        self.fees = 0.0; self.turnover = 0.0; self.n_fills = 0; self.liquidated = False; self.avg = np.zeros(len(symbols))   # average entry price of the open position

    def equity(self, prices): return self.cash + float((self.q * prices).sum())
    def gross(self, prices): return float(np.abs(self.q * prices).sum())

    def max_qty(self, j, sign, price, prices, fee_rate):
        """Largest quantity we may trade without breaking cash / leverage / shorting rules."""
        if sign < 0 and not self.allow_short: return max(float(self.q[j]), 0.0)
        eq = self.equity(prices); others = self.gross(prices) - abs(self.q[j] * prices[j])
        def ok(x):
            new_gross = others + abs((self.q[j] + sign * x) * price); eq_after = eq - x * price * fee_rate
            return new_gross <= self.leverage * eq_after + 1e-9 and (self.leverage > 1 or self.allow_short or sign < 0 or x * price * (1 + fee_rate) <= self.cash + 1e-9)
        hi = (self.leverage * eq + abs(self.q[j] * price)) / price + 1.0
        if ok(hi): return hi
        lo = 0.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if ok(mid): lo = mid
            else: hi = mid
        return lo

    def fill(self, j, sign, qty, price, fee_rate):
        notional = qty * price; fee = notional * fee_rate
        q0 = self.q[j]; self.cash -= sign * notional + fee; self.q[j] += sign * qty; self.fees += fee; self.turnover += notional; self.n_fills += 1
        if q0 * sign >= 0: self.avg[j] = (abs(q0) * self.avg[j] + qty * price) / (abs(q0) + qty) if abs(q0) + qty > 0 else price   # adding to (or opening) a position
        elif abs(self.q[j]) < 1e-12: self.avg[j] = 0.0
        elif q0 * self.q[j] < 0: self.avg[j] = price                                                                   # flipped through zero
