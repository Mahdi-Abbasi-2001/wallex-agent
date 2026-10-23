"""Fill models: how a MARKET order executes against the (simulated) book. Returns (avg_price, filled_qty).
 Defaults for spreads/depth are rough guesses from a few Wallex measurements (BTCUSDT: spread 0.11-0.24%, ~$1-7k in the top 10 levels). Override them."""
DEFAULT_SPREAD_BPS = {"BTCUSDT": 15, "ETHUSDT": 25, "PAXGUSDT": 25}
DEFAULT_DEPTH_USD = {"BTCUSDT": 4000, "ETHUSDT": 2000, "PAXGUSDT": 1500}     # top-10 levels, per side
OTHER_SPREAD_BPS, OTHER_DEPTH_USD = 40, 800


class MidFill:
    """Fills at the reference price in any size (no spread, no impact). Fees still apply in the account."""
    name = "mid"
    def market(self, sym, sign, qty, ref):
        return ref, qty


class BidAskFill:
    """Fills at the touch: ref +/- half the spread, any size."""
    def __init__(self, spread_bps=None, default_bps=OTHER_SPREAD_BPS):
        self.sp = {**DEFAULT_SPREAD_BPS, **(spread_bps or {})}; self.default = default_bps; self.name = "bid/ask"
    def market(self, sym, sign, qty, ref):
        h = self.sp.get(sym, self.default) / 2e4; return ref * (1 + sign * h), qty


class BookFill:
    """Walks a synthetic order book: 10 levels, each holding depth/10 USD, spaced `step_bps` apart starting at half the spread.
    Wallex rule: a market order that would move price more than `cap_pct` is cancelled (partial=False) or truncated (partial=True)."""
    def __init__(self, spread_bps=None, depth_usd=None, depth_scale=1.0, step_bps=5.0, levels=10, cap_pct=2.0, partial=True):
        self.sp = {**DEFAULT_SPREAD_BPS, **(spread_bps or {})}; self.dp = {**DEFAULT_DEPTH_USD, **(depth_usd or {})}
        self.scale, self.step, self.levels, self.cap, self.partial = depth_scale, step_bps, levels, cap_pct, partial; self.name = "book"
    def market(self, sym, sign, qty, ref):
        half = self.sp.get(sym, OTHER_SPREAD_BPS) / 2.0; level_qty = self.dp.get(sym, OTHER_DEPTH_USD) * self.scale / self.levels / ref
        got_qty = cost = 0.0
        for k in range(self.levels):
            off = half + k * self.step                       # bps away from the reference price
            if off / 100.0 > self.cap: break
            take = min(level_qty, qty - got_qty)
            if take <= 0: break
            got_qty += take; cost += take * ref * (1 + sign * off / 1e4)
        if got_qty <= 0 or (not self.partial and got_qty < qty * 0.999): return ref, 0.0
        return cost / got_qty, got_qty
