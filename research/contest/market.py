import numpy as np


class Market:
    """Aligned multi-asset candle replay. bars: {symbol: DataFrame(t,o,h,l,c,v)}, tf_seconds: bar length."""

    def __init__(self, bars, tf_seconds):
        self.symbols = list(bars)
        idx = sorted(set.intersection(*[set(d["t"]) for d in bars.values()]))
        self.t = np.array(idx)
        mat = lambda k: np.column_stack([bars[s].set_index("t").loc[idx, k].to_numpy(float) for s in self.symbols])
        self.O, self.H, self.L, self.C, self.V = (mat(k) for k in "ohlcv")
        self.sidx = {s: j for j, s in enumerate(self.symbols)}
        self.tf = tf_seconds
        self.bpy = 365 * 86400 / tf_seconds

    def __len__(self):
        return len(self.t)


class Ctx:
    """What an agent sees after bar i has closed (never anything later)."""

    def __init__(self, mkt, acct, i, start_i):
        self.m, self.acct, self.i, self.start_i = mkt, acct, i, start_i

    def _col(self, arr, sym, n):
        j = self.m.sidx[sym]; return arr[max(0, self.i - n + 1):self.i + 1, j]
    def closes(self, sym, n): return self._col(self.m.C, sym, n)
    def highs(self, sym, n): return self._col(self.m.H, sym, n)
    def lows(self, sym, n): return self._col(self.m.L, sym, n)
    def opens(self, sym, n): return self._col(self.m.O, sym, n)
    def volumes(self, sym, n): return self._col(self.m.V, sym, n)
    def price(self, sym): return float(self.m.C[self.i, self.m.sidx[sym]])
    @property
    def symbols(self): return self.m.symbols
    @property
    def prices(self): return self.m.C[self.i]
    @property
    def equity(self): return self.acct.equity(self.prices)
    @property
    def bars_elapsed(self): return self.i - self.start_i + 1
    def weights(self):
        e = self.equity; return {s: float(self.acct.q[j] * self.m.C[self.i, j] / e) if e > 0 else 0.0 for s, j in self.m.sidx.items()}

    def position_return(self, sym):
        """Unrealised return of the open position vs its average entry (positive = in profit). None if flat."""
        j = self.m.sidx[sym]; q = self.acct.q[j]
        if abs(q) < 1e-12 or self.acct.avg[j] <= 0: return None
        return float(np.sign(q) * (self.m.C[self.i, j] / self.acct.avg[j] - 1))
