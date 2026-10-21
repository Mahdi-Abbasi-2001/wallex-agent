"""Agents: on_bar(ctx) -> list[Order]. Orders placed after bar i closes execute during bar i+1.
Portfolio-style agents return target weights and use Agent.rebalance(). An LLM agent plugs in the same way (see LLMAgent)."""
import numpy as np
import pandas as pd

from .orders import Order


def _ema_last(x, n): return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()
def _sma_last(x, n): return float(np.mean(x[-n:])) if len(x) >= n else np.nan


class Agent:
    name = "agent"
    def reset(self): pass
    def on_bar(self, ctx): return []

    @staticmethod
    def rebalance(ctx, target, min_trade=0.02):
        """Market orders (sells first) to move current weights to `target` ({sym: weight}, weights of equity; negative = short)."""
        cur = ctx.weights(); eq = ctx.equity; sells, buys = [], []
        for s in ctx.symbols:
            d = target.get(s, 0.0) - cur.get(s, 0.0)
            if abs(d) > min_trade: (buys if d > 0 else sells).append(Order(s, "buy" if d > 0 else "sell", "market", notional=abs(d) * eq))
        return sells + buys


class Cash(Agent):
    name = "cash"


class HoldAll(Agent):
    def __init__(self, syms=None, weight=1.0, name="hold"):
        self.syms, self.weight, self.name, self.done = syms, weight, name, False
    def reset(self): self.done = False
    def on_bar(self, ctx):
        if self.done: return []
        self.done = True; syms = self.syms or ctx.symbols
        return self.rebalance(ctx, {s: self.weight / len(syms) for s in syms})


# ----- trend ensemble (15 generic daily rules, position = share of rules that are long, equal capital per coin) -----
def _donchian(h, l, c, N, M):
    p = 0
    for i in range(N, len(c)):
        if p == 0 and c[i] > h[i - N:i].max(): p = 1
        elif p == 1 and c[i] < l[i - M:i].min(): p = 0
    return p

def trend_votes(c, h, l):
    """Number of the 15 rules that are long. Needs ~260 bars of history (otherwise fewer rules vote)."""
    v = 0; n = len(c)
    for f, s in ((10, 50), (20, 100), (50, 200)):
        if n >= s and _sma_last(c, f) > _sma_last(c, s): v += 1
    for k in (50, 100, 200):
        if n >= k and c[-1] > _sma_last(c, k): v += 1
    for k in (40, 80, 160):
        if n > k and c[-1] / c[-1 - k] > 1: v += 1
    for a, b, s in ((12, 26, 9), (24, 52, 18), (48, 104, 36)):
        if n >= b + s:
            m = _ema_last(c, a) - _ema_last(c, b); v += int(m[-1] > _ema_last(m, s)[-1])
    for N, M in ((20, 10), (40, 20), (80, 40)):
        if n > N + 5: v += _donchian(h, l, c, N, M)
    return v


class TrendEnsemble(Agent):
    def __init__(self, syms=None, name="trend ensemble (15 rules)", min_trade=0.03): self.syms, self.name, self.mt = syms, name, min_trade
    def on_bar(self, ctx):
        syms = self.syms or ctx.symbols; tgt = {s: trend_votes(ctx.closes(s, 260), ctx.highs(s, 260), ctx.lows(s, 260)) / 15.0 / len(syms) for s in syms}
        return self.rebalance(ctx, tgt, self.mt)


class Rotation(Agent):
    """Every `rb` bars hold the top-k coins by trailing L-bar return (optionally only if positive), equal weight, times `lev`."""
    def __init__(self, L=20, k=3, rb=7, absolute=False, lev=1.0, name=None):
        self.L, self.k, self.rb, self.absolute, self.lev = L, k, rb, absolute, lev
        self.name = name or f"top-{k} momentum {L}d" + (", cash if <0" if absolute else "") + (f" x{lev:g}" if lev != 1 else "")
    def on_bar(self, ctx):
        if ctx.bars_elapsed % self.rb != 0: return []
        mom = {}
        for s in ctx.symbols:
            c = ctx.closes(s, self.L + 1)
            if len(c) > self.L: mom[s] = c[-1] / c[0] - 1
        sel = [s for s in sorted(mom, key=mom.get, reverse=True)[:self.k] if (mom[s] > 0 or not self.absolute)]
        return self.rebalance(ctx, {s: self.lev / self.k for s in sel})


class RandomHolder(Agent):
    def __init__(self, k, seed, weight=1.0): self.k, self.seed, self.weight, self.name, self.done = k, seed, weight, "random holder", False
    def reset(self): self.done = False; self.rng = np.random.default_rng(self.seed)
    def on_bar(self, ctx):
        if self.done: return []
        self.done = True; sel = list(self.rng.choice(ctx.symbols, self.k, replace=False)); return self.rebalance(ctx, {s: self.weight / self.k for s in sel})


class RandomTrader(Agent):
    def __init__(self, k, period, seed): self.k, self.period, self.seed, self.name = k, period, seed, "random trader"
    def reset(self): self.rng = np.random.default_rng(self.seed)
    def on_bar(self, ctx):
        if ctx.bars_elapsed % self.period != 0: return []
        sel = list(self.rng.choice(ctx.symbols, self.k, replace=False)); return self.rebalance(ctx, {s: 1.0 / self.k for s in sel})


class SmaTrader(Agent):
    """Field member: a random handful of coins, each held while above its own SMA(n)."""
    def __init__(self, k, n, seed): self.k, self.n, self.seed, self.name = k, n, seed, "SMA trader"
    def reset(self): self.rng = np.random.default_rng(self.seed); self.sel = None
    def on_bar(self, ctx):
        if self.sel is None: self.sel = list(self.rng.choice(ctx.symbols, self.k, replace=False))
        t = {s: (1.0 / self.k if ctx.price(s) > _sma_last(ctx.closes(s, self.n), self.n) else 0.0) for s in self.sel}
        return self.rebalance(ctx, t, 0.05)


class LLMAgent(Agent):
    """Skeleton for the real contest agent: gather features per coin -> ask the LLM judge (JSON schema) -> risk engine clips
    weights/leverage -> return Orders. Fall back to `baseline` if the LLM fails or times out."""
    name = "llm agent"
    def __init__(self, baseline, judge=None): self.baseline, self.judge = baseline, judge
    def reset(self): self.baseline.reset()
    def on_bar(self, ctx):
        orders = self.baseline.on_bar(ctx)
        return orders      # TODO: pass proposed orders through self.judge(...) and the risk engine


class Concentrated(Agent):
    """Pick the top-k coins by trailing L-bar momentum (optionally momentum x volatility), optionally short the bottom short_k,
    optionally only when `gate` coin is above its SMA(gate_n) (else flat). Re-picks every `rb` bars. lev scales total gross per side."""
    def __init__(self, L=14, k=1, rb=7, lev=1.0, short_k=0, pick="momentum", gate=None, gate_n=50, stop=None, short_only=False, name=None):
        self.L, self.k, self.rb, self.lev, self.short_k, self.pick, self.gate, self.gate_n, self.stop, self.short_only = L, k, rb, lev, short_k, pick, gate, gate_n, stop, short_only
        self.name = name or ("SHORT bottom-%d" % max(k, 1) if short_only else f"top-{k}") + f" {pick} {L}d rb{rb}" + (f" + SHORT bottom-{short_k}" if short_k and not short_only else "") + (f" x{lev:g}" if lev != 1 else "") + (" gated" if gate else "") + (f" stop{stop*100:g}%" if stop else "")
    def score(self, ctx):
        out = {}
        for s in ctx.symbols:
            c = ctx.closes(s, self.L + 1)
            if len(c) <= self.L: continue
            m = c[-1] / c[0] - 1
            if self.pick == "vol_mom": m = m * float(np.std(np.diff(c) / c[:-1]))
            out[s] = m
        return out
    def on_bar(self, ctx):
        orders, stopped = [], set()
        if self.stop:                                              # close any position that is down `stop` vs its entry (checked on bar closes)
            for s in ctx.symbols:
                pr = ctx.position_return(s)
                if pr is not None and pr <= -self.stop:
                    q = ctx.acct.q[ctx.m.sidx[s]]; orders.append(Order(s, "sell" if q > 0 else "buy", "market", qty=abs(q))); stopped.add(s)
        if ctx.bars_elapsed % self.rb != 0: return orders
        tgt = {}
        flat = False
        if self.gate:
            g = ctx.closes(self.gate, self.gate_n); flat = len(g) >= self.gate_n and g[-1] < g.mean()
        if not flat:
            sc = self.score(ctx); order = sorted(sc, key=sc.get, reverse=True)
            both = self.short_k > 0 and not self.short_only; per = self.lev / 2.0 if both else self.lev      # long+short: leverage is split so gross exposure == lev
            if not self.short_only:
                for s in order[:self.k]:
                    if sc[s] > 0: tgt[s] = per / self.k
            if both:
                for s in order[::-1][:self.short_k]:
                    if sc[s] < 0 and s not in tgt: tgt[s] = -per / self.short_k
            elif self.short_only:
                for s in order[::-1][:self.k]:
                    if sc[s] < 0: tgt[s] = -self.lev / self.k
        return orders + [o for o in self.rebalance(ctx, tgt) if o.sym not in stopped]


class YOLO(Agent):
    """Field member: one random coin, max leverage, random direction (if shorting is allowed), held for the whole window."""
    def __init__(self, lev, seed, can_short): self.lev, self.seed, self.can_short, self.name, self.done = lev, seed, can_short, "yolo", False
    def reset(self): self.done = False; self.rng = np.random.default_rng(self.seed)
    def on_bar(self, ctx):
        if self.done: return []
        self.done = True; s = str(self.rng.choice(ctx.symbols)); d = -1.0 if (self.can_short and self.rng.random() < 0.4) else 1.0
        return self.rebalance(ctx, {s: d * self.lev})


class ActiveGuess(Agent):
    """Field member imitating a busy LLM agent: re-picks 1-3 coins every 1-3 bars, random direction if shorting is allowed, random leverage."""
    def __init__(self, seed, can_short, max_lev): self.seed, self.can_short, self.max_lev, self.name = seed, can_short, max_lev, "active guess"
    def reset(self): self.rng = np.random.default_rng(self.seed); self.period = int(self.rng.integers(1, 4)); self.lev = float(self.rng.uniform(1, self.max_lev)); self.k = int(self.rng.integers(1, 4))
    def on_bar(self, ctx):
        if ctx.bars_elapsed % self.period != 0: return []
        sel = list(self.rng.choice(ctx.symbols, self.k, replace=False)); tgt = {}
        for s in sel: tgt[s] = (-1.0 if (self.can_short and self.rng.random() < 0.4) else 1.0) * self.lev / self.k
        return self.rebalance(ctx, tgt)


class RegimeSwitch(Agent):
    """Regime = gate coin (BTC) above/below its SMA(gate_n). Up regime: long the top-k momentum coins. Down regime: short the bottom-k
    ('short') or stay in cash ('flat'). Re-picks every rb bars; lev is total gross exposure."""
    def __init__(self, L=14, k=1, rb=7, lev=3.0, gate="BTCUSDT", gate_n=50, down="short", name=None):
        self.L, self.k, self.rb, self.lev, self.gate, self.gate_n, self.down = L, k, rb, lev, gate, gate_n, down
        self.name = name or f"regime(BTC>SMA{gate_n}) long top-{k} / {'SHORT bottom-%d' % k if down == 'short' else 'cash'} {L}d rb{rb} x{lev:g}"
    def on_bar(self, ctx):
        if ctx.bars_elapsed % self.rb != 0: return []
        g = ctx.closes(self.gate, self.gate_n)
        if len(g) < self.gate_n: return self.rebalance(ctx, {})
        up = g[-1] >= g.mean(); sc = {}
        for s in ctx.symbols:
            c = ctx.closes(s, self.L + 1)
            if len(c) > self.L: sc[s] = c[-1] / c[0] - 1
        order = sorted(sc, key=sc.get, reverse=True); tgt = {}
        if up:
            for s in order[:self.k]: tgt[s] = self.lev / self.k
        elif self.down == "short":
            for s in order[::-1][:self.k]: tgt[s] = -self.lev / self.k
        return self.rebalance(ctx, tgt)


class Basket(Agent):
    """Equal-weight basket of every coin up more than `thr` over L bars and above its SMA(sma), at most kmax coins; flat when none qualify
    (and, if btc_gate, when BTC is below its SMA(sma)). Re-evaluated every `rb` bars. Spot only."""
    def __init__(self, L=1, thr=0.03, sma=20, kmax=3, btc_gate=True, rb=1, gate="BTCUSDT", name=None):
        self.L, self.thr, self.sma, self.kmax, self.btc_gate, self.rb, self.gate = L, thr, sma, kmax, btc_gate, rb, gate
        self.name = name or f"basket L{L} thr{thr*100:g}% sma{sma} kmax{kmax}" + (" btc-gated" if btc_gate else "")
    def on_bar(self, ctx):
        if ctx.bars_elapsed % self.rb != 0: return []
        if self.btc_gate:
            g = ctx.closes(self.gate, self.sma)
            if len(g) < self.sma or g[-1] < g.mean(): return self.rebalance(ctx, {})
        sc, n = {}, max(self.L, self.sma)
        for s in ctx.symbols:
            x = ctx.closes(s, n + 1)
            if len(x) <= n: continue
            m = x[-1] / x[-1 - self.L] - 1
            if m > self.thr and x[-1] > x[-self.sma:].mean(): sc[s] = m
        pick = sorted(sc, key=sc.get, reverse=True)[:self.kmax]
        return self.rebalance(ctx, {s: 1.0 / len(pick) for s in pick})
