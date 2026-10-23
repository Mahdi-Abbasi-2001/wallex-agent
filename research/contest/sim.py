from dataclasses import dataclass, field

import numpy as np

from .account import Account
from .fills import MidFill
from .market import Ctx
from .scoring import metrics


@dataclass
class SimConfig:
    start_cash: float = 1000.0
    taker: float = 0.0013            # market / stop orders
    maker: float = 0.0010            # limit orders
    fill: object = field(default_factory=MidFill)
    leverage: float = 1.0
    allow_short: bool = False
    limit_rule: str = "touch"        # 'touch' (low <= limit) or 'through' (low < limit)
    maint: float = 0.05              # liquidation when equity < maint * gross exposure (margin accounts)
    borrow_daily: float = 0.0        # daily cost on borrowed cash / short value (margin accounts)


@dataclass
class Result:
    eq: np.ndarray; ret: float; mdd: float; sharpe: float; calmar: float; fees: float; turnover: float; fills: int; liquidated: bool


def run_sim(mkt, agent, start, end, cfg=None):
    """Agent decides after bar i closes; its orders execute during bar i+1. Window = bars [start, end). Equity marked at each close."""
    cfg = cfg or SimConfig(); acct = Account(mkt.symbols, cfg.start_cash, cfg.leverage, cfg.allow_short, cfg.maint)
    agent.reset(); pending = []; eq = []
    pending += agent.on_bar(Ctx(mkt, acct, start - 1, start)) or []
    for i in range(start, end):
        o, h, l, c = mkt.O[i], mkt.H[i], mkt.L[i], mkt.C[i]; prices_open = o
        nxt = []
        for od in pending:
            if od.kind == "cancel_all":
                nxt = [x for x in nxt if od.sym and x.sym != od.sym]; continue
            j = mkt.sidx[od.sym]; sign = 1 if od.side == "buy" else -1; ref = None; rate = cfg.taker; kind = od.kind
            if kind == "market": ref = o[j]
            elif kind == "limit":
                touched = (l[j] <= od.price) if sign > 0 else (h[j] >= od.price)
                if cfg.limit_rule == "through": touched = (l[j] < od.price) if sign > 0 else (h[j] > od.price)
                if touched: ref = min(od.price, o[j]) if sign > 0 else max(od.price, o[j]); rate = cfg.maker
            elif kind == "stop":
                trig = (l[j] <= od.stop) if sign < 0 else (h[j] >= od.stop)
                if trig: ref = min(od.stop, o[j]) if sign < 0 else max(od.stop, o[j])
            if ref is None:
                od.ttl -= 1
                if od.ttl > 0: nxt.append(od)
                continue
            qty = od.qty if od.qty > 0 else od.notional / ref
            qty = min(qty, acct.max_qty(j, sign, ref, o, rate))
            if qty <= 1e-12: continue
            if kind == "limit": px, q = ref, qty
            else: px, q = cfg.fill.market(od.sym, sign, qty, ref)
            if q > 1e-12: acct.fill(j, sign, q, px, rate)
        pending = nxt
        eqt = acct.equity(c)
        if (cfg.leverage > 1 or cfg.allow_short):
            borrowed = max(0.0, -acct.cash) + float(np.clip(-acct.q, 0, None) @ c)
            acct.cash -= borrowed * cfg.borrow_daily * mkt.tf / 86400; eqt = acct.equity(c)
            if eqt <= cfg.maint * acct.gross(c) and acct.gross(c) > 0:
                for j in range(len(acct.q)):
                    if abs(acct.q[j]) > 1e-12: acct.fill(j, -1 if acct.q[j] > 0 else 1, abs(acct.q[j]), c[j], cfg.taker)
                acct.liquidated = True; pending = []; eqt = acct.equity(c)
        eq.append(max(eqt, 0.0))
        if i < end - 1 and not (acct.liquidated and eqt <= 0): pending += agent.on_bar(Ctx(mkt, acct, i, start)) or []
    m = metrics(eq, cfg.start_cash, mkt.bpy)
    return Result(np.array(eq), m["ret"], m["mdd"], m["sharpe"], m["calmar"], acct.fees, acct.turnover, acct.n_fills, acct.liquidated)
