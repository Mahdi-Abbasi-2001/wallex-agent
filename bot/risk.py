"""Hard risk guard. Pure code: the LLM can propose, only this module can approve. Every rejection returns a reason that is logged."""
import math
from dataclasses import dataclass

from .futures import MIN_STEP


@dataclass
class Sized:
    collateral: float
    risk_coef: float
    exposure: float
    stop_price: float
    take_price: float
    risk_toman: float


def phase(cfg, now):
    """trade -> no_new -> unwind -> force, driven by time left until the official end."""
    left = cfg.challenge_end - now
    if left <= 0: return "ended"
    if left <= cfg.force_min * 60: return "force"
    if left <= cfg.unwind_h * 3600: return "unwind"
    if left <= cfg.no_new_h * 3600: return "no_new"
    return "trade"


class RiskGuard:
    def __init__(self, cfg, state):
        self.cfg, self.st = cfg, state

    # ---- account level ----
    def update_equity(self, equity, now):
        st, cfg = self.st, self.cfg
        if st["start_equity"] is None: st["start_equity"] = equity or cfg.start_equity
        st["peak"] = max(st["peak"] or equity, equity)
        day = int((now + 3.5 * 3600) // 86400)                    # Tehran calendar day
        if st["day"] != day: st["day"], st["day_start_equity"] = day, equity
        ret = equity / cfg.start_equity - 1
        dd = 1 - equity / st["peak"]
        day_ret = equity / st["day_start_equity"] - 1
        action = None
        if not st["halted"]:
            if dd >= cfg.max_drawdown: action = f"drawdown {dd:.1%} >= {cfg.max_drawdown:.0%}"
            elif ret >= cfg.lock_flat_at: action = f"profit lock: up {ret:.1%} >= {cfg.lock_flat_at:.0%}"
            if action: st["halted"], st["halt_reason"] = True, action
        return {"ret": ret, "dd": dd, "day_ret": day_ret, "halt": action}

    def max_leverage(self):
        m = self.st["params"].get("max_lev") or self.cfg.max_lev
        return min(m, self.cfg.hard_lev_cap)

    # ---- order level ----
    CONVICTION = {"low": 0.5, "medium": 0.75, "high": 1.0}

    def check_open(self, p, equity, positions, now, min_exposure=None):
        """p: dict(market, side, entry, stop_pct, take_pct, lev, conviction). Returns (Sized|None, reason).
        Size = guard maximum x conviction (low 50%, medium 75%, high 100%); the AI can shrink a position, never enlarge it."""
        cfg, st = self.cfg, self.st
        ph = phase(cfg, now)
        if ph != "trade": return None, f"phase={ph}: no new positions"
        if st["halted"]: return None, f"halted: {st['halt_reason']}"
        ret = equity / cfg.start_equity - 1
        if equity / st["day_start_equity"] - 1 <= -cfg.daily_loss_cap: return None, "daily loss cap reached"
        if len(positions) >= (1 if ret >= cfg.lock_reduce_at else cfg.max_positions): return None, f"max positions ({len(positions)}), ret {ret:.1%}"
        if any(x["market"] == p["market"] for x in positions): return None, "already positioned in this market"
        if now < st["cooldowns"].get(p["market"], 0): return None, "market cooldown"
        if p["side"] != "long": return None, "spot account: long only"
        sp = p["stop_pct"]
        if not cfg.min_stop_pct <= sp <= cfg.max_stop_pct: return None, f"stop distance {sp:.2%} outside [{cfg.min_stop_pct:.1%},{cfg.max_stop_pct:.0%}]"
        used = sum(x["collateral"] for x in positions)
        exposure = min(cfg.max_pos_frac * equity, cfg.max_risk_pct * equity / sp, equity * 0.97 - used)   # risk-based size, capped
        exposure *= self.CONVICTION.get(str(p.get("conviction", "high")).lower(), 0.5)    # unknown value -> smallest size
        exposure = math.floor(exposure / MIN_STEP) * MIN_STEP
        if exposure < cfg.min_order: return None, f"position size {exposure:,.0f} below minimum order or not enough free cash"
        if used + exposure > cfg.max_gross * equity: return None, "gross exposure above cap"
        collateral, lev, risk = exposure, 1.0, exposure * sp
        e = p["entry"]; sgn = 1 if p["side"] == "long" else -1
        return Sized(collateral, lev, exposure, e * (1 - sgn * sp), e * (1 + sgn * p["take_pct"]), risk), "ok"

    def post_close(self, pos, now):
        self.st["cooldowns"][pos["market"]] = now + self.cfg.cooldown_s
        self.st["closed"] += 1
