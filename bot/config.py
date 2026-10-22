"""Run configuration. Everything is overridable by env var so the Liara deployment needs no code change."""
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

TEHRAN = timezone(timedelta(hours=3, minutes=30))


def _f(name, default): return float(os.getenv(name, default))
def _i(name, default): return int(os.getenv(name, default))


def _ts(name, default_iso):
    return datetime.fromisoformat(os.getenv(name, default_iso)).timestamp()


def blocked_tmn():
    p = os.path.join(os.path.dirname(__file__), "blocked_tmn.txt")
    return {x + "TMN" for l in open(p) if not l.startswith("#") for x in l.split()} if os.path.exists(p) else set()


@dataclass
class Config:
    mode: str = "shadow"                       # shadow (paper spot on live prices) | live (real spot orders). Challenge account = SPOT ONLY, no leverage/margin.
    universe: tuple = tuple(os.getenv("UNIVERSE", "BTCTMN,ETHTMN,SOLTMN,XRPTMN,BNBTMN,DOGETMN,TRXTMN,ADATMN,LTCTMN,LINKTMN,AVAXTMN,PEPETMN,PUMPTMN,SHIBTMN,ARBTMN,STMN,HYPETMN,UNITMN,AAVETMN,BCHTMN,ATOMTMN,WLDTMN,APETMN,FLOKITMN,PENGUTMN,XAUTTMN,USOONTMN").split(","))
    start_equity: float = _f("START_EQUITY", 5_000_000)     # toman
    challenge_end: float = _ts("CHALLENGE_END", "2026-10-23T23:50:00+03:30")
    # --- risk guard (hard limits the LLM cannot override) ---
    max_lev: float = 1.0                                     # spot only
    hard_lev_cap: float = 1.0
    max_positions: int = _i("MAX_POSITIONS", 2)
    max_gross: float = 1.0                                   # total invested / equity (no leverage)
    max_pos_frac: float = _f("MAX_POS_FRAC", 0.33)           # one position <= 33% of equity
    min_order: float = 150_000                               # toman, above Wallex minNotional
    max_risk_pct: float = _f("MAX_RISK_PCT", 0.015)          # equity lost if one stop is hit (before slippage)
    min_stop_pct: float = 0.012
    max_stop_pct: float = 0.05
    daily_loss_cap: float = _f("DAILY_LOSS_CAP", 0.01)
    max_drawdown: float = _f("MAX_DRAWDOWN", 0.05)
    lock_reduce_at: float = _f("LOCK_REDUCE_AT", 0.03)       # up this much: only 1 position at a time
    lock_flat_at: float = _f("LOCK_FLAT_AT", 0.06)
    no_new_entries: bool = os.getenv("NO_NEW_ENTRIES", "0") == "1"   # close-out mode: manage open positions only, never buy           # up this much: close everything, stop trading
    cooldown_s: int = _i("COOLDOWN_S", 4 * 3600)
    max_hold_s: int = _i("MAX_HOLD_S", 36 * 3600)
    entry_mode: str = os.getenv("ENTRY_MODE", "pullback")   # momentum | pullback | lead
    regime_gate: str = os.getenv("REGIME_GATE", "strict")    # normal | strict (BTC above its 50h average AND up over 24h, else no entries)
    tp_mult: float = _f("TP_MULT", 2.0)                      # take-profit distance = tp_mult x stop distance
    trail: bool = os.getenv("TRAIL", "0") == "1"             # trailing stop once a position is up by one stop distance
    # --- what the AI sees (chosen by ablation; "lean" = only what the rules do not already decide) ---
    ai_data: str = os.getenv("AI_DATA", "lean")             # lean | lean_bn (lean + Binance/USDT context) | full (old, every field + analyst + reflection)
    ai_max_tool_calls: int = _i("AI_MAX_TOOL_CALLS", 2)
    min_ai_hold_s: int = _i("MIN_AI_HOLD_S", 4 * 3600)       # the AI may not close a position younger than this (stops/targets still work)
    # --- end game (hours before challenge_end) ---
    no_new_h: float = _f("NO_NEW_H", 2.0)                     # no new positions in the last 2 hours
    unwind_h: float = _f("UNWIND_H", 0.5)                    # staggered limit sells from 30 min before the end
    force_min: float = _f("FORCE_MIN", 5.0)                  # everything sold, toman only, 5 min before the end
    # --- timing ---
    tick_s: int = _i("TICK_S", 30)                           # price / protection loop
    decide_s: int = _i("DECIDE_S", 1800)                     # LLM trading decision cycle
    reflect_s: int = _i("REFLECT_S", 8 * 3600)
    # --- exchange numbers (re-read from the API at startup when available) ---
    trade_fee: float = _f("TRADE_FEE", 0.003)                # per side; conservative until /fees is read with the real key
    open_fee: float = 0.0
    extend_fee: float = 0.0
    liq_margin: float = 0.15                                 # unused in spot
    min_exposure: float = 0.0
    state_file: str = os.getenv("STATE_FILE", "logs/state.json")
    kill_file: str = os.getenv("KILL_FILE", "logs/KILL")
    log_dir: str = os.getenv("LOG_DIR", "logs")

    def __post_init__(self):
        bad = [m for m in self.universe if m in blocked_tmn()]
        if bad: self.universe = tuple(m for m in self.universe if m not in bad)    # never trade a market the organizers blocked
        self.dropped_blocked = bad
