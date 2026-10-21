"""Main loop. python -m bot --mode shadow|live   (live additionally needs --confirm-live)."""
import argparse
import os
import sys
import time

from agent.llm import LLM, load_env
from agent.wallex_client import Wallex

from .agents import Brain
from .config import Config
from .futures import PaperFutures, SpotLive
from .health import STATUS, serve
from .log import Log
from .mcp import WallexMCP
from .risk import RiskGuard, phase
from .external import LiveExternal
from .signals import Candles, candidates, features, regime
from .state import State, killed


def code_version():
    """Which code is running: git commit (+ dirty flag) when available, else GIT_COMMIT from the deploy, plus a hash of the bot's source
    files so the logs prove the submitted code is the code that traded."""
    import hashlib, subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h = hashlib.sha256()
    for d in ("bot", "agent"):
        for f in sorted(os.listdir(os.path.join(root, d))):
            if f.endswith((".py", ".txt")): h.update(f.encode()); h.update(open(os.path.join(root, d, f), "rb").read())
    out = {"source_sha256": h.hexdigest()[:16], "git_commit": os.getenv("GIT_COMMIT")}
    try:
        out["git_commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5).stdout.strip() or out["git_commit"]
        out["git_dirty"] = bool(subprocess.run(["git", "status", "--porcelain", "bot", "agent"], cwd=root, capture_output=True, text=True, timeout=5).stdout.strip())
    except Exception: pass
    return out


def quotes_for(w, universe):
    """Live best bid/ask for every market in one call (/v2/depth/all). The /v1/markets price stats lag by minutes, so they are never used for trading."""
    code, body = w._req("GET", "/v2/depth/all")
    allq, q = (body or {}).get("result") or {}, {}
    for s in universe:
        d = allq.get(s) or {}
        try: q[s] = (float(d["bid"][0]["price"]), float(d["ask"][0]["price"]))
        except Exception: pass
    return q


def quote_one(w, market):
    d = w.depth(market)
    return float(d["bid"][0]["price"]), float(d["ask"][0]["price"])


class Bot:
    def __init__(self, cfg: Config, live=False, llm=None):
        self.cfg, self.live = cfg, live
        self.w = Wallex(); self.log = Log(cfg.log_dir); self.st = State(cfg.state_file); self.guard = RiskGuard(cfg, self.st)
        self.broker = SpotLive(self.w, cfg) if live else PaperFutures(cfg)
        self.cand = Candles(self.w, cfg.universe)
        self.ext = LiveExternal(self.w)
        mcp = None
        if llm:
            try: mcp = WallexMCP()
            except Exception as e: self.log.decision("mcp_unavailable", error=repr(e))
        self.brain = Brain(llm, self.log, mcp, cfg.ai_max_tool_calls) if llm else None
        self.last_quotes = {}

    # ---- helpers ----
    def quotes(self, now):
        """Every tick: fresh order books only for the coins held (stops/targets). The full universe is refreshed by decide()."""
        q = dict(self.last_quotes)
        if not q: q.update(quotes_for(self.w, self.cfg.universe))
        for p in self.broker.positions():
            try: q[p["market"]] = quote_one(self.w, p["market"])
            except Exception: pass
        return q

    def refresh_all_quotes(self):
        try: self.last_quotes.update(quotes_for(self.w, self.cfg.universe))
        except Exception as e: self.log.decision("quotes_failed", error=repr(e))

    def exit_price(self, pos, q): return q[0] if pos["side"] == "long" else q[1]

    def close_position(self, pos, why, now):
        q = self.last_quotes.get(pos["market"])
        if not q: return
        try:
            res = self.broker.close(pos["id"], self.exit_price(pos, q), now, reason=why)
        except Exception as e:
            self.log.decision("close_failed", id=pos["id"], market=pos["market"], error=repr(e)); return
        self.guard.post_close(pos, now)
        self.log.decision("close", id=pos["id"], market=pos["market"], side=pos["side"], why=why, price=res.get("close_price", self.exit_price(pos, q)), pnl=res.get("profit"),
                          latency=res.get("latency"), meta=self.st["meta"].get(str(pos["id"])))

    def equity(self): return self.broker.equity(self.last_quotes)

    # ---- one fast tick: prices, protection, end game ----
    def tick(self, now):
        cfg = self.cfg
        self.last_quotes = self.quotes(now)
        for c in self.broker.sync(self.last_quotes, now):
            self.guard.post_close(c, now)
            self.log.decision("close", id=c["id"], market=c["market"], side=c["side"], why=c["close_reason"], price=c.get("close_price"), pnl=c.get("profit"), meta=self.st["meta"].get(str(c["id"])))
        eq = self.equity(); info = self.guard.update_equity(eq, now); ph = phase(cfg, now)
        positions = self.broker.positions()
        STATUS.update(last_tick=now, equity=round(eq), ret=round(info["ret"], 5), dd=round(info["dd"], 5), phase=ph, open=len(positions), halted=self.st["halted"], mode="live" if self.live else "shadow")
        self.log.equity(equity=round(eq), ret=round(info["ret"], 5), dd=round(info["dd"], 5), phase=ph, open=len(positions), halted=self.st["halted"])
        if info["halt"]: self.log.decision("halt", reason=info["halt"], equity=eq)
        if cfg.trail: self.trail_stops(positions)
        stop_all = killed(cfg.kill_file) or self.st["halted"] or ph == "force"
        for p in positions:
            held = now - (self.st["meta"].get(str(p["id"])) or {}).get("opened_ts", now)
            if stop_all: self.close_position(p, "kill_switch" if killed(cfg.kill_file) else (self.st["halt_reason"] or "end_game_force"), now)
            elif ph == "unwind": self.close_position(p, "end_game_unwind", now); break         # one per tick -> staggered exits
            elif held > cfg.max_hold_s: self.close_position(p, "max_hold_time", now)
        if ph in ("force", "ended") and self.live and hasattr(self.broker, "leftovers"): self.end_check(now)
        self.st.save()
        return ph, eq

    def end_check(self, now):
        """Final rule of the challenge: only toman at the end. Sell any coin balance worth more than the minimum order, log the rest."""
        left = self.broker.leftovers(self.last_quotes)
        for m, (qty, val) in left.items():
            if val >= 50_000:
                q, px = self.broker.sell_all(m, qty, val / qty)
                self.log.decision("end_sell_leftover", market=m, qty=qty, sold=q, price=px)
        if not self.st.get("end_logged"):
            self.log.decision("end_check", leftovers={m: round(v) for m, (q, v) in self.broker.leftovers(self.last_quotes).items()}); self.st["end_logged"] = True

    def trail_stops(self, positions):
        """Once a position is up by one stop distance, trail the stop at (high-water mark - stop distance), never below breakeven + fees."""
        for p in positions:
            m = self.st["meta"].get(str(p["id"])); q = self.last_quotes.get(p["market"])
            if not m or not q: continue
            sp = m["signal"]["stop_pct"]; hw = max(m.get("hw", p["entry"]), q[0]); m["hw"] = hw
            if hw >= p["entry"] * (1 + sp):
                new = max(hw * (1 - sp), p["entry"] * (1 + 2 * self.cfg.trade_fee + 0.001))
                if new > p["stop"]:
                    self.broker.set_stop(p["id"], new); self.log.decision("trail_stop", id=p["id"], market=p["market"], stop=new, high_water=hw)

    # ---- slow cycle: AI decisions ----
    def decide(self, now, eq):
        cfg, st = self.cfg, self.st
        if not self.brain: return
        self.refresh_all_quotes()
        data = self.cand.update(now)
        feats = {s: features(df) for s, df in data.items()}; feats = {k: v for k, v in feats.items() if v}
        ext_all = None
        if cfg.entry_mode == "lead":
            try: ext_all, _ = self.ext.get(now, list(feats))
            except Exception: ext_all = {}
        reg, cands = candidates(feats, self.last_quotes, cfg, st["params"].get("min_score", 1.0), ext_all)
        cands = [c for c in cands if c["market"] not in st.get("refused", [])]
        feasible = [c for c in cands if c["stop_pct"] * cfg.min_exposure <= cfg.max_risk_pct * eq]
        if len(feasible) < len(cands): self.log.decision("filtered", dropped=[c["market"] for c in cands if c not in feasible], why="min position size makes the stop too risky for this account")
        cands = feasible
        positions = self.broker.positions()
        if not cands and not positions:                              # nothing to choose and nothing to manage: no AI call (saves credits)
            self.log.decision("cycle", regime=reg, candidates=[], open=0, ai="skipped: no candidates, no positions"); return
        full, use_bn = cfg.ai_data == "full", cfg.ai_data in ("full", "lean_bn")
        ext, ext_ctx = {}, {}
        if use_bn:
            try: ext, ext_ctx = self.ext.get(now, list({"BTCTMN", *[c["market"] for c in cands], *[p["market"] for p in positions]}))
            except Exception as e: self.log.decision("external_data_failed", error=repr(e))
        for c in cands: c.update(ext.get(c["market"], {}))
        btc = feats.get("BTCTMN") or {}
        if full:
            a = self.brain.analyst({**btc, **ext.get("BTCTMN", {}), **ext_ctx}, {k: {x: v[x] for x in ("ret_24h", "sma20_gap", "atr_1h")} for k, v in feats.items()}, st["regime"])
            if a: st["regime"] = a.get("summary", "")
            ctx = {"rule_engine_regime": reg, "usdt_tmn": ext_ctx, "btc_binance": ext.get("BTCTMN"), "analyst": a, "candidates": cands, "equity": round(eq), "return": round(eq / cfg.start_equity - 1, 4)}
        else:                                                         # lean: only what the rules do not already decide
            keep = ("market", "side", "entry", "spread", "ret_6h", "ret_24h", "ret_72h", "sma20_gap", "sma50_gap", "atr_1h", "stop_pct", "take_pct",
                    "bn_ret_15m", "bn_ret_1h", "bn_ret_24h", "wx_ret_15m", "premium_vs_binance")
            a = None
            ctx = {"market_regime": reg, "btc": {k: btc.get(k) for k in ("ret_6h", "ret_24h", "sma20_gap", "sma50_gap")},
                   "candidates": [{k: c[k] for k in keep if k in c} for c in cands]}
            if use_bn: ctx["btc"].update({k: v for k, v in ext.get("BTCTMN", {}).items() if k.startswith("bn_")}); ctx["usdt_tmn_ret_24h"] = ext_ctx.get("usdt_tmn_ret_24h")
        ctx["open_positions"] = [{**{k: p[k] for k in ("id", "market", "entry", "stop", "take")}, "pnl_pct": round(p["profit"] / max(p["collateral"], 1), 4),
                                  "age_hours": round((now - (st["meta"].get(str(p["id"])) or {}).get("opened_ts", now)) / 3600, 1)} for p in positions]
        self.log.decision("cycle", regime=reg, candidates=cands, analyst=a, open=len(positions), ai_data=cfg.ai_data)
        out = self.brain.trader(ctx)
        if not out: self.log.decision("llm_unavailable", note="no new positions this cycle"); return
        for c in out.get("closes") or []:
            p = next((x for x in positions if x["id"] == c.get("id")), None)
            if not p: continue
            age = now - (self.st["meta"].get(str(p["id"])) or {}).get("opened_ts", now)
            if age < self.cfg.min_ai_hold_s: self.log.decision("ai_close_blocked", id=p["id"], market=p["market"], age_min=round(age / 60), reason=c.get("reason")); continue
            self.close_position(p, "ai: " + str(c.get("reason", "")), now)
        for o in (out.get("opens") or [])[:1]:
            self.try_open(o, cands, positions, eq, ctx, now)
        self.st["last_decide"] = now

    def try_open(self, o, cands, positions, eq, ctx, now):
        cand = next((c for c in cands if c["market"] == o.get("market") and c["side"] == o.get("side")), None)
        if not cand: self.log.decision("reject", stage="candidate", proposal=o, reason="not in candidate list"); return
        prop = {"market": cand["market"], "side": cand["side"], "entry": cand["entry"], "stop_pct": cand["stop_pct"], "take_pct": cand["take_pct"], "lev": 1, "conviction": o.get("conviction", "low")}
        sized, why = self.guard.check_open(prop, eq, self.broker.positions(), now)
        if not sized: self.log.decision("reject", stage="risk_guard", proposal=o, reason=why); return
        rev = self.brain.reviewer({**prop, "ai_reason": o.get("reason")}, {"size_toman": sized.exposure, "share_of_equity": round(sized.exposure / eq, 3), "conviction": prop["conviction"], "risk_toman_if_stopped": sized.risk_toman}, ctx)
        if not rev or not rev.get("approve"): self.log.decision("reject", stage="reviewer", proposal=o, review=rev); return
        try: pos = self.broker.open(prop["market"], prop["side"], sized.collateral, sized.risk_coef, prop["entry"], sized.stop_price, sized.take_price, now)
        except Exception as e:
            self.log.decision("open_failed", proposal=o, error=repr(e))
            if "امکان ثبت سفارش" in repr(e) or "symbol" in repr(e):     # Wallex refuses this market for the account: never propose it again
                self.st.d.setdefault("refused", []).append(prop["market"]); self.log.decision("market_refused", market=prop["market"])
            return
        self.st["meta"][str(pos["id"])] = {"reason": o.get("reason"), "review": rev, "opened_ts": now, "signal": cand}
        self.log.decision("open", id=pos["id"], proposal=o, signal=cand, sized=sized.__dict__, review=rev, ai_reason=o.get("reason"),
                          fill={"price": pos.get("entry"), "qty": pos.get("qty"), "toman": pos.get("collateral")}, latency=pos.get("latency"))

    def reflect(self, now):
        if self.cfg.ai_data != "full": return                       # lean setups: no self-tuning (reproducible behaviour)
        if not self.brain or now - self.st["last_reflect"] < self.cfg.reflect_s: return
        closed = [{k: t.get(k) for k in ("market", "side", "profit", "close_reason")} for t in (self.broker.history[-10:] if hasattr(self.broker, "history") else [])]
        self.st["last_reflect"] = now
        if not closed: return
        r = self.brain.reflect(closed, self.st["params"])
        if r:
            p = self.st["params"]
            p["min_score"] = float(min(2.0, max(0.7, r.get("min_score", p["min_score"]))))
            self.log.decision("reflection", result=r, params=dict(p))

    def mcp_account_check(self):
        """Read the challenge account through the Wallex MCP (balances + fees) and log it, as an independent check of the REST view."""
        mcp = getattr(self.brain, "mcp", None) if self.brain else None
        if not mcp or not hasattr(mcp, "tools"): return
        for tool in ("wallex_credit_sub_account_balances_details", "wallex_basic_get_account_fees"):
            try: self.log.decision("mcp_account", tool=tool, result=mcp.call(tool, {})[:1500])
            except Exception as e: self.log.decision("mcp_account", tool=tool, error=repr(e))

    def run(self):
        cfg = self.cfg
        if os.getenv("PORT"):
            try: serve(os.getenv("PORT"))
            except Exception as e: self.log.decision("health_endpoint_failed", error=repr(e))   # never let the status page stop trading
        self.log.decision("start", mode="live" if self.live else "shadow", universe=cfg.universe, end=cfg.challenge_end, ai=bool(self.brain),
                          model=getattr(getattr(self.brain, "llm", None), "model", None), code=code_version(), config={k: v for k, v in cfg.__dict__.items() if k not in ("universe",)})
        self.mcp_account_check()
        while True:
            now = time.time()
            try:
                ph, eq = self.tick(now)
                if ph == "ended" or (killed(cfg.kill_file) and not self.broker.positions()): self.log.decision("stop", reason="ended or killed", equity=eq); return
                if ph == "trade" and not self.st["halted"] and now - self.st["last_decide"] >= cfg.decide_s: self.decide(now, eq); self.st["last_decide"] = now
                self.reflect(now); self.st.save()
            except Exception as e:                                   # never die: log and keep protecting positions
                self.log.decision("error", error=repr(e))
            time.sleep(cfg.tick_s)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="shadow", choices=["shadow", "live"]); ap.add_argument("--confirm-live", action="store_true")
    ap.add_argument("--provider", default=None); ap.add_argument("--model", default=os.getenv("BOT_MODEL", "gpt-5.4-mini")); ap.add_argument("--no-llm", action="store_true")
    a = ap.parse_args(argv); load_env()
    if a.mode == "live" and not (a.confirm_live and os.getenv("CONFIRM_LIVE") == "yes"):
        sys.exit("live mode needs --confirm-live and CONFIRM_LIVE=yes in the environment")
    llm = None if a.no_llm else LLM(a.provider, model=a.model)
    Bot(Config(mode=a.mode), live=a.mode == "live", llm=llm).run()
