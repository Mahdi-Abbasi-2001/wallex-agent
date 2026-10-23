"""Mock-trade the challenge account over past days without placing orders: real Wallex TMN candles are replayed minute by minute
through the SAME bot code (signals, AI agents, risk guard, paper broker, logs). Nothing is sent to the exchange.
 python -m bot.replay --days 3 --rules        rules-only baseline (no LLM)
 python -m bot.replay --days 3                with the AI agents (OpenAI); the AI's tools only see candles up to the simulated time"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from agent.llm import LLM, load_env
from agent.wallex_client import Wallex

from .config import Config
from .external import bn_symbol, build, klines
from .futures import PaperFutures
from .runner import Bot


def fetch(w, sym, res, start, end, step):
    rows, t = [], start
    while t < end:
        c = w.candles(sym, res, t, min(t + step, end))
        if c.get("s") == "ok": rows += list(zip(c["t"], c["o"], c["h"], c["l"], c["c"], c["v"]))
        t += step
    df = pd.DataFrame(rows, columns=list("tohlcv")).drop_duplicates("t").sort_values("t").reset_index(drop=True)
    return df.astype({k: float for k in "ohlcv"}).astype({"t": int})


def load(cfg, start, end, cache="data/replay"):
    os.makedirs(cache, exist_ok=True); w = Wallex(); h, m = {}, {}
    sf = f"{cache}/spreads_{start}_{end}.json"
    saved = json.load(open(sf)) if os.path.exists(sf) else None
    mk = {} if saved else w.markets(); spreads = {}
    for s in cfg.universe:
        f1, fm = f"{cache}/{s}_60_{start}_{end}.csv", f"{cache}/{s}_1_{start}_{end}.csv"
        if os.path.exists(f1): h[s], m[s] = pd.read_csv(f1), pd.read_csv(fm)
        else:
            h[s] = fetch(w, s, "60", start - 6 * 86400, end, 5 * 86400); m[s] = fetch(w, s, "1", start, end, 86400)
            h[s].to_csv(f1, index=False); m[s].to_csv(fm, index=False)
        if saved: spreads[s] = saved.get(s, 0.003); continue
        st = mk.get(s, {}).get("stats", {})
        try: b, a = float(st["bidPrice"]), float(st["askPrice"]); spreads[s] = max((a - b) / ((a + b) / 2), 0.0004)
        except Exception: spreads[s] = 0.003
    if not saved: json.dump(spreads, open(sf, "w"), indent=1)            # frozen so every rerun of this window is identical
    return h, m, spreads


def load_external(cfg, start, end, m, cache="data/replay"):
    """Binance 1m closes (from 25h before start) + Wallex USDTTMN 1m, cached per window."""
    bn = {}
    for s in cfg.universe:
        sym = bn_symbol(s); f = f"{cache}/bn_{sym}_{start}_{end}.csv"
        if not sym: continue
        if os.path.exists(f): d = pd.read_csv(f)
        else:
            parts, t = [], start - 25 * 3600
            while t < end:
                k = klines(sym, "1m", t, min(t + 60000, end))
                if k is None or not len(k): break
                parts.append(k); t = int(k.t.iloc[-1]) + 60
            d = pd.concat(parts) if parts else pd.DataFrame(columns=["t", "c", "qv"]); d.to_csv(f, index=False)
        if len(d): bn[s] = (d.t.to_numpy(dtype=int), d.c.to_numpy(dtype=float))
    f = f"{cache}/USDTTMN_1_{start}_{end}.csv"
    if os.path.exists(f): u = pd.read_csv(f)
    else: u = fetch(Wallex(), "USDTTMN", "1", start - 25 * 3600, end, 86400); u.to_csv(f, index=False)
    wx = {s: (d.t.to_numpy(dtype=int), d.c.to_numpy(dtype=float)) for s, d in m.items() if len(d)}
    return bn, wx, (u.t.to_numpy(dtype=int), u.c.to_numpy(dtype=float))


class HistExternal:
    def __init__(self, bn, wx, usdt): self.bn, self.wx, self.usdt = bn, wx, usdt
    def get(self, now, markets): return build(now, markets, self.bn, self.wx, self.usdt)


class HistCandles:
    def __init__(self, h): self.h = h
    def update(self, now):                                          # only fully closed hourly bars
        return {s: d[d.t + 3600 <= now].reset_index(drop=True) for s, d in self.h.items() if len(d)}


class HistTools:
    """Stand-in for the Wallex MCP candle tool that cannot see the future."""
    def __init__(self, h, m, clock): self.h, self.m, self.clock = h, m, clock
    def openai_tools(self):
        return [{"type": "function", "function": {"name": "wallex_basic_candle_history", "description": "Retrieve OHLCV candles for one market symbol and resolution (1, 5, 15, 60, 240 minutes), most recent bars up to now.",
                 "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}, "resolution": {"type": "string"}, "bars": {"type": "integer"}}, "required": ["symbol", "resolution"]}}}]
    def call(self, name, args):
        s, res, n, now = args.get("symbol"), str(args.get("resolution", "60")), min(int(args.get("bars", 48)), 96), self.clock()
        if s not in self.h: return "unknown symbol"
        if res in ("60", "240"):
            d = self.h[s][self.h[s].t + 3600 <= now]
            if res == "240": d = d.groupby(d.t // 14400).agg(t=("t", "first"), o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), v=("v", "sum"))
        else:
            d = self.m[s][self.m[s].t + 60 <= now]; k = int(res) * 60
            if k > 60: d = d.groupby(d.t // k).agg(t=("t", "first"), o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), v=("v", "sum"))
        d = d.tail(n)
        return json.dumps({"t": d.t.astype(int).tolist(), "o": d.o.tolist(), "h": d.h.tolist(), "l": d.l.tolist(), "c": d.c.tolist(), "v": d.v.round(4).tolist()})


class RulesBrain:
    """Deterministic stand-in for the LLM roles (baseline for the ablation): take the top candidate, approve it, no reflection."""
    mcp = None
    def analyst(self, *a): return None
    def trader(self, ctx):
        held = {p["market"] for p in ctx["open_positions"]}; c = [x for x in ctx["candidates"] if x["market"] not in held]; return {"opens": [{"market": c[0]["market"], "side": "long", "conviction": "high", "reason": f"rules: top candidate {c[0]['market']}"}] if c else [], "closes": []}
    def reviewer(self, *a): return {"approve": True, "reason": "rules baseline"}
    def reflect(self, *a): return None


class ReplayBot(Bot):
    def __init__(self, cfg, h, m, spreads, llm=None, ext=None):
        self.sim = 0
        super().__init__(cfg, live=False, llm=None)
        self.log.clock = lambda: self.sim
        self.st.autosave = False                                      # speed: no per-minute state file writes in replay
        _eq = self.log.equity; self._n = 0
        def thin(**kw):                                               # speed: equity sampled every 5 simulated minutes
            self._n += 1
            if self._n % 5 == 1: _eq(**kw)
        self.log.equity = thin
        self.h, self.m, self.spreads = h, m, spreads
        self.mi = {s: (d.t.to_numpy(), d.c.to_numpy()) for s, d in m.items() if len(d)}
        self.cand = HistCandles(h)
        if llm:
            from .agents import Brain
            self.brain = Brain(llm, self.log, HistTools(h, m, lambda: self.sim), cfg.ai_max_tool_calls, cfg.ai_style)
        else: self.brain = RulesBrain()
        self.broker = PaperFutures(cfg)
        if ext: self.ext = ext

    def quotes(self, now):
        q = {}
        for s, (t, c) in self.mi.items():
            i = np.searchsorted(t, now - 60, side="right") - 1      # last closed minute
            if i >= 0: hs = self.spreads[s] / 2; q[s] = (c[i] * (1 - hs), c[i] * (1 + hs))
        return q

    def run_replay(self, start, end, step=60):
        self.st["last_decide"] = 0; t = start
        while t <= end:
            self.sim = t; ph, eq = self.tick(t)
            if ph == "trade" and not self.st["halted"] and t - self.st["last_decide"] >= self.cfg.decide_s:
                self.decide(t, eq); self.st["last_decide"] = t
            self.reflect(t); t += step
        for p in self.broker.positions(): self.close_position(p, "replay_end", end)
        return self.equity()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--days", type=float, default=3); ap.add_argument("--rules", action="store_true")
    ap.add_argument("--model", default=os.getenv("BOT_MODEL")); ap.add_argument("--out", default=None); ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--entry", default="pullback"); ap.add_argument("--tp", type=float, default=2.0); ap.add_argument("--trail", action="store_true"); ap.add_argument("--decide", type=int, default=1800)
    ap.add_argument("--gate", default="strict"); ap.add_argument("--maxpos", type=int, default=2); ap.add_argument("--dlcap", type=float, default=0.01); ap.add_argument("--fetch-only", action="store_true"); ap.add_argument("--data", default="lean"); ap.add_argument("--posfrac", type=float, default=0.33); ap.add_argument("--style", default="balanced"); ap.add_argument("--provider", default=None)
    a = ap.parse_args(); load_env()
    end = a.end or int(time.time()) // 3600 * 3600; start = int(end - a.days * 86400)
    out = a.out or f"logs/replay_{'rules' if a.rules else 'ai'}_{time.strftime('%m%d_%H%M', time.gmtime(end + 12600))}"
    import shutil; shutil.rmtree(out, ignore_errors=True); os.makedirs(out, exist_ok=True)
    cfg = Config(log_dir=out, state_file=f"{out}/state.json", kill_file=f"{out}/KILL", challenge_end=end + 60,
                 entry_mode=a.entry, tp_mult=a.tp, trail=a.trail, decide_s=a.decide, regime_gate=a.gate, max_positions=a.maxpos, daily_loss_cap=a.dlcap, ai_data=a.data, max_pos_frac=a.posfrac, ai_style=a.style)
    h, m, spreads = load(cfg, start, end)
    if a.fetch_only: load_external(cfg, start, end, m); print("fetched", start, end); return
    bot = ReplayBot(cfg, h, m, spreads, None if a.rules else LLM(a.provider, model=a.model), HistExternal(*load_external(cfg, start, end, m)))
    t0 = time.time(); eq = bot.run_replay(start, end)
    trades = bot.broker.history
    eqs = [json.loads(l)["equity"] for l in open(f"{out}/equity.jsonl")]
    peak = np.maximum.accumulate(eqs); dd = float(np.max(1 - np.array(eqs) / peak))
    summ = {"mode": "rules" if a.rules else f"ai:{a.model}", "start": start, "end": end, "final_equity": round(eq), "return_pct": round((eq / cfg.start_equity - 1) * 100, 2),
            "max_drawdown_pct": round(dd * 100, 2), "trades": len(trades), "wins": sum(t["profit"] > 0 for t in trades),
            "by_reason": pd.Series([t["close_reason"] for t in trades]).value_counts().to_dict() if trades else {},
            "pnl_by_market": pd.DataFrame(trades).groupby("market").profit.sum().round().to_dict() if trades else {}, "seconds": round(time.time() - t0)}
    json.dump(summ, open(f"{out}/summary.json", "w"), indent=1, default=float); print(json.dumps(summ, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
