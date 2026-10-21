"""The AI layer. Four LLM roles, all JSON-in/JSON-out, every call logged in full (logs/llm_calls.jsonl):
 analyst -> regime summary | trader -> picks/closes from the rule-engine candidates | reviewer -> approves or vetoes each proposal | reflection -> tunes bounded params.
If the LLM fails the bot never opens anything (it only manages and exits)."""
import json
import time

RULES = ("You trade Wallex SPOT toman markets, LONG ONLY, no leverage, 5,000,000 toman account; a round trip costs ~0.6% fees plus spread. "
         "STRATEGY: buy healthy PULLBACKS IN AN UPTREND. A rule engine has already selected the candidates because they dipped: each one is in an uptrend "
         "(above its 50-hour average, up over 72 hours) and has pulled back to around its 20-hour average in the last hours, while BTC itself is above its 50-hour average and up over 24 hours. "
         "So a recent dip, a negative 6h return or price near/below the 20-hour average is THE SETUP, not a warning sign. "
         "Every position gets a stop-loss and a take-profit at twice the stop distance, set by the rules. "
         "Data notes: returns and gaps are fractions (0.012 = +1.2%); sma20_gap/sma50_gap = price vs its 20/50-hour average; atr_1h = typical hourly range; "
         "stop_pct/take_pct = the stop and target distances; pnl_pct = open profit vs position size. "
         "Never invent markets: choose only from the candidate list. A hard risk guard sits after you; do not try to bypass it. "
         "Keep reasons short, factual and specific to the numbers you were given.")


TRADER_TASK = (" You make two SEPARATE decisions. "
    "(1) NEW ENTRY: by default BUY the best candidate (normally the first one). Your job is to tell a healthy pullback from a breakdown. "
    "Skip a candidate only for a concrete red flag, e.g.: the dip is much deeper than usual (well below the 20-hour average with the 50-hour trend rolling over), "
    "the 15-minute/1-hour candles show a sharp waterfall still accelerating with no sign of a low, BTC is falling hard right now, or the spread is wide. "
    "Express doubt through conviction (low/medium) rather than refusing; refuse only when a red flag is clear. If several candidates qualify, pick the cleanest one. "
    "(2) EXITS for positions you already hold: every position already has a stop-loss, a take-profit and a time limit managed by code, so the default is to HOLD. "
    "Propose a close ONLY if the uptrend itself is clearly broken (price below its 50-hour average and still falling, or BTC broke down hard). "
    "Being below the 20-hour average, small red or green fluctuations or a choppy 15-minute chart are NOT reasons to close. Positions younger than 4 hours cannot be closed by you.")
TRADER_SCHEMA = ('{"opens":[{"market":str,"side":"long","conviction":"low|medium|high","reason":str}],'
                 '"closes":[{"id":int,"reason":str}],"note":str}. At most 1 new position per call. conviction sets the size: high = the full size the risk guard allows, '
                 'medium = 75%, low = 50%. Use high only when the setup, the trend and the market regime all agree.')


class Brain:
    def __init__(self, llm, log, mcp=None, max_tools=2):
        self.llm, self.log, self.mcp, self.max_tools = llm, log, mcp, max_tools

    def _ask(self, role, system, payload, retries=2):
        user = json.dumps(payload, ensure_ascii=False, default=float)
        for i in range(retries):
            t0 = time.time()
            try:
                out = self.llm.decide(system, user)
                self.log.llm(role, system, user, out, seconds=round(time.time() - t0, 2), tokens=[getattr(self.llm, "prompt_tokens", None), getattr(self.llm, "completion_tokens", None)])
                return out
            except Exception as e:
                self.log.llm(role, system, user, None, error=repr(e), seconds=round(time.time() - t0, 2))
                time.sleep(2 * (i + 1))
        return None

    def analyst(self, btc, universe_feats, prev):
        return self._ask("analyst", RULES + ' Summarise the market regime for a trader in <=60 words. Reply JSON {"regime":"up|down|range","summary":str}.',
                         {"btc": btc, "markets": universe_feats, "previous_summary": prev})

    def trader(self, ctx):
        if self.mcp and hasattr(self.llm, "agent"):
            sys = (RULES + TRADER_TASK + ' You may call the market-data tools (candles in any resolution; live also order book depth and latest trades) to check a candidate '
                   'before deciding. Use at most ' + str(self.max_tools) + ' tool calls. Then reply JSON ' + TRADER_SCHEMA)
            user = json.dumps(ctx, ensure_ascii=False, default=float); t0 = time.time(); calls = []
            try:
                out = self.llm.agent(sys, user, self.mcp.openai_tools(), self.mcp.call, max_rounds=self.max_tools, on_tool=lambda n, a, r: calls.append({"tool": n, "args": a, "result": r[:500]}))
                self.log.llm("trader", sys, user, out, seconds=round(time.time() - t0, 2), tokens=[getattr(self.llm, "prompt_tokens", None), getattr(self.llm, "completion_tokens", None)]); self.log.decision("mcp_research", source=type(self.mcp).__name__, n_calls=len(calls), calls=calls)
                return out
            except Exception as e:
                self.log.llm("trader", sys, user, None, error=repr(e), seconds=round(time.time() - t0, 2))
        sys = RULES + TRADER_TASK + ' Reply JSON ' + TRADER_SCHEMA
        return self._ask("trader", sys, ctx)

    def reviewer(self, proposal, guard_note, ctx):
        sys = (RULES + ' You are the independent risk reviewer of a proposed entry. Approve by default. Veto ONLY for a concrete red flag: a breakdown rather than a pullback '
               '(e.g. deep below the 20-hour average with the 50-hour trend turning down), a crash in progress on the short-term candles, BTC falling hard right now, or a wide spread. '
               'A normal dip is the intended setup and is not a reason to veto. Reply JSON {"approve":bool,"reason":str}.')
        return self._ask("reviewer", sys, {"proposal": proposal, "guard": guard_note, "context": ctx})

    def reflect(self, trades, params):
        sys = (RULES + ' Review the last closed trades and tune bounded parameters. min_score in [0.7,2.0] (higher = pickier). '
               'Reply JSON {"min_score":number,"note":str}. Prefer small changes; after losses be more selective.')
        return self._ask("reflection", sys, {"trades": trades, "params": params})
