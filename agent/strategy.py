"""Strategy interface: decide(features) -> {"action": buy|sell|hold, "fraction": 0..1, "reason": str}."""
import json


class Strategy:
    def decide(self, feats: dict, state: dict) -> dict:
        raise NotImplementedError


class Hold(Strategy):
    def decide(self, feats, state):
        return {"action": "hold", "fraction": 0, "reason": "hold"}


class SmaTrend(Strategy):
    """Cheap rule-based baseline: trend + RSI filter. Benchmark for the LLM agent."""

    def decide(self, f, s):
        if "sma_50" not in f:
            return {"action": "hold", "fraction": 0, "reason": "warmup"}
        if f["sma_10"] > f["sma_50"] and f["rsi_14"] < 70:
            return {"action": "buy", "fraction": 1.0, "reason": "uptrend"}
        if f["sma_10"] < f["sma_50"] or f["rsi_14"] > 80:
            return {"action": "sell", "fraction": 1.0, "reason": "downtrend/overbought"}
        return {"action": "hold", "fraction": 0, "reason": "no signal"}


SYSTEM_PROMPT = (
    "You are a disciplined spot crypto trading agent. Long-only. Given market features and "
    "portfolio state, answer ONLY with JSON: "
    '{"action":"buy|sell|hold","fraction":0..1,"reason":"short"}. Prefer hold unless edge is clear.'
)


class LLMStrategy(Strategy):
    """Wraps agent.llm.LLM. Any failure/invalid output degrades to hold. Call every `every` bars."""

    def __init__(self, llm, every=1, system=SYSTEM_PROMPT):
        self.llm, self.every, self.system, self.n = llm, every, system, 0

    def decide(self, f, s):
        self.n += 1
        if (self.n - 1) % self.every:
            return {"action": "hold", "fraction": 0, "reason": "skipped"}
        try:
            return self.llm.decide(self.system, json.dumps({"features": f, "state": s}))
        except Exception as e:  # network, budget, bad JSON
            return {"action": "hold", "fraction": 0, "reason": f"llm error: {type(e).__name__}"}
