"""OpenAI-compatible LLM wrapper. Provider is chosen by LLM_PROVIDER (openai | boxapi)
so moving from a personal OpenAI key to BoxAPI is a .env change only."""
import json
import os
from pathlib import Path

from openai import OpenAI


def load_env(path=".env"):
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


load_env()

PROVIDERS = {
    "openai": dict(key="OPENAI_API_KEY", base=None, model="OPENAI_MODEL", default_model="gpt-4o-mini"),
    "boxapi": dict(key="BOXAPI_KEY", base="https://ai.boxapi.ir/ai/v1", model="BOXAPI_MODEL", default_model=None),
}


class LLM:
    def __init__(self, provider=None, model=None, max_tokens_budget=None):
        self.provider = provider or os.getenv("LLM_PROVIDER", "openai")
        cfg = PROVIDERS[self.provider]
        api_key = os.getenv(cfg["key"])
        if not api_key:
            raise RuntimeError(f"{cfg['key']} is not set (put it in .env)")
        base = os.getenv("BOXAPI_BASE_URL", cfg["base"]) if self.provider == "boxapi" else cfg["base"]
        self.client = OpenAI(api_key=api_key, base_url=base)
        self.model = model or os.getenv(cfg["model"]) or cfg["default_model"]
        self.budget = max_tokens_budget
        self.used = 0
        self.prompt_tokens = self.completion_tokens = 0     # cumulative, for cost reporting

    def models(self):
        return [m.id for m in self.client.models.list().data]

    def decide(self, system, user):
        """Ask for a JSON object; returns parsed dict."""
        if self.budget is not None and self.used >= self.budget:
            raise RuntimeError("LLM token budget exhausted")
        kw = dict(model=self.model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                  response_format={"type": "json_object"})
        if not self.model.startswith(("gpt-5", "gpt-6", "o3", "o4")):
            kw["temperature"] = 0.2                     # reasoning models reject temperature
        r = self.client.chat.completions.create(**kw)
        if r.usage:
            self.used += r.usage.total_tokens; self.prompt_tokens += r.usage.prompt_tokens; self.completion_tokens += r.usage.completion_tokens
        return json.loads(r.choices[0].message.content)

    def agent(self, system, user, tools, call_tool, max_rounds=4, on_tool=None):
        """Tool-using loop: the model may call `tools` (executed by call_tool(name, args)) and must end with a JSON object.
        gpt-6 models only allow tools + reasoning on the Responses API, so they take that path."""
        if self.model.startswith("gpt-6"):
            return self._agent_responses(system, user, tools, call_tool, max_rounds, on_tool)
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for i in range(max_rounds + 1):
            kw = dict(model=self.model, messages=msgs)
            if tools and i < max_rounds: kw["tools"] = tools
            else: kw["response_format"] = {"type": "json_object"}
            r = self.client.chat.completions.create(**kw)
            if r.usage: self.used += r.usage.total_tokens; self.prompt_tokens += r.usage.prompt_tokens; self.completion_tokens += r.usage.completion_tokens
            m = r.choices[0].message
            if m.tool_calls and i < max_rounds:
                msgs.append({"role": "assistant", "content": m.content or "", "tool_calls": [tc.model_dump() for tc in m.tool_calls]})
                for tc in m.tool_calls:
                    try: args = json.loads(tc.function.arguments or "{}"); res = call_tool(tc.function.name, args)
                    except Exception as e: args, res = {}, f"error: {e!r}"
                    if on_tool: on_tool(tc.function.name, args, res)
                    msgs.append({"role": "tool", "tool_call_id": tc.id, "content": str(res)})
                continue
            try: return json.loads(m.content)
            except Exception:
                msgs.append({"role": "assistant", "content": m.content or ""}); msgs.append({"role": "user", "content": "Reply with the final JSON object only."})
        return None

    def _agent_responses(self, system, user, tools, call_tool, max_rounds, on_tool):
        rtools = [{"type": "function", "name": t["function"]["name"], "description": t["function"].get("description", ""),
                   "parameters": t["function"]["parameters"]} for t in tools]
        items = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for i in range(max_rounds + 1):
            kw = dict(model=self.model, input=items, text={"format": {"type": "json_object"}})
            if rtools and i < max_rounds: kw["tools"] = rtools
            r = self.client.responses.create(**kw)
            if r.usage:
                self.used += r.usage.total_tokens; self.prompt_tokens += r.usage.input_tokens; self.completion_tokens += r.usage.output_tokens
            calls = [o for o in r.output if o.type == "function_call"]
            if calls and i < max_rounds:
                items += [o.model_dump(exclude_none=True) for o in r.output]
                for c in calls:
                    try: args = json.loads(c.arguments or "{}"); res = call_tool(c.name, args)
                    except Exception as e: args, res = {}, f"error: {e!r}"
                    if on_tool: on_tool(c.name, args, res)
                    items.append({"type": "function_call_output", "call_id": c.call_id, "output": str(res)})
                continue
            try: return json.loads(r.output_text)
            except Exception:
                items += [o.model_dump(exclude_none=True) for o in r.output]; items.append({"role": "user", "content": "Reply with the final JSON object only."})
        return None
