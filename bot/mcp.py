"""Minimal Wallex MCP client (Streamable HTTP, JSON-RPC 2.0) at https://api.wallex.ir/mcp.
The AI trader uses READ-ONLY tools from here to research before deciding; orders never go through the LLM."""
import json
import os

import requests

URL = os.getenv("WALLEX_MCP_URL", "https://api.wallex.ir/mcp")
READ_ONLY = {"wallex_basic_candle_history", "wallex_basic_get_market_depth", "wallex_basic_get_latest_trades",
             "wallex_basic_get_account_fees", "wallex_credit_sub_account_balances_details"}


class WallexMCP:
    def __init__(self, api_key=None, sub=None, timeout=20):
        self.s, self.timeout, self.sid, self.n = requests.Session(), timeout, None, 0
        self.h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        key, sub = api_key or os.getenv("WALLEX_API_KEY"), sub or os.getenv("WALLEX_SUB_ACCOUNT_CLIENT_ID")
        if key: self.h["X-API-Key"] = key
        if sub: self.h["sub-account-client-id"] = sub
        self._rpc("initialize", {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "wallex-bot", "version": "1"}})
        self.s.post(URL, headers=self._hdr(), json={"jsonrpc": "2.0", "method": "notifications/initialized"}, timeout=timeout)
        self.tools = {t["name"]: t for t in self._rpc("tools/list", {})["tools"]}

    def _hdr(self): return {**self.h, **({"Mcp-Session-Id": self.sid} if self.sid else {})}

    def _rpc(self, method, params):
        self.n += 1
        r = self.s.post(URL, headers=self._hdr(), json={"jsonrpc": "2.0", "id": self.n, "method": method, "params": params}, timeout=self.timeout)
        self.sid = r.headers.get("Mcp-Session-Id", self.sid)
        t = r.text
        if "data:" in t[:200]: t = "\n".join(l[5:] for l in t.splitlines() if l.startswith("data:"))
        j = json.loads(t)
        if "error" in j: raise RuntimeError(j["error"])
        return j["result"]

    def call(self, name, args):
        if name not in READ_ONLY: raise PermissionError(f"{name} is not in the read-only allow-list")
        res = self._rpc("tools/call", {"name": name, "arguments": args})
        txt = "".join(c.get("text", "") for c in res.get("content", []))
        return txt[:6000]                                         # keep prompts small

    def openai_tools(self, names=("wallex_basic_candle_history", "wallex_basic_get_market_depth", "wallex_basic_get_latest_trades")):
        out = []
        for n in names:
            t = self.tools.get(n)
            if t: out.append({"type": "function", "function": {"name": n, "description": t.get("description", ""), "parameters": t.get("inputSchema") or {"type": "object", "properties": {}}}})
        return out
