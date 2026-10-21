"""Append-only JSONL logs: decisions.jsonl (one record per decision/order) and llm_calls.jsonl (full prompts + replies)."""
import json
import os
import threading
import time

_lock = threading.Lock()


class Log:
    def __init__(self, directory="logs", clock=None):
        self.dir, self.clock = directory, clock          # clock: replay passes simulated time
        os.makedirs(directory, exist_ok=True)

    def _write(self, name, rec):
        t = self.clock() if self.clock else time.time()
        rec = {"ts": round(t, 3), "iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t + 3.5 * 3600)) + "+0330", **rec}
        with _lock, open(os.path.join(self.dir, name), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def decision(self, kind, **kw): self._write("decisions.jsonl", {"kind": kind, **kw})
    def llm(self, role, system, user, reply, error=None, seconds=None, tokens=None):
        self._write("llm_calls.jsonl", {"role": role, "system": system, "user": user, "reply": reply, "error": error, "seconds": seconds, "tokens_total": tokens})
    def equity(self, **kw): self._write("equity.jsonl", kw)
