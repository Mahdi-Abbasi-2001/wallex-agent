"""Persistent bot state (survives restarts) + kill switch."""
import json
import os
import time


class State:
    DEFAULTS = {"start_equity": None, "peak": None, "day": None, "day_start_equity": None, "halted": False, "halt_reason": "",
                "cooldowns": {}, "meta": {}, "regime": "", "last_decide": 0, "last_reflect": 0, "closed": 0, "params": {"min_score": 1.0, "max_lev": None}}

    def __init__(self, path):
        self.path = path
        self.d = json.loads(json.dumps(self.DEFAULTS))
        if os.path.exists(path):
            try: self.d.update(json.load(open(path)))
            except Exception: pass

    def __getitem__(self, k): return self.d[k]
    def get(self, k, default=None): return self.d.get(k, default)
    def __setitem__(self, k, v): self.d[k] = v

    autosave = True                                                  # replay turns this off (saved once at the end)

    def save(self):
        if not self.autosave: return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        json.dump(self.d, open(tmp, "w"), default=str)
        os.replace(tmp, self.path)


def killed(path): return os.path.exists(path)
