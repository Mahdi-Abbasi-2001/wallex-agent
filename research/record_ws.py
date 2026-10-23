"""Record Wallex websocket data to disk for microstructure research.
 data/ws/trades_YYYY-MM-DD.jsonl : {"r": recv_ms, "s": sym, "p": price, "q": qty, "b": isBuy, "x": exchange_ts}
 data/ws/book_YYYY-MM-DD.jsonl   : {"r": recv_ms, "s": sym, "bid": [[p,q]*10], "ask": [[p,q]*10]}  (<=1 per second per symbol)
 Finished days are gzip-compressed. Reconnects automatically (server drops connections every 30 min).
 python record_ws.py            (runs forever; see data/ws/recorder.log)"""
import gzip
import json
import os
import shutil
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import websocket

SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "TRXUSDT"]
URL = "wss://api.wallex.ir/ws"
OUT = Path(__file__).parent / "data" / "ws"
LEVELS = 10
OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "recorder.log", "a", buffering=1)


def log(msg):
    LOG.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {msg}\n")


class Rec:
    def __init__(self):
        self.lock = threading.Lock()
        self.book = {s: {"bid": None, "ask": None, "dirty": False} for s in SYMS}
        self.files = {}      # (kind, day) -> file handle
        self.n_trades = self.n_books = self.n_msgs = 0
        self.last_msg = time.time()

    def _fh(self, kind, day):
        k = (kind, day)
        if k not in self.files:
            for (kk, dd), fh in list(self.files.items()):      # close + compress previous days
                if dd != day:
                    fh.close(); del self.files[(kk, dd)]
                    p = OUT / f"{kk}_{dd}.jsonl"
                    if p.exists():
                        with open(p, "rb") as a, gzip.open(str(p) + ".gz", "wb") as b:
                            shutil.copyfileobj(a, b)
                        p.unlink(); log(f"compressed {p.name}")
            self.files[k] = open(OUT / f"{kind}_{day}.jsonl", "a", buffering=1 << 16)
        return self.files[k]

    def on_open(self, ws):
        log("connected")
        for s in SYMS:
            for ch in ("trade", "buyDepth", "sellDepth"):
                ws.send(json.dumps(["subscribe", {"channel": f"{s}@{ch}"}]))

    def on_message(self, ws, msg):
        now = time.time(); self.last_msg = now
        try:
            ch, data = json.loads(msg); sym, kind = ch.split("@")
        except (ValueError, TypeError, AttributeError):
            return
        if sym not in self.book:
            return
        self.n_msgs += 1
        with self.lock:
            if kind == "trade" and isinstance(data, dict):
                day = datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%d")
                self._fh("trades", day).write(json.dumps({"r": int(now * 1000), "s": sym, "p": float(data["price"]), "q": float(data["quantity"]),
                                                          "b": bool(data["isBuyOrder"]), "x": data.get("timestamp")}, separators=(",", ":")) + "\n")
                self.n_trades += 1
            elif kind in ("buyDepth", "sellDepth") and isinstance(data, list):
                side = "bid" if kind == "buyDepth" else "ask"
                self.book[sym][side] = [[o["price"], o["quantity"]] for o in data[:LEVELS]]
                self.book[sym]["dirty"] = True

    def flusher(self):
        while True:
            time.sleep(1)
            now = time.time(); day = datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%d")
            with self.lock:
                for s, b in self.book.items():
                    if b["dirty"] and b["bid"] and b["ask"]:
                        self._fh("book", day).write(json.dumps({"r": int(now * 1000), "s": s, "bid": b["bid"], "ask": b["ask"]}, separators=(",", ":")) + "\n")
                        b["dirty"] = False; self.n_books += 1
                for fh in self.files.values():
                    fh.flush()

    def reporter(self):
        while True:
            time.sleep(300)
            log(f"msgs={self.n_msgs} trades={self.n_trades} books={self.n_books} idle={time.time()-self.last_msg:.0f}s")


def main():
    r = Rec()
    threading.Thread(target=r.flusher, daemon=True).start()
    threading.Thread(target=r.reporter, daemon=True).start()
    while True:
        ws = websocket.WebSocketApp(URL, on_open=r.on_open, on_message=r.on_message,
                                    on_error=lambda w, e: log(f"error {e}"), on_close=lambda w, c, m: log(f"closed {c} {m}"))
        try:
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:
            log(f"run_forever exception {e}")
        time.sleep(2)


if __name__ == "__main__":
    main()
