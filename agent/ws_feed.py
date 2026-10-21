"""Real-time feed: wss://api.wallex.ir/ws  (pip install websocket-client)
Channels per symbol: @trade, @buyDepth, @sellDepth. Server drops connections after 30 min,
so we reconnect (and resubscribe) in a loop."""
import json
import threading
import time

import websocket

URL = "wss://api.wallex.ir/ws"


class PriceFeed:
    def __init__(self, symbols, depth=True):
        self.symbols, self.depth = symbols, depth
        self.last = {}      # symbol -> last trade price
        self.trades = {s: [] for s in symbols}   # (price, qty, is_buy, ts)
        self.bids = {}      # symbol -> [(price, qty)] best first
        self.asks = {}
        self.updated = 0.0  # time of last message
        self.lock = threading.Lock()
        self._stop = False

    def _on_open(self, ws):
        for s in self.symbols:
            chans = [f"{s}@trade"] + ([f"{s}@buyDepth", f"{s}@sellDepth"] if self.depth else [])
            for ch in chans:
                ws.send(json.dumps(["subscribe", {"channel": ch}]))

    def _on_message(self, ws, msg):
        try:
            ch, data = json.loads(msg)
            sym, kind = ch.split("@")
        except (ValueError, TypeError, AttributeError):
            return
        with self.lock:
            self.updated = time.time()
            if kind == "trade" and isinstance(data, dict):
                p = float(data["price"])
                self.last[sym] = p
                t = self.trades.setdefault(sym, [])
                t.append((p, float(data["quantity"]), bool(data["isBuyOrder"]), time.time()))
                del t[:-1000]
            elif kind in ("buyDepth", "sellDepth") and isinstance(data, list):
                book = [(float(o["price"]), float(o["quantity"])) for o in data]
                if kind == "buyDepth":
                    self.bids[sym] = sorted(book, reverse=True)
                else:
                    self.asks[sym] = sorted(book)

    def _run(self):
        while not self._stop:
            ws = websocket.WebSocketApp(URL, on_open=self._on_open, on_message=self._on_message)
            try:
                ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception:
                pass
            time.sleep(2)

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self._stop = True

    def snapshot(self, sym, window=300, levels=10):
        """Microstructure features from the last `window` seconds of trades + current book."""
        with self.lock:
            now = time.time()
            tr = [t for t in self.trades.get(sym, []) if now - t[3] <= window]
            b, a = self.bids.get(sym, [])[:levels], self.asks.get(sym, [])[:levels]
            out = {"ws_age_s": round(now - self.updated, 1) if self.updated else None}
            if sym in self.last:
                out["last"] = self.last[sym]
            if tr:
                buy = sum(t[1] for t in tr if t[2]); sell = sum(t[1] for t in tr if t[1] and not t[2])
                out["trades_n"] = len(tr)
                out["flow_imbalance"] = round((buy - sell) / (buy + sell), 3) if buy + sell else 0.0
                out["ret_window"] = round(tr[-1][0] / tr[0][0] - 1, 5)
            if b and a:
                bv, av = sum(q for _, q in b), sum(q for _, q in a)
                out["spread_pct"] = round((a[0][0] - b[0][0]) / b[0][0], 5)
                out["book_imbalance"] = round((bv - av) / (bv + av), 3)
            return out
