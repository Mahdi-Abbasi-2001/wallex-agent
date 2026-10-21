"""Tiny status endpoint (GET / or /health) so the host can see the bot is alive: last tick time, equity, phase, open positions.
Started only when PORT is set (Liara). Read-only; exposes no keys and accepts no commands."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATUS = {"started": time.time()}


class _H(BaseHTTPRequestHandler):
    def do_GET(self):
        age = time.time() - STATUS.get("last_tick", 0)
        body = json.dumps({**STATUS, "seconds_since_tick": round(age, 1), "ok": age < 300}, default=str).encode()
        self.send_response(200 if age < 300 else 503); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)

    def log_message(self, *a): pass


def serve(port):
    srv = ThreadingHTTPServer(("0.0.0.0", int(port)), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
