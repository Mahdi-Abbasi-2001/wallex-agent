"""Minimal Wallex REST client (public market data + spot orders)."""
import os
import time
import threading
from collections import deque

import requests

BASE = "https://api.wallex.ir"


class RateLimiter:
    """Sliding window: max `n` calls per `per` seconds (orders: 20 / 10s)."""

    def __init__(self, n=20, per=10.0):
        self.n, self.per = n, per
        self.calls = deque()
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            while self.calls and now - self.calls[0] > self.per:
                self.calls.popleft()
            if len(self.calls) >= self.n:
                time.sleep(self.per - (now - self.calls[0]) + 0.05)
            self.calls.append(time.monotonic())


class Wallex:
    def __init__(self, api_key=None, sub_account_client_id=None, timeout=10):
        self.api_key = api_key or os.getenv("WALLEX_API_KEY")
        self.sub = sub_account_client_id or os.getenv("WALLEX_SUB_ACCOUNT_CLIENT_ID")
        self.timeout = timeout
        self.s = requests.Session()
        self.s.headers["Content-Type"] = "application/json"
        self.order_limiter = RateLimiter(20, 10.0)

    def _headers(self):
        h = {}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        if self.sub:
            h["sub-account-client-id"] = self.sub
        return h

    def _req(self, method, path, retries=3, **kw):
        for i in range(retries):
            try:
                r = self.s.request(method, BASE + path, headers=self._headers(),
                                   timeout=self.timeout, **kw)
                if r.status_code == 429 or r.status_code >= 500:
                    time.sleep(1 + i)
                    continue
                return r.status_code, r.json()
            except (requests.RequestException, ValueError):
                if i == retries - 1:
                    raise
                time.sleep(1 + i)
        raise RuntimeError(f"{method} {path} failed after retries")

    # ---- public ----
    def markets(self):
        return self._req("GET", "/v1/markets")[1]["result"]["symbols"]

    def depth(self, symbol):
        return self._req("GET", "/v1/depth", params={"symbol": symbol})[1]["result"]

    def candles(self, symbol, resolution, start, end):
        """resolution in minutes ('1','5','60') or 'D'. Returns UDF dict (t,o,h,l,c,v)."""
        return self._req("GET", "/v1/udf/history", params={
            "symbol": symbol, "resolution": resolution, "from": start, "to": end})[1]

    # ---- private ----
    def place_order(self, symbol, side, type_, quantity, price=None,
                    stop_price=None, client_id=None):
        body = {"symbol": symbol, "side": side, "type": type_,
                "quantity": str(quantity)}
        if price is not None:
            body["price"] = str(price)
        if stop_price is not None:
            body["stop_Price"] = str(stop_price)
        if client_id:
            body["client_id"] = client_id
        self.order_limiter.wait()
        return self._req("POST", "/v1/account/orders", json=body, retries=1)

    def cancel_order(self, client_id):
        return self._req("DELETE", f"/v1/account/orders/{client_id}", retries=1)

    def orders(self, **params):
        return self._req("GET", "/v1/account/orders", params=params)[1]

    # ---- account info (documented in basic-services) ----
    def balances(self):
        """Main-account wallet: {asset: {value, locked, ...}}. Sub-account wallet endpoint: see trade-credit docs."""
        return self._req("GET", "/v1/account/balances")[1]["result"]["balances"]

    def fees(self):
        """Per-market maker/taker fee rates for this account."""
        return self._req("GET", "/v1/account/fee")[1]["result"]

    def free(self, asset):
        b = self.balances().get(asset)
        return float(b["value"]) - float(b["locked"]) if b else 0.0

    # ---- order queries / misc public (documented) ----
    def order(self, client_id):
        return self._req("GET", f"/v1/account/orders/{client_id}")[1]

    def open_orders(self, symbol=None, page=1, per_page=50):
        p = {"page": page, "per_page": per_page}
        if symbol:
            p["symbol"] = symbol
        return self._req("GET", "/v1/account/openOrders", params=p)[1]

    def trades(self, symbol):
        """Latest executed trades (docs warn: don't poll often, use websocket)."""
        return self._req("GET", "/v1/trades", params={"symbol": symbol})[1]["result"]["latestTrades"]

    def sub_accounts(self):
        return self._req("GET", "/sub-accounts")[1]
