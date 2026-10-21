"""Wallex margin ("futures") access: LiveFutures sends real positions, PaperFutures simulates them with the same fee/liquidation rules.
Both expose: open(), close(), positions(), equity(prices), sync(prices) -> list of auto-closed positions."""
import math
import time

from agent.wallex_client import Wallex

MIN_STEP = 10_000   # collateral rounding (toman)


class LiveFutures:
    """Thin wrapper over /margin-trade/v1. Field semantics marked (verify) must be confirmed with the first minimum-size test."""
    def __init__(self, w: Wallex = None):
        self.w = w or Wallex()

    def _call(self, method, path, **kw):
        self.w.order_limiter.wait()
        code, body = self.w._req(method, path, retries=2, **kw)
        if code >= 400 or not body.get("success", True):
            raise RuntimeError(f"{method} {path} -> {code} {body}")
        return body.get("result", body)

    def markets(self): return self.w._req("GET", "/margin-trade/v1/public/markets")[1].get("result")
    def loan(self, market, side, risk_coef, open_price=None):
        b = {"market": market, "side": side, "risk_coef": str(risk_coef)}
        if open_price: b["open_price"] = str(open_price)
        return self._call("POST", "/margin-trade/v1/public/loan", json=b)
    def dry_run(self, market, side, collateral, risk_coef, open_price, stop_loss=None, take_profit=None):
        b = {"market": market, "side": side, "collateral": str(int(collateral)), "risk_coef": str(risk_coef), "open_price": str(open_price)}
        if stop_loss: b["stop_loss"] = str(stop_loss)
        if take_profit: b["take_profit"] = str(take_profit)
        return self._call("POST", "/margin-trade/v1/public/dry-run", json=b)

    def open(self, market, side, collateral, risk_coef, price, stop_loss, take_profit, now=None):
        b = {"market": market, "side": side, "collateral": str(int(collateral)), "risk_coef": str(risk_coef), "open_price": str(price),
             "stop_loss": str(stop_loss), "take_profit": str(take_profit)}
        r = self._call("POST", "/margin-trade/v1/positions", json=b)
        return self._norm(r)

    def close(self, pid, price, now=None):
        return self._norm(self._call("PATCH", f"/margin-trade/v1/positions/{pid}/close", json={"price": str(price)}))

    def positions(self, active=True):
        r = self._call("GET", "/margin-trade/v1/positions", params={"active": str(active).lower()})
        rows = r if isinstance(r, list) else r.get("positions", r.get("items", []))
        return [self._norm(x) for x in rows]

    @staticmethod
    def _norm(p):
        val = lambda x: float(x["value"]) if isinstance(x, dict) else float(x or 0)
        return {"id": p["id"], "market": p["market"], "side": p["side"], "collateral": val(p.get("collateral")), "risk_coef": float(p.get("risk_coef") or 1),
                "entry": float(p.get("open_filled_price") or p.get("open_price") or 0), "stop": float(p.get("stop_loss") or 0), "take": float(p.get("take_profit") or 0),
                "state": p.get("state"), "profit": val(p.get("profit")), "liq": float(p.get("liquidity_price") or 0), "opened_at": p.get("opened_at"),
                "close_reason": p.get("close_reason"), "raw": p}

    def equity(self, prices):
        """TMN wallet (free) + collateral + unrealised PnL of open positions. (verify: sub-account balance endpoint)"""
        free = float(self.w.balances().get("TMN", {}).get("value", 0))
        return free + sum(p["collateral"] + p["profit"] for p in self.positions())

    def sync(self, prices, now=None): return []


class PaperFutures:
    """Same interface, simulated on live bid/ask. Fees follow the Wallex dry-run: 0.25% of exposure per side + open fee + extend fee per 4h;
    liquidation when 15% of collateral is left; native stop/take trigger on bid/ask."""
    def __init__(self, cfg, cash=None):
        self.cfg, self.cash, self.pos, self.next_id, self.history = cfg, cash if cash is not None else cfg.start_equity, {}, 1, []

    def _exposure(self, p): return p["collateral"] * p["risk_coef"]
    def _pnl(self, p, mark):
        units = self._exposure(p) / p["entry"]
        return units * (mark - p["entry"]) * (1 if p["side"] == "long" else -1)
    def _fees_accrued(self, p, now):
        return self._exposure(p) * (self.cfg.open_fee + self.cfg.trade_fee) + self._exposure(p) * self.cfg.extend_fee * int((now - p["opened_ts"]) // 14400)

    def open(self, market, side, collateral, risk_coef, price, stop_loss, take_profit, now=None):
        now = now or time.time(); collateral = float(int(collateral))
        if collateral > self.cash: raise RuntimeError("insufficient paper cash")
        pid = self.next_id; self.next_id += 1
        self.cash -= collateral
        self.pos[pid] = {"id": pid, "market": market, "side": side, "collateral": collateral, "risk_coef": float(risk_coef), "entry": float(price),
                         "stop": float(stop_loss), "take": float(take_profit), "state": "open", "opened_ts": now, "profit": 0.0, "close_reason": None}
        return dict(self.pos[pid])

    def _close(self, pid, price, reason, now):
        p = self.pos.pop(pid)
        pnl = self._pnl(p, price) - self._fees_accrued(p, now)
        pnl -= self._exposure(p) * self.cfg.trade_fee            # exit side trade fee
        back = max(0.0, p["collateral"] + pnl)
        self.cash += back
        out = {**p, "state": "closed", "close_price": price, "close_reason": reason, "profit": pnl, "closed_ts": now}
        self.history.append(out)
        return out

    def set_stop(self, pid, stop):
        if pid in self.pos: self.pos[pid]["stop"] = float(stop)

    def close(self, pid, price, now=None, reason="manual"): return self._close(pid, float(price), reason, now or time.time())

    def positions(self, active=True): return [dict(p, profit=p["profit"]) for p in self.pos.values()] if active else list(self.history)

    def sync(self, prices, now=None):
        """Apply stops / takes / liquidation / expiry using current bid-ask. prices: {market: (bid, ask)}."""
        now = now or time.time(); closed = []
        for pid, p in list(self.pos.items()):
            if p["market"] not in prices: continue
            bid, ask = prices[p["market"]]
            mark = bid if p["side"] == "long" else ask               # price at which we could exit
            p["profit"] = self._pnl(p, mark) - self._fees_accrued(p, now)
            liq = p["collateral"] + p["profit"] <= self.cfg.liq_margin * p["collateral"]
            long_ = p["side"] == "long"
            hit_sl = p["stop"] and ((mark <= p["stop"]) if long_ else (mark >= p["stop"]))
            hit_tp = p["take"] and ((mark >= p["take"]) if long_ else (mark <= p["take"]))
            if liq: closed.append(self._close(pid, mark, "liquidated", now))
            elif hit_sl: closed.append(self._close(pid, mark if (mark < p["stop"]) == long_ else p["stop"], "stop_loss", now))
            elif hit_tp: closed.append(self._close(pid, p["take"], "take_profit", now))
            elif now - p["opened_ts"] > 21 * 86400: closed.append(self._close(pid, mark, "expired", now))
        return closed

    def equity(self, prices):
        eq = self.cash
        for p in self.pos.values():
            if p["market"] in prices:
                bid, ask = prices[p["market"]]; mark = bid if p["side"] == "long" else ask
                eq += p["collateral"] + self._pnl(p, mark) - self._fees_accrued(p, time.time()) - self._exposure(p) * self.cfg.trade_fee
            else: eq += p["collateral"]
        return eq


class SpotLive:
    """Real SPOT broker for the challenge account (long only, no leverage). Same interface as PaperFutures.
    Entry/exit are marketable LIMIT orders (price control + 2%-market-order rule safe), market fallback on exit.
    Stop/take are enforced by the bot loop (sync), checked every tick on bid/ask. (verify) order-response field names on the first real order."""
    def __init__(self, w: Wallex, cfg, path=None):
        import json, os
        path = path or os.path.join(cfg.log_dir, "spot_positions.json")          # on the persistent disk with the other logs (survives restarts)
        self.w, self.cfg, self.path, self.history, self.next_id = w, cfg, path, [], 1
        self.meta = {s: m for s, m in w.markets().items()}
        self.pos = {int(k): v for k, v in (json.load(open(path)) if os.path.exists(path) else {}).items()}
        if self.pos: self.next_id = max(self.pos) + 1

    def _save(self):
        import json, os
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True); json.dump(self.pos, open(self.path, "w"))

    def _qty(self, market, qty):
        step = int(self.meta[market].get("stepSize", 8)); f = 10 ** step
        return math.floor(qty * f) / f

    def _wait(self, cid, secs=12):
        end = time.time() + secs; o = {}
        while time.time() < end:
            o = self.w.order(cid).get("result", {}) or {}
            if str(o.get("status", "")).upper() in ("FILLED", "CANCELED", "REJECTED", "EXPIRED"): break
            time.sleep(1)
        return o

    @staticmethod
    def _filled(o):
        q = float(o.get("executedQty") or o.get("executed_qty") or 0)
        val = float(o.get("executedSum") or 0)
        px = val / q if q > 0 and val > 0 else float(o.get("executedPrice") or o.get("price") or 0)   # executedPrice echoes the limit price even when unfilled
        return q, px

    def open(self, market, side, collateral, risk_coef, price, stop_loss, take_profit, now=None):
        assert side == "long", "spot account is long only"
        try: price = max(price, float(self.w.depth(market)["ask"][0]["price"]))          # live best ask at order time
        except Exception: pass
        limit = price * 1.001                                                            # marketable limit: fills now, caps slippage at 0.1%
        qty = self._qty(market, collateral / limit); cid = f"BOT{int(time.time()*1000)}B"
        t0 = time.time()
        code, body = self.w.place_order(market, "BUY", "LIMIT", qty, price=self._px(market, limit), client_id=cid)
        t_ack = time.time()
        if code >= 300: raise RuntimeError(f"buy rejected {code} {body}")
        o = self._wait(cid); q, px = self._filled(o)
        if q < qty: self.w.cancel_order(cid); o = self._wait(cid, 4); q, px = self._filled(o)
        if q <= 0: raise RuntimeError("buy not filled")
        lat = {"ack_s": round(t_ack - t0, 3), "fill_s": round(time.time() - t0, 3), "orders": 1, "client_id": cid}
        pid = self.next_id; self.next_id += 1
        self.pos[pid] = {"id": pid, "market": market, "side": "long", "collateral": q * px, "risk_coef": 1.0, "entry": px, "qty": q, "stop": stop_loss, "take": take_profit,
                         "state": "open", "opened_ts": now or time.time(), "profit": 0.0, "close_reason": None, "latency": lat}
        self._save(); return dict(self.pos[pid])

    def _px(self, market, price):
        return round(price, int(self.meta[market].get("tickSize") or 0))      # tickSize = number of price decimals on Wallex

    def _best_bid(self, market):
        try: return float(self.w.depth(market)["bid"][0]["price"])
        except Exception: return None

    def sell_all(self, market, qty, price):
        """Sell `qty` for toman with escalating marketable limit orders (most TMN markets do not accept MARKET orders);
        MARKET only as the last step where Wallex enables it. Returns (filled_qty, avg_price)."""
        done = val = 0.0; min_notional = float(self.meta[market].get("minNotional") or 50_000); t0 = time.time(); acks, n = [], 0
        for k, disc in enumerate((0.0, 0.002, 0.005, 0.01, 0.02)):
            left = self._qty(market, qty - done)
            ref = self._best_bid(market) or price
            if left <= 0 or left * ref < min_notional: break
            cid = f"BOT{int(time.time() * 1000)}S{k}"
            ta = time.time(); code, body = self.w.place_order(market, "SELL", "LIMIT", left, price=self._px(market, ref * (1 - disc)), client_id=cid)
            acks.append(round(time.time() - ta, 3)); n += 1
            if code >= 300: continue
            o = self._wait(cid, 6); q, px = self._filled(o)
            if q < left: self.w.cancel_order(cid); o = self._wait(cid, 3); q, px = self._filled(o)
            done += q; val += q * px
        left = self._qty(market, qty - done)
        if left > 0 and self.meta[market].get("isMarketTypeEnable") and left * price >= min_notional:
            cid = f"BOT{int(time.time() * 1000)}SM"; ta = time.time(); self.w.place_order(market, "SELL", "MARKET", left, client_id=cid); acks.append(round(time.time() - ta, 3)); n += 1
            q, px = self._filled(self._wait(cid, 8)); done += q; val += q * (px or price)
        self.last_sell = {"ack_s": max(acks) if acks else None, "fill_s": round(time.time() - t0, 3), "orders": n}
        return done, (val / done if done else 0.0)

    def close(self, pid, price, now=None, reason="manual"):
        p = self.pos[pid]
        held = float(self.w.balances().get(p["market"][:-3], {}).get("value", p["qty"]))
        q, px = self.sell_all(p["market"], min(p["qty"], held), price)
        if q <= 0: raise RuntimeError("sell not filled")
        pnl = q * (px - p["entry"]) - (p["collateral"] + q * px) * self.cfg.trade_fee
        self.pos.pop(pid); out = {**p, "state": "closed", "close_price": px, "close_reason": reason, "profit": pnl, "closed_ts": now or time.time(), "unsold_qty": max(0.0, p["qty"] - q), "latency": getattr(self, "last_sell", None)}
        self.history.append(out); self._save(); return out

    def leftovers(self, prices):
        """Non-toman balances still worth selling (used by the end-of-challenge check)."""
        out = {}
        for asset, b in self.w.balances().items():
            v = float(b.get("value", 0) or 0); m = asset + "TMN"
            if asset != "TMN" and v > 0 and m in self.meta:
                val = v * (prices.get(m, (0, 0))[0] or self._best_bid(m) or 0)
                out[m] = (v, val)
        return out

    def positions(self, active=True): return [dict(p) for p in self.pos.values()] if active else list(self.history)

    def set_stop(self, pid, stop):
        if pid in self.pos: self.pos[pid]["stop"] = float(stop); self._save()

    def sync(self, prices, now=None):
        closed = []
        for pid, p in list(self.pos.items()):
            if p["market"] not in prices: continue
            bid, ask = prices[p["market"]]; p["profit"] = p["qty"] * (bid - p["entry"])
            why = "stop_loss" if p["stop"] and bid <= p["stop"] else ("take_profit" if p["take"] and bid >= p["take"] else None)
            if why:
                try: closed.append(self.close(pid, bid, now, reason=why))
                except Exception: pass
        return closed

    def equity(self, prices):
        eq = 0.0
        for asset, b in self.w.balances().items():
            v = float(b.get("value", 0) or 0)
            if asset == "TMN": eq += v
            elif v > 0 and asset + "TMN" in prices: eq += v * prices[asset + "TMN"][0]
        return eq
