"""Candle-replay backtester using the same Risk + PaperBroker as paper/live runs."""
from .broker import PaperBroker
from .features import summarize
from .risk import Risk, RiskConfig


def run(df, strategy, cash=1000.0, fee=0.0013, slippage=0.0005, risk_cfg=None, warmup=60, verbose=False):
    b, risk = PaperBroker(cash, fee, slippage), Risk(risk_cfg or RiskConfig())
    curve = []
    for i in range(warmup, len(df)):
        price = float(df["c"].iloc[i])
        feats = summarize(df.iloc[: i + 1])
        eq = b.equity(price)
        state = {"cash": round(b.cash, 2), "position_value": round(b.base * price, 2), "equity": round(eq, 2)}
        act = strategy.decide(feats, state)
        order = risk.check(act, eq, b.base, price, i)
        if order:
            f = b.market(order[0], order[1], price)
            if verbose and f:
                print(i, f, act.get("reason"))
        curve.append(b.equity(price))
    first = float(df["c"].iloc[warmup])
    last = float(df["c"].iloc[-1])
    peak, mdd = 0, 0
    for e in curve:
        peak = max(peak, e)
        mdd = max(mdd, (peak - e) / peak)
    return {
        "return_pct": round((curve[-1] / cash - 1) * 100, 2),
        "buy_hold_pct": round((last / first - 1) * 100, 2),
        "max_drawdown_pct": round(mdd * 100, 2),
        "trades": len(b.fills),
        "fees_paid": round(sum(f.fee for f in b.fills), 2),
        "halted": risk.halted,
    }
