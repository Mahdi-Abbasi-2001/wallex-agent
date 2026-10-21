"""Scenario playbook: pick the agent for the rules the organizers announce.
 make_agent(Rules(can_short=..., max_leverage=..., style='hold'|'trade', objective='rank'|'safe', taker_fee_pct=...)) -> Agent
Defaults come from playbook_search.py (selected on weeks before 2025-07, checked on later weeks; 10 majors; 7-day contests).
Everything is momentum on daily closes; 'rank' maximises the chance of a top finish, 'safe' keeps variance low."""
from dataclasses import dataclass

from .agents import Basket, Concentrated

BTC = "BTCUSDT"


@dataclass(frozen=True)
class Rules:
    can_short: bool = False
    max_leverage: float = 1.0
    style: str = "hold"            # 'hold' = pick once for the week, 'trade' = re-pick every 1-3 days
    objective: str = "rank"        # 'rank' = top finish (winner takes most), 'safe' = low variance
    taker_fee_pct: float = 0.13    # one-way taker fee in percent (0 if the engine charges nothing)
    contest_days: int = 7          # expected trading period; <=2 days with a spot 'safe' objective selects the cash-allowed Basket


def make_agent(r: Rules):
    lev = max(1.0, min(3.0, r.max_leverage)); cheap = r.taker_fee_pct <= 0.03
    if r.objective == "basket" or (r.objective == "safe" and r.contest_days <= 2 and not r.can_short and lev < 2):
        return Basket(L=1, thr=0.03, kmax=3, btc_gate=True)                    # criteria basket, flat when nothing qualifies (1-2 day contests)
    if r.objective == "safe":                                                   # low variance: diversified, trend-gated, no leverage
        if r.style == "hold": return Concentrated(L=14, k=3, rb=7, gate=BTC)
        return Concentrated(L=1, k=2, rb=1) if cheap else Concentrated(L=7, k=2, rb=2)
    if r.can_short:                                                             # short the weakest-momentum coin (works in both halves of the sample)
        if r.style == "hold": return Concentrated(L=14, k=1, rb=7, lev=lev, short_only=True)
        return Concentrated(L=1, k=1, rb=1, lev=lev, short_only=True) if cheap else Concentrated(L=7, k=1, rb=3, lev=lev, short_only=True)
    if lev >= 2:                                                                # leveraged long only: weaker, degrades out of sample
        if r.style == "hold": return Concentrated(L=14, k=1, rb=7, lev=lev)
        return Concentrated(L=1, k=1, rb=1, lev=lev) if cheap else Concentrated(L=3, k=1, rb=3, lev=lev)
    if r.style == "hold": return Concentrated(L=30, k=1, rb=7, stop=0.15)        # spot: strongest 30-day momentum coin, hold the week
    return Concentrated(L=1, k=1, rb=1) if cheap else Concentrated(L=3, k=1, rb=3)


# Evidence (10 majors, 7-day contests, vs 30 simulated competitors; train = weeks before 2025-07, test = later weeks; chance: P(#1)=3.2%, P(top3)=9.7%)
EVIDENCE = {
    "spot / hold / rank   : top-1 momentum 30d, weekly, 15% stop": "test P(top3) 17%, P(#1) 3.9% (about chance); median -0.2%, p10 -10%, p90 +9%",
    "spot / hold / safe   : top-3 momentum 14d, BTC>SMA50 gate": "test P(top3) 10%; median 0%, p10 -4.8%, p90 +5.0%, no ruin",
    "spot / trade / rank  : top-1 momentum 1d, daily (zero fees) | 3d every 3 days (fees)": "train->test P(#1) 13%->12% | 8%->8%; median -0.5% | -2.2%; mean +0.6% | -1.1%",
    "margin / hold / rank : SHORT weakest 14d momentum, x3, weekly": "train P(#1) 10% -> test 13-16%; median 0%, p10 -22%, p90 +26%, P(<-30%) 7%",
    "margin / trade / rank: SHORT yesterday's loser x3 daily (no fees) | SHORT weakest 7d every 3 days (fees)": "test P(#1) 15% (ruin 1%, median +1.5%) | 11% (ruin 7%)",
    "spot / 1-2 days / safe: Basket (coins up >3% over 1d and above SMA20, max 3, flat if BTC<SMA20)": "test mean -0.1..+0.04% vs top-1 -0.4..-0.5% and BTC -0.3%; 80-95% of windows in cash; last week 0.0% vs -7.8% (top-1), -3% (BTC)",
    "no-short leverage    : top-1 momentum 14d x3, weekly": "train P(#1) 11% -> test 6.5% (degrades), median -2.4%, ruin 11%",
}
