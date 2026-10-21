# Wallex contest playbook

Pick the agent from the rules the organizers publish. Code: `contest/playbook.py` (`make_agent(Rules(...))`), evidence: `playbook_search.py`, simulator: `contest/`.

```python
from contest.playbook import Rules, make_agent
agent = make_agent(Rules(can_short=True, max_leverage=3.0, style="hold", objective="rank", taker_fee_pct=0.13))
```

## Scenario -> agent (10 majors, 7-day contests; chance level: P(#1) 3.2%, P(top 3) 9.7% with 31 participants)

| Rules | Style | Agent | Train -> test evidence (Wallex-like fees) |
|---|---|---|---|
| Spot only, no leverage | hold | strongest 30-day momentum coin, weekly, 15% stop | P(top3) 16% -> 17%, P(#1) 3.3% -> 3.9% (about chance) |
| Spot only | trade | strongest 3-day momentum coin, re-pick every 3 days (1-day, daily if fees ~ 0) | P(#1) 8.2% -> 8.2% (13% -> 12% with no fees); median -2% |
| Leverage, no shorting | hold | strongest 14-day momentum, x3 | P(#1) 11% -> 6.5% (degrades); median -2.4%, ruin 11% |
| **Shorting + leverage** | hold | **short the weakest 14-day momentum coin, x3, weekly** | P(top3) 24% -> 31%, P(#1) 9.9% -> 16.5%; median 0%, p10 -22%, p90 +26%, ruin 7% |
| Shorting + leverage | trade | short the weakest 7-day coin, re-pick every 3 days (short yesterday's loser daily if fees ~ 0) | P(#1) 7% -> 11% (10% -> 15% no fees, ruin 1%) |
| Any, objective = low variance | either | top-3 momentum 14d with BTC>SMA50 gate (spot) | P(top3) 9% -> 10%, never ruined, p10 -4.8%, p90 +5% |

Takeaways: weekly holding beats daily trading once fees are charged; the only setup that beat chance clearly in both halves is shorting the weakest coin with leverage.

## What would change the choice
- **Scoring rule.** `rank` assumes the winner takes most of the prize. If it is risk-adjusted or the agent is also judged on risk control, use `objective="safe"`.
- **Fees / engine.** Zero-fee engines favour daily re-picking. A book-walking engine on thin Wallex markets makes large starting balances undeployable (see `run_contest.py --cash 100000 --fills book`).
- **Borrow/funding cost and liquidation rules** are not modelled (`SimConfig.borrow_daily=0`).

## Caveats
- The shorting edge was strongest in 2025-26, a falling alt market. It also beat chance in the earlier bullish weeks, but treat it as regime-sensitive.
- The competitor field is assumed (random holders, busy random traders, simple trend bots, YOLO leveraged bets). P(#1) estimates carry roughly +/-5 points of noise (about 30-60 independent weeks).
- Results on the 35-coin liquid universe look stronger but come from a few pump weeks (top 5% of windows produce more than 100% of the total return), a survivor-only coin list, and coins with almost no volume on Wallex. Treat them as an upper bound.
- A bug found on the way: a long/short pair at x3 used to collapse into short-only because of the 3x gross cap. Long+short now splits leverage per side.

## 1-2 day contests (majors, Wallex fees, 30-agent field; outputs in data/out/pb_len1.txt, pb_len2.txt)
Chance level with 31 participants: P(#1) 3.2%. Test = windows after 2025-07.
| Rules | Agent | P(#1) 1-day | P(#1) 2-day |
|---|---|---|---|
| Spot, hold or trade | top-1 momentum coin, 1-3d lookback (1d lookback for 1-day contests) | 5.1% | 4.7-7.7% |
| Leverage, no shorting | top-1 momentum x3 | 6-8.5% (train 10-12%, degrades) | 6-8% |
| **Shorting + leverage** | **short the weakest 14-30d momentum coin x3** | **14.5-18% (train 9-12%)** | **11-12%** |
Note: spot median return is about 0%, and the short x3 median is 0% with p10 -10 to -16%, p90 +11 to +15%. Edge is thin and regime-sensitive; ruin 1-3%.
