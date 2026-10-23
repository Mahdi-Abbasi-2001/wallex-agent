# Wallex Agentic Trading Bot: Strategy and System

Starcoach × Wallex Agentic AI hackathon, Wallex challenge. Author: Mahdi Abbasi (مهدی عباسی).

> **خلاصه فارسی.** این ایجنت در بازارهای اسپات تومانی والکس معامله می‌کند. یک سیستم قاعده‌محور هر ۳۰ دقیقه ۲۷ بازار را بررسی می‌کند و «اصلاح سالم در روند صعودی» را پیدا می‌کند. مدل gpt-6-luna (از طریق BoxAPI) با کمک ابزارهای خواندنی Wallex MCP درباره هر نامزد قضاوت می‌کند، اندازه را با ضریب اطمینان تعیین می‌کند و یک بازبین مستقل می‌تواند آن را وتو کند. کنترل ریسک در کد است و هوش مصنوعی نمی‌تواند آن را دور بزند. سفارش‌ها فقط توسط کد و از طریق Wallex API ثبت می‌شوند. هر تصمیم با دلیلش در لاگ ثبت می‌شود و بک‌تست‌ها با یک دستور قابل اجرای دوباره هستند.

## 1. Architecture

Every 30 minutes one decision passes through six stages; prices and stops are checked every 30 seconds.

```
Wallex data ──► Rule engine ──► AI trader ──► Risk guard ──► AI reviewer ──► Executor ──► Logs
(REST quotes,    (price rules,     (gpt-6-luna,    (hard limits    (independent   (Wallex API,
 1h candles)      ≤5 candidates)    Wallex MCP       in code, can    veto)          limit orders,
                                    tools)           only reject)                   stops every 30 s)
```

- The AI never talks to the exchange. It proposes; code sizes, checks and executes.
- If the AI is unavailable, the bot opens nothing and keeps protecting open positions (stops, targets, end-of-challenge sell-off).
- Code: `bot/signals.py` (rules), `bot/agents.py` (AI roles), `bot/risk.py` (risk guard), `bot/futures.py` (`SpotLive` real broker, `PaperFutures` paper broker), `bot/mcp.py` (Wallex MCP client), `bot/runner.py` (main loop), `bot/replay.py` (backtest), `bot/config.py` (all settings, overridable by environment variables).

## 2. The account and its constraints (verified on the live account)

- Prop sub-account with 5,000,000 TMN; spot only (no margin, no shorting, no withdrawals).
- Fees on TMN markets: maker 0.25%, taker 0.30% (read from the account through the API and MCP, confirmed on real fills). A round trip costs about 0.6% plus spread.
- 62 of 187 TMN markets are tradable (the organizers blocked 125; the list is in `bot/blocked_tmn.txt`, and USDTTMN is refused by Wallex as well). The bot drops blocked markets automatically.
- Only 4 TMN markets accept MARKET orders (BTC, ETH, USDT, XAUT), so entries and exits use marketable limit orders.
- The score is measured in toman: everything is sold back to TMN before the end.

## 3. Strategy logic (rule engine)

The idea comes from the author's manual price-action trading in forex: **buy a healthy pullback in the direction of the trend**. Manual backtesting and forward testing on thousands of hours of XAUUSD and US30 (New York session) gave a 48% win rate at a 2:1 reward-to-risk ratio (expected value ≈ +0.44R per trade). The idea was turned into seven measurable price rules and re-tested on Wallex data.

**Universe:** 27 liquid TMN markets: BTC, ETH, SOL, XRP, BNB, DOGE, TRX, ADA, LTC, LINK, AVAX, PEPE, PUMP, SHIB, ARB, S, HYPE, UNI, AAVE, BCH, ATOM, WLD, APE, FLOKI, PENGU, XAUT, USOON.

**Buy candidate** (all must hold, computed on Wallex hourly candles and the live order book):

| Rule | Exact definition | Purpose |
|---|---|---|
| Market health | BTC above its 50-hour average **and** BTC 24h return > 0 (otherwise no candidates at all) | do not buy in a falling market |
| Uptrend | price above its 50-hour average **and** 72h return > 0 | trade with the trend |
| Pullback | 6h return < 0 | enter on a dip, not at the top |
| Entry zone | price between −1.5% and +0.3% of its 20-hour average | enter near short-term value; deeper means a possible breakdown |
| Relative strength | z-score of (½·24h return + ½·72h return) across the universe; top 5 go to the AI | strongest candidates first |
| Liquidity | bid-ask spread < 0.5% | low entry and exit cost |
| Exclusions | not already held, not within 4 hours of selling it, not refused by Wallex | no churning |

**Stop-loss and take-profit:** stop distance = 1.8 × twice the average hourly high-low range of the last 14 candles, clipped to [1.2%, 5%]; take-profit = 2 × stop distance. Both are checked every 30 seconds on the live bid. Examples from the live run: TRX entry 88,338, stop 85,501 (−3.2%), target 93,898 (+6.3%); XAUT stop −1.33%, target +2.66%.

**Exits:** stop-loss, take-profit, an AI exit if the uptrend is clearly broken (never in the first 4 hours), a time limit (36 h by default, extended for the live run, see §8), and the end-of-challenge sell-off.

**End of challenge** (relative to `CHALLENGE_END`, currently Monday 1405/08/04 23:50 Tehran): no new buys in the last 2 hours, staggered limit sells from 30 minutes before, forced sale 5 minutes before (stepped limit orders at the bid, then −0.2%, −0.5%, −1%, −2%; MARKET only where Wallex allows it), then a leftover check that sells any coin worth more than 50,000 TMN.

## 4. Use of AI and Wallex tools

**Model:** `gpt-6-luna` through BoxAPI (chosen by backtest, §6). Every call is logged with input, reply, latency and tokens.

**Roles**

1. **Trader** sees the market regime, BTC's 6h/24h returns and distance from its averages, the candidates (entry, spread, 6/24/72h returns, distance from the 20h and 50h averages, hourly volatility, stop and target) and open positions (entry, P&L, stop, target, age). It may call Wallex MCP tools on its own (at most 2 calls), then decides: buy one candidate or none, with **conviction** low / medium / high (size 50% / 75% / 100% of the guard's maximum), and proposes exits only if a position's uptrend is clearly broken. It writes a reason for every decision.
2. **Reviewer** is a separate call that sees the proposal, its size and the loss if the stop hits, and approves or vetoes it for concrete red flags (a breakdown instead of a pullback, a drop still accelerating, BTC falling hard, a wide spread).
3. **Exit manager** is the trader's second task: every 30 minutes it reviews open positions; the default is to hold.

**What the AI receives (lean data, chosen by ablation):** only what the rules do not already decide. An earlier version also had an analyst role, all 27 markets' numbers, Binance prices and 8-hourly self-tuning; these were removed because they did not improve results, made runs less reproducible and roughly tripled token use.

**Wallex MCP** (`https://api.wallex.ir/mcp`, client in `bot/mcp.py`):
- The AI's research tools: `wallex_basic_candle_history`, `wallex_basic_get_market_depth`, `wallex_basic_get_latest_trades`. Calls and results are logged as `mcp_research`.
- At every start the bot reads the credit account's balances (`wallex_credit_sub_account_balances_details`) and fees (`wallex_basic_get_account_fees`) through the MCP and logs them (`mcp_account`).
- The client enforces a read-only allow-list. State-changing MCP tools (orders, transfers, account changes) are deliberately out of the AI's reach.

**Wallex API** is used by code for live prices (`/v2/depth/all`, `/v1/depth`), candles, balances, placing, checking and cancelling orders. Orders are placed only after the risk guard; the measured time to fill is about 0.15–0.21 s.

**The AI cannot:** buy outside the candidate list, enlarge a position beyond the guard's size, place orders or call state-changing tools, close a position in its first 4 hours, or bypass any risk limit or the end-of-challenge sell-off.

## 5. Capital and risk management

**Position size** = the smallest of (a) 33% of equity, (b) the amount that loses 1.5% of equity if the stop is hit, (c) free cash minus a 3% fee buffer, then × the AI's conviction (0.5 / 0.75 / 1.0). The AI can only make a position smaller.

**Account limits** (defaults in `bot/config.py`; live overrides in §8):

| Limit | Default |
|---|---|
| Open positions | 2 (live: 3), one new position per decision cycle |
| Leverage / shorting | none (spot only) |
| Daily loss cap | 1% (stops new entries for the day) |
| Drawdown halt | 5% from peak (sell everything, stop) |
| Profit lock | +3%: one position at a time; +6%: sell everything, stop |
| Re-entry cooldown | 4 hours per market |
| AI outage | no new entries; protection continues |
| Kill switch | file `KILL` in the log folder: sell everything and stop |

## 6. Evidence: backtests and shadow test

All tests use the bot's own code (`bot/replay.py`) on real Wallex 1-minute and 1-hour candles, real 0.3% fees, real spreads (stored with the data) and the end-of-challenge sell-off. In AI backtests the MCP candle tool is replaced by history up to the simulated minute, so the model cannot see the future.

**Rejected first.** Price-action and breakout strategies on 1-minute to 1-hour bars showed no edge even before fees; buying the strongest-momentum coin lost on average and hit the −5% halt 4 times in 10 windows.

**Choosing the rule set** (10 three-day windows, 30 days, 1405/06/30–1405/07/29):

| Rule set | Avg return | Positive windows | Worst window | Max drawdown | Trades | Win rate |
|---|---|---|---|---|---|---|
| **Pullback + BTC gate, 2 positions, 1% daily cap (chosen)** | **+0.75%** | 5/10 | **−0.92%** | **3.06%** | 37 | 51% |
| Pullback without BTC gate | +0.98% | 5/10 | −5.20% | 5.20% | 51 | 47% |
| Momentum + BTC gate | −1.33% | 4/10 | −5.06% | 5.15% | 56 | 38% |
| Momentum (buy the strongest) | −1.95% | 4/10 | −5.13% | 5.25% | 62 | 34% |

(The comparison rows used 3 positions and a 2% daily cap; the chosen row was re-run with the final settings and end-game timing.)

**Challenge-length check** (15 two-day windows): the chosen settings averaged +0.09% (6/15 positive, worst −3.62%, max drawdown 4.78%). Larger positions (50%) did not add return but raised drawdown to 5.30%; closer targets (1× or 1.5× the stop) did worse. Over 2 days the rules alone are about break-even.

**Rules vs rules + AI** (5 three-day windows: two falling, one flat, two rising):

| Window | BTC | Rules: return / max DD | Rules + gpt-6-luna: return / max DD |
|---|---|---|---|
| 1405/07/14–17 | −4.1% | −0.09% / 0.5% | −0.07% / 0.3% |
| 1405/07/17–20 | −2.8% | −0.38% / 3.1% | −0.01% / 1.8% |
| 1405/07/02–05 | +1.0% | −0.84% / 1.7% | −0.27% / 1.2% |
| 1405/07/26–29 | +6.1% | +1.49% / 1.3% | +1.23% / 1.1% |
| 1405/07/08–11 | +7.0% | +1.37% / 2.8% | +1.47% / 2.6% |
| **Average** | | **+0.31%, 17 trades** | **+0.47%, 18 trades** |

The AI lowered drawdown in all five windows. The difference in average return is small and within noise.

**Model choice** (same 3-day window): rules only +0.45%; `gpt-5.4-mini` −2.48% (it closed positions 30–90 minutes after buying them, paying fees each time); `gpt-6-luna` +1.75% with the lowest drawdown (0.89%).

**AI design fixes found in the logs:** (1) the first prompt let the AI close positions for "being extended", so entry and exit decisions were separated and early exits were limited to a broken uptrend after 4 hours; (2) the next version treated the pullback itself as a warning sign and traded 3 times in 15 days, so the strategy was explained to the model and refusal was limited to concrete red flags (18 trades).

**Shadow test** (live prices, paper orders; Thursday 1405/07/30 01:49–03:42): 4 decision cycles, 6 AI calls with 0 errors (≈4.5 s each), 2 paper buys (S, high conviction; DOGE, medium) and one MCP-based skip (TRX: "0.47% spread is wide relative to fees; recent trades below the ask"). Before it, a 50,000 TMN live test trade (PEPETMN) verified the order path: filled at the live ask in 0.21 s, fee 0.3%, total cost 341 TMN.

**What we do not claim:** guaranteed profit over a short window, a large edge after 0.6% fees per trade, statistical certainty from 30 days of data, or the same behaviour in every market.

## 7. Re-running and evaluating

```bash
pip install -r requirements.txt
cp .env.example .env                                   # keys: see README
python tests/test_bot.py                                # risk guard, fees, stops, end game, sell ladder, pipeline
python -m bot.replay --days 3 --rules --end 1792605600 # rules-only backtest of one window (identical on every run)
python -m bot.replay --days 3 --model gpt-6-luna --end 1792605600   # same window with the AI agents
python scripts/run_variants.py                          # the 10-window rule-set table
python scripts/run_2day.py                              # the 15 two-day windows
python scripts/summarize.py logs/w10                    # summarise any folder of replays
python -m bot --mode shadow                             # live prices, paper orders
```

- The first replay of a window downloads its data from Wallex and stores it (with the spreads) in `data/replay/`; later runs reuse it, so rules-only results are identical. AI runs can differ slightly from run to run because model answers vary; every prompt and reply is logged.
- Each run writes `decisions.jsonl` (every decision with its inputs, the AI's reason, the reviewer's verdict, the risk guard's verdict, fills, order latency and P&L), `llm_calls.jsonl` (full prompts and replies with tokens), `equity.jsonl` and `summary.json`.
- Every bot start records the git commit, a SHA-256 fingerprint of the bot's source files and the full configuration.

## 8. The live run

- Hosted on Liara (Docker, persistent disk, health endpoint `/health`), mode `live`, model `gpt-6-luna` via BoxAPI.
- **Code that traded:** commit `188ffe8` (tag `challenge-live`); the bot's start records in the live log show this commit and its source fingerprint. Later commits only add backtest options and an optional `AI_STYLE=picky`, which was not used live.
- Live logs: `logs/live/` in this repository (decisions, AI calls, equity).

Setting changes during the challenge (all applied as environment variables and logged in each start record):

| Time (Tehran) | Change | Reason |
|---|---|---|
| Thu 1405/07/30 03:42 | Live start: 2 positions, 1% daily cap, 5% drawdown halt | |
| Thu 10:54 / 10:55 | Daily loss cap 10%, drawdown halt 10% | the 1% cap (which counts open positions at market value) had stopped all entries for 13 hours |
| Thu 11:12 | Max positions 3 | more chances for the AI while the daily cap was relaxed |
| Thu 17:53 | Close-out mode (`NO_NEW_ENTRIES=1`) | manage the three open positions only |
| Fri 1405/08/01 14:31 | End moved to Mon 1405/08/04 23:50; time limit 240 h | the organizers extended the challenge to Monday |

## 9. Limitations

- Spot and long-only: in a falling market the only defence is holding toman.
- Fees (≈0.6% per round trip) are large relative to typical moves; the strategy trades rarely and targets larger moves.
- 30 days of data and a few dozen trades cannot prove an edge; the evidence supports controlled risk and a working decision process more than a profit claim.
- AI decisions are not bit-for-bit reproducible; the logs are the record of what the model saw and said.
