# Wallex Agentic-AI trading bot (spot, long-only, TMN markets)

An LLM-driven trading engine for the Starcoach x Wallex hackathon. A rule engine proposes candidates, four LLM roles decide (BoxAPI), a hard risk guard approves, and every decision is logged.

## Pipeline
`Wallex quotes + 1h candles -> signal engine (momentum / trend / ATR / BTC regime) -> LLM analyst -> LLM trader (picks from candidates, or "no trade") -> hard risk guard -> LLM risk reviewer -> dry-run -> position with native stop-loss/take-profit -> logs`

* `bot/signals.py` - features, BTC regime, ranked candidates (strongest relative momentum above its trend; cash in a down regime).
* `bot/agents.py` - analyst, trader, reviewer, reflection (tunes `min_score` / `max_lev` within bounds every 8 h).
* `bot/risk.py` - hard limits: spot only (the challenge account has no margin/leverage), <= 3 positions, each <= 33% of equity, per-trade stop risk <= 1.5% of equity, every position has a stop/take enforced every 30 s, daily loss cap -2%, drawdown halt -5%, profit lock (+3% one position, +6% flat), cooldowns, end-game phases.
* `bot/futures.py` - `SpotLive` (real Wallex spot orders: marketable limit entry/exit, market fallback) and `PaperFutures` (paper spot broker with the same fee model, used for shadow mode and tests).
* `bot/runner.py` - main loop. The LLM can never place an order; if it is down the bot only manages and exits positions.
* End game: no new positions 6 h before the end, staggered unwind from 3 h before, forced market close 45 min before, then everything is TMN.

## Run
```bash
pip install -r requirements.txt
cp .env.example .env            # fill LLM_PROVIDER=boxapi, BOXAPI_KEY, BOXAPI_MODEL, WALLEX_API_KEY, WALLEX_SUB_ACCOUNT_CLIENT_ID
python -m bot --mode shadow     # real prices, simulated positions (default, safe)
CONFIRM_LIVE=yes python -m bot --mode live --confirm-live   # real positions
touch logs/KILL                 # kill switch: closes everything and stops
python tests/test_bot.py        # risk guard / paper broker / end game / pipeline tests
```
Deploy on Liara: `liara deploy` (Dockerfile + `liara.json`, logs on a persistent disk at `/data`).

## Logs (deliverables)
* `logs/decisions.jsonl` - one record per decision: candidates seen, the AI's reason, the risk-guard verdict, reviewer verdict, dry-run (fees, liquidation price), open/close with price and PnL, halts, errors.
* `logs/llm_calls.jsonl` - full prompt and reply of every LLM call.
* `logs/equity.jsonl` - equity, return and drawdown every tick.
