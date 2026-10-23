# Wallex Agentic Trading Bot

An AI trading agent for Wallex spot toman markets, built for the Starcoach × Wallex Agentic AI hackathon by Mahdi Abbasi.

A rule engine finds healthy pullbacks in uptrending coins, `gpt-6-luna` (via BoxAPI) judges each candidate using Wallex MCP tools and sets the size, an independent AI reviewer can veto, and a hard risk guard in code approves and executes through the Wallex API. Every decision is logged with its reason.

**Full description of the strategy, AI use, risk management, backtests and the live run: [STRATEGY.md](STRATEGY.md).**

## How it works

```
Wallex data → rule engine (≤5 candidates) → AI trader (+ Wallex MCP) → risk guard → AI reviewer → executor (Wallex API) → logs
```

| File | Role |
|---|---|
| `bot/signals.py` | price rules: BTC gate, uptrend, pullback, entry zone, ranking, stop/target from volatility |
| `bot/agents.py` | AI roles: trader (entry, conviction, exits) and reviewer |
| `bot/mcp.py` | Wallex MCP client with a read-only allow-list |
| `bot/risk.py` | hard limits: size, positions, loss caps, drawdown halt, profit lock, end-game phases |
| `bot/futures.py` | `SpotLive` (real Wallex orders: marketable limit entries, stepped limit sells) and `PaperFutures` (paper broker for shadow mode and backtests) |
| `bot/runner.py` | main loop: prices and stops every 30 s, decisions every 30 min, kill switch, health endpoint |
| `bot/replay.py` | backtest: replays real Wallex minute data through the same code |
| `bot/config.py` | every setting, overridable by environment variable |
| `scripts/` | backtest batches (`run_variants.py`, `run_2day.py`, `summarize.py`), `test_trade.py`, `deploy.sh` |
| `tests/test_bot.py` | unit tests: risk guard, fees, stops, liquidation, end game, sell ladder, close-out mode, picky style, full pipeline |

`research/` holds the earlier research (strategy screening scripts, a contest simulator, notes in `research/PLAYBOOK.md`); the trading bot does not use it. Run its scripts from the repository root with `PYTHONPATH=. python research/<script>.py`. `agent/` provides the bot's Wallex REST client and LLM client; its other modules are from the first prototype.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

`.env` needs:

```
LLM_PROVIDER=boxapi
BOXAPI_KEY=...
BOXAPI_BASE_URL=https://ai.boxapi.ir/ai/v1
BOXAPI_MODEL=gpt-6-luna
WALLEX_API_KEY=...                  # the challenge (sub-)account's key
WALLEX_SUB_ACCOUNT_CLIENT_ID=...    # the sub-account's client id (GET /sub-accounts)
```

## Run

```bash
python tests/test_bot.py                                  # unit tests
python -m bot.replay --days 3 --rules --end 1792605600    # rules-only backtest of one 3-day window
python -m bot.replay --days 3 --model gpt-6-luna --end 1792605600   # same window with the AI
python scripts/run_variants.py                            # 10 three-day windows × rule-set variants
python scripts/run_2day.py                                # 15 two-day windows
python -m bot --mode shadow                               # live prices, paper orders
CONFIRM_LIVE=yes python -m bot --mode live --confirm-live # real orders
touch logs/KILL                                           # kill switch: sell everything and stop
```

Useful settings (environment variables): `CHALLENGE_END` (ISO time), `MAX_POSITIONS`, `DAILY_LOSS_CAP`, `MAX_DRAWDOWN`, `NO_NEW_ENTRIES=1` (close-out mode), `MAX_HOLD_S`, `AI_STYLE` (`balanced` or `picky`), `BOT_MODEL`, `LOG_DIR`. See `bot/config.py` for all.

## Deploy (Liara)

`Dockerfile` + `liara.json` (port 8000, persistent disk at `/data`). Set the keys above and `BOT_MODE=shadow` (or `BOT_MODE=live`, `BOT_FLAGS=--confirm-live`, `CONFIRM_LIVE=yes`) as app environment variables, then:

```bash
bash scripts/deploy.sh      # records the git commit in the image and deploys
```

`GET /health` returns the last tick time, equity, drawdown, phase and open positions.

## Logs

| File | Content |
|---|---|
| `decisions.jsonl` | one record per event: start (git commit, source fingerprint, config), each cycle's candidates, MCP research, AI proposal and reason, reviewer verdict, risk-guard verdict, fills with order latency, closes with reason and P&L |
| `llm_calls.jsonl` | every model call: full prompt, reply, latency, tokens |
| `equity.jsonl` | equity, return and drawdown over time |

The live run's logs are in `logs/live/`. The code that traded live is commit `188ffe8` (tag `challenge-live`).
