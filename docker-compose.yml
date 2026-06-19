# 🤖 QuantPilot — LLM-Driven Autonomous Trading Research Agent

> An AI agent that reads financial news, SEC filings, and market data,
> then autonomously generates and backtests trading hypotheses —
> with a **full audit trail** of its reasoning.

---

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi)
![LangChain](https://img.shields.io/badge/LangChain-Agent-1C3C3C)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)
![License](https://img.shields.io/badge/License-MIT-green)

> ⚠️ **Disclaimer:** QuantPilot is a research / educational project demonstrating
> agentic AI system design. It is **not financial advice** and is not intended
> for live trading. All "trades" are simulated backtests against historical data.

---

## What it does

| Stage | What happens |
|---|---|
| **1. Research** | A LangChain tool-calling agent gathers news, SEC filings, price action, and fundamentals for a ticker |
| **2. Hypothesis** | The agent reasons step-by-step and emits a structured trade thesis (direction, confidence, horizon) |
| **3. Audit trail** | Every tool call and reasoning step is persisted to Postgres — fully inspectable, nothing is a black box |
| **4. Backtest** | A custom backtesting engine tests the hypothesis against real historical price data, no look-ahead bias |
| **5. Dashboard** | FastAPI serves win rate, Sharpe ratio, drawdown, and performance vs. SPY benchmark |

---

## Why this is a strong portfolio project

This isn't a toy chatbot wrapper. It demonstrates:
- **Agentic AI system design** — tool-calling agent with custom tools, not just a single prompt
- **Full observability** — every agent decision is logged and auditable (a real requirement in fin-tech / ML systems)
- **Sound backtesting methodology** — no look-ahead bias, benchmark-relative performance
- **Production architecture** — FastAPI + Postgres + Docker Compose, not a notebook hack

---

## Architecture

```
QuantPilot/
├── agent/
│   ├── schemas.py            # Pydantic models: TradeHypothesis, ReasoningStep, BacktestResult
│   ├── tools.py               # LangChain tools: news search, SEC EDGAR, price/fundamentals
│   └── hypothesis_agent.py    # The core agent — orchestrates tools + LLM reasoning
├── backtest/
│   ├── engine.py               # Backtesting logic: entry/exit, PnL, Sharpe, drawdown, benchmark
│   └── stats.py                 # Portfolio-level aggregation for the dashboard
├── api/
│   ├── main.py                  # FastAPI endpoints
│   └── db.py                     # SQLAlchemy models + persistence
├── scripts/
│   └── run_pipeline.py            # CLI: run the whole pipeline without the API server
├── tests/
│   └── test_backtest.py            # Tests (network-independent ones run in CI)
├── docker-compose.yml
└── Dockerfile
```

---

## Quick start (CLI — fastest way to see it work)

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/QuantPilot.git
cd QuantPilot

# 2. Install
pip install -r requirements.txt

# 3. Set your OpenAI API key
cp .env.example .env
# edit .env and paste your key — get one at https://platform.openai.com/api-keys
export OPENAI_API_KEY=sk-...          # or: source .env (with export added)

# 4. Run the pipeline on a few tickers
python scripts/run_pipeline.py --tickers AAPL MSFT NVDA
```

You'll see the agent's research process print live (verbose mode), followed by
a generated thesis, a backtest result, and a portfolio summary.

**Cost note:** each ticker costs roughly $0.01–0.05 in OpenAI API usage with `gpt-4o-mini`.

---

## Running the full stack (API + Postgres dashboard)

```bash
# Requires Docker + Docker Compose
echo "OPENAI_API_KEY=sk-..." > .env
docker compose up --build
```

Then visit:
- `http://localhost:8000/docs` — interactive API docs (Swagger UI)
- `POST /hypotheses/generate?ticker=TSLA` — generate a new hypothesis
- `GET /dashboard/stats` — portfolio-level win rate, Sharpe, PnL
- `GET /dashboard/vs-benchmark` — agent performance vs SPY

---

## Example output

```
🤖 QuantPilot — researching 1 ticker(s): NVDA

🔎 Researching NVDA...
────────────────────────────────────────────────────────
  NVDA  →  LONG   (confidence: 0.62, horizon: 5d)
────────────────────────────────────────────────────────
  Thesis: Recent data center demand commentary combined with
  above-average volume suggests continued institutional accumulation...
  Sources: Reuters: NVDA data center revenue, SEC 8-K filed 2024-...
  Reasoning steps: 4

  [✓ WIN ]  PnL: +3.42%  |  SPY: +1.10%  (Sharpe: 1.84, Max DD: -1.20%)

════════════════════════════════════════════════════════
  PORTFOLIO SUMMARY
════════════════════════════════════════════════════════
  Total hypotheses backtested : 1
  Win rate                    : 100.0%
  Avg PnL per trade            : +3.42%
  Alpha vs SPY (avg)           : +2.32%
════════════════════════════════════════════════════════
```

---

## How the backtest avoids common pitfalls

- **No look-ahead bias**: entry is always the *next* trading day's open after
  the hypothesis was generated — never the same-day close
- **Benchmark-relative**: every trade is compared against SPY over the identical
  window, so you can see if the agent actually adds alpha vs. just buying the market
- **Honest Sharpe**: computed from actual daily returns during the holding window,
  annualised properly — not a single-trade vanity number

---

## Running tests

```bash
# Fast, offline-safe tests (used in CI)
pytest tests/ -v -m "not network"

# Full suite including live yfinance calls
pytest tests/ -v
```

---

## Extending QuantPilot

| Want to... | How |
|---|---|
| Swap OpenAI for Claude/local LLM | Replace `ChatOpenAI` in `hypothesis_agent.py` with any LangChain-compatible chat model |
| Add options/futures support | Extend `BacktestResult` schema + adjust `engine.py` PnL math for the instrument type |
| Add a proper news API | Replace the DuckDuckGo scraper in `tools.py` with NewsAPI, Tiingo, or Benzinga |
| Scan an entire sector automatically | Use `HypothesisAgent.scan_universe(tickers)` with an S&P 500 ticker list |

---

## License

MIT — see [LICENSE](LICENSE).

---

*Built with [LangChain](https://python.langchain.com/), [FastAPI](https://fastapi.tiangolo.com/),
[yfinance](https://github.com/ranaroussi/yfinance), PostgreSQL, Docker*
