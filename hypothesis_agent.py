name: QuantPilot CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Cache pip
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}

      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run offline-safe tests
        # Skips tests marked 'network' since CI runners may have flaky
        # access to Yahoo Finance — these are still run locally / on demand.
        run: python -m pytest tests/ -v -m "not network" --tb=short

      - name: Check core imports
        run: |
          python -c "from agent.schemas import TradeHypothesis, BacktestResult; print('schemas OK')"
          python -c "from backtest.engine import BacktestEngine; print('backtest engine OK')"
          python -c "from backtest.stats import compute_portfolio_stats; print('stats OK')"
