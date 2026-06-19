"""
QuantPilot - backtest/engine.py
Takes a TradeHypothesis and tests it against historical price data.

This is intentionally simple and transparent (no look-ahead bias, no
survivorship bias tricks) — the point is to demonstrate sound backtesting
methodology, not to produce unrealistically good numbers.

Methodology:
  - Entry: next trading day's open after hypothesis creation date
  - Exit: `horizon_days` trading days later, at close
  - Benchmark: SPY over the identical entry/exit window
  - Sharpe: computed from daily returns over the holding period
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional

from agent.schemas import TradeHypothesis, BacktestResult, SignalDirection


class BacktestEngine:

    def __init__(self, benchmark_ticker: str = "SPY"):
        self.benchmark_ticker = benchmark_ticker
        self._price_cache = {}

    def _get_prices(self, ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Cached price fetch to avoid redundant API calls within a run."""
        key = (ticker, start.date(), end.date())
        if key not in self._price_cache:
            df = yf.download(ticker, start=start, end=end, progress=False)
            self._price_cache[key] = df
        return self._price_cache[key]

    def run(
        self,
        hypothesis: TradeHypothesis,
        entry_date: Optional[datetime] = None,
    ) -> Optional[BacktestResult]:
        """
        Backtest a single hypothesis.
        Returns None if direction is 'neutral' (no trade to test) or
        if price data is unavailable for the window.
        """
        if hypothesis.direction == SignalDirection.NEUTRAL.value or \
           hypothesis.direction == SignalDirection.NEUTRAL:
            return None

        entry_date = entry_date or hypothesis.created_at
        exit_date = entry_date + timedelta(days=hypothesis.horizon_days * 1.5)
        # *1.5 buffer to account for weekends/holidays when slicing trading days

        df = self._get_prices(hypothesis.ticker, entry_date, exit_date)
        if df.empty or len(df) < 2:
            return None

        # Use the first available trading day's open as entry,
        # and the Nth trading day's close as exit (no look-ahead).
        entry_price = float(df["Open"].iloc[0])
        exit_idx = min(hypothesis.horizon_days, len(df) - 1)
        exit_price = float(df["Close"].iloc[exit_idx])
        actual_entry_date = df.index[0].to_pydatetime()
        actual_exit_date = df.index[exit_idx].to_pydatetime()

        direction_value = (
            hypothesis.direction.value
            if isinstance(hypothesis.direction, SignalDirection)
            else hypothesis.direction
        )

        raw_return = (exit_price - entry_price) / entry_price
        pnl_pct = raw_return * 100 if direction_value == "long" else -raw_return * 100

        # ── drawdown within the holding window ──
        window_prices = df["Close"].iloc[:exit_idx + 1]
        if direction_value == "long":
            running_max = window_prices.cummax()
            drawdown = (window_prices - running_max) / running_max
        else:
            running_min = window_prices.cummin()
            drawdown = (running_min - window_prices) / window_prices
        max_drawdown_pct = float(drawdown.min() * 100)

        # ── Sharpe ratio over the holding period (annualised) ──
        daily_returns = window_prices.pct_change().dropna()
        if direction_value == "short":
            daily_returns = -daily_returns
        sharpe = None
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))

        # ── benchmark comparison ──
        bench_df = self._get_prices(self.benchmark_ticker, entry_date, exit_date)
        benchmark_pnl_pct = None
        if not bench_df.empty and len(bench_df) > exit_idx:
            bench_entry = float(bench_df["Open"].iloc[0])
            bench_exit = float(bench_df["Close"].iloc[min(exit_idx, len(bench_df) - 1)])
            benchmark_pnl_pct = (bench_exit - bench_entry) / bench_entry * 100

        return BacktestResult(
            hypothesis_id=hypothesis.id or 0,
            ticker=hypothesis.ticker,
            direction=hypothesis.direction,
            entry_date=actual_entry_date,
            exit_date=actual_exit_date,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_pct=round(pnl_pct, 3),
            max_drawdown_pct=round(max_drawdown_pct, 3),
            sharpe_ratio=round(sharpe, 3) if sharpe is not None else None,
            win=pnl_pct > 0,
            benchmark_pnl_pct=round(benchmark_pnl_pct, 3) if benchmark_pnl_pct is not None else None,
        )

    def run_batch(self, hypotheses: list[TradeHypothesis]) -> list[BacktestResult]:
        """Backtest a list of hypotheses, skipping neutrals and failures."""
        results = []
        for hyp in hypotheses:
            try:
                r = self.run(hyp)
                if r is not None:
                    results.append(r)
            except Exception as e:
                print(f"[BacktestEngine] Failed to backtest {hyp.ticker}: {e}")
        return results
