"""
QuantPilot - tests/test_backtest.py
Tests for the backtest engine and stats aggregation.
These do NOT require an OpenAI API key — they construct TradeHypothesis
objects manually and test the pure backtesting/stats logic.

Price-fetching tests are marked to skip gracefully if no network access
is available in CI (yfinance needs internet).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime, timedelta, timezone

from agent.schemas import TradeHypothesis, SignalDirection, BacktestResult
from backtest.stats import compute_portfolio_stats, compute_vs_benchmark


# ─────────────────────────────────────── schema validation

class TestSchemas:

    def test_trade_hypothesis_defaults(self):
        hyp = TradeHypothesis(
            ticker="AAPL",
            direction=SignalDirection.LONG,
            thesis="Strong earnings momentum.",
            confidence=0.7,
        )
        assert hyp.ticker == "AAPL"
        assert hyp.horizon_days == 5
        assert hyp.reasoning_trail == []

    def test_confidence_bounds_enforced(self):
        with pytest.raises(Exception):
            TradeHypothesis(
                ticker="AAPL",
                direction=SignalDirection.LONG,
                thesis="x",
                confidence=1.5,   # out of bounds — should raise
            )

    def test_neutral_direction_allowed(self):
        hyp = TradeHypothesis(
            ticker="XYZ",
            direction=SignalDirection.NEUTRAL,
            thesis="No clear edge found.",
            confidence=0.1,
        )
        assert hyp.direction == "neutral"


# ─────────────────────────────────────── portfolio stats (pure math, no network)

class TestPortfolioStats:

    def _make_result(self, pnl, sharpe=1.0, win=None, bench=None):
        return BacktestResult(
            hypothesis_id=1,
            ticker="TEST",
            direction="long",
            entry_date=datetime.now(timezone.utc) - timedelta(days=5),
            exit_date=datetime.now(timezone.utc),
            entry_price=100.0,
            exit_price=100.0 * (1 + pnl / 100),
            pnl_pct=pnl,
            max_drawdown_pct=-2.0,
            sharpe_ratio=sharpe,
            win=win if win is not None else pnl > 0,
            benchmark_pnl_pct=bench,
        )

    def test_empty_results(self):
        stats = compute_portfolio_stats([])
        assert stats.total_hypotheses == 0
        assert stats.win_rate == 0.0

    def test_win_rate_calculation(self):
        results = [
            self._make_result(5.0),    # win
            self._make_result(-2.0),   # loss
            self._make_result(3.0),    # win
            self._make_result(-1.0),   # loss
        ]
        stats = compute_portfolio_stats(results)
        assert stats.total_hypotheses == 4
        assert stats.win_rate == 50.0

    def test_avg_pnl_calculation(self):
        results = [self._make_result(10.0), self._make_result(-4.0)]
        stats = compute_portfolio_stats(results)
        assert stats.avg_pnl_pct == 3.0

    def test_best_worst_trade(self):
        results = [self._make_result(10.0), self._make_result(-4.0), self._make_result(2.0)]
        stats = compute_portfolio_stats(results)
        assert stats.best_trade_pnl == 10.0
        assert stats.worst_trade_pnl == -4.0

    def test_vs_benchmark_alpha(self):
        results = [
            self._make_result(10.0, bench=5.0),
            self._make_result(2.0, bench=4.0),
        ]
        comparison = compute_vs_benchmark(results)
        assert comparison["agent_avg_pnl"] == 6.0
        assert comparison["benchmark_avg_pnl"] == 4.5
        assert comparison["alpha"] == 1.5

    def test_vs_benchmark_handles_missing_data(self):
        results = [self._make_result(10.0, bench=None)]
        comparison = compute_vs_benchmark(results)
        assert comparison["alpha"] is None


# ─────────────────────────────────────── backtest engine (network-dependent, skip if offline)

class TestBacktestEngineIntegration:
    """
    These tests hit yfinance for real data. They're marked so CI can
    skip them if network access to Yahoo Finance isn't available.
    """

    @pytest.mark.network
    def test_backtest_long_position_on_known_ticker(self):
        from backtest.engine import BacktestEngine

        engine = BacktestEngine()
        hyp = TradeHypothesis(
            id=1,
            ticker="AAPL",
            direction=SignalDirection.LONG,
            thesis="test",
            confidence=0.5,
            horizon_days=5,
            created_at=datetime.now(timezone.utc) - timedelta(days=30),
        )
        result = engine.run(hyp)
        assert result is not None
        assert result.ticker == "AAPL"
        assert isinstance(result.pnl_pct, float)

    @pytest.mark.network
    def test_backtest_neutral_returns_none(self):
        from backtest.engine import BacktestEngine

        engine = BacktestEngine()
        hyp = TradeHypothesis(
            ticker="AAPL",
            direction=SignalDirection.NEUTRAL,
            thesis="no edge",
            confidence=0.1,
        )
        result = engine.run(hyp)
        assert result is None
