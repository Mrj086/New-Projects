"""
QuantPilot - backtest/stats.py
Aggregates a list of BacktestResults into portfolio-level statistics
for the dashboard.
"""

from typing import List, Optional
import numpy as np

from agent.schemas import BacktestResult, PortfolioStats


def compute_portfolio_stats(results: List[BacktestResult]) -> PortfolioStats:
    if not results:
        return PortfolioStats(
            total_hypotheses=0,
            win_rate=0.0,
            avg_pnl_pct=0.0,
            avg_sharpe=None,
            best_trade_pnl=0.0,
            worst_trade_pnl=0.0,
            total_simulated_pnl_pct=0.0,
        )

    pnls = [r.pnl_pct for r in results]
    wins = [r.win for r in results]
    sharpes = [r.sharpe_ratio for r in results if r.sharpe_ratio is not None]

    return PortfolioStats(
        total_hypotheses=len(results),
        win_rate=round(sum(wins) / len(wins) * 100, 2),
        avg_pnl_pct=round(float(np.mean(pnls)), 3),
        avg_sharpe=round(float(np.mean(sharpes)), 3) if sharpes else None,
        best_trade_pnl=round(max(pnls), 3),
        worst_trade_pnl=round(min(pnls), 3),
        total_simulated_pnl_pct=round(float(np.sum(pnls)), 3),
    )


def compute_vs_benchmark(results: List[BacktestResult]) -> dict:
    """
    Compares average hypothesis PnL against average benchmark (SPY) PnL
    over the same windows — i.e. did the agent actually beat passive holding?
    """
    paired = [(r.pnl_pct, r.benchmark_pnl_pct) for r in results if r.benchmark_pnl_pct is not None]
    if not paired:
        return {"agent_avg_pnl": None, "benchmark_avg_pnl": None, "alpha": None}

    agent_pnls = [p[0] for p in paired]
    bench_pnls = [p[1] for p in paired]

    agent_avg = float(np.mean(agent_pnls))
    bench_avg = float(np.mean(bench_pnls))

    return {
        "agent_avg_pnl": round(agent_avg, 3),
        "benchmark_avg_pnl": round(bench_avg, 3),
        "alpha": round(agent_avg - bench_avg, 3),
        "n_trades": len(paired),
    }
