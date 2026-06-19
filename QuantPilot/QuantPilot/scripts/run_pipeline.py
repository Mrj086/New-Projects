#!/usr/bin/env python3
"""
QuantPilot - scripts/run_pipeline.py
End-to-end CLI: generate hypotheses for a list of tickers, backtest them,
print a results table. No API server needed — good for demos and quick runs.

Usage:
    python scripts/run_pipeline.py --tickers AAPL MSFT NVDA TSLA
    python scripts/run_pipeline.py --tickers AAPL --save-db
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.hypothesis_agent import HypothesisAgent
from backtest.engine import BacktestEngine
from backtest.stats import compute_portfolio_stats, compute_vs_benchmark


def print_hypothesis(hyp):
    print(f"\n{'─' * 60}")
    print(f"  {hyp.ticker}  →  {hyp.direction.upper() if isinstance(hyp.direction, str) else hyp.direction.value.upper()}"
          f"   (confidence: {hyp.confidence:.2f}, horizon: {hyp.horizon_days}d)")
    print(f"{'─' * 60}")
    print(f"  Thesis: {hyp.thesis}")
    if hyp.source_refs:
        print(f"  Sources: {', '.join(hyp.source_refs[:3])}")
    print(f"  Reasoning steps: {len(hyp.reasoning_trail)}")


def print_backtest(result):
    win_marker = "✓ WIN " if result.win else "✗ LOSS"
    bench = f"  |  SPY: {result.benchmark_pnl_pct:+.2f}%" if result.benchmark_pnl_pct is not None else ""
    print(f"  [{win_marker}]  PnL: {result.pnl_pct:+.2f}%{bench}  "
          f"(Sharpe: {result.sharpe_ratio if result.sharpe_ratio else 'N/A'}, "
          f"Max DD: {result.max_drawdown_pct:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Run QuantPilot's agent + backtest pipeline")
    parser.add_argument("--tickers", nargs="+", required=True, help="Tickers to research, e.g. AAPL MSFT")
    parser.add_argument("--save-db", action="store_true", help="Persist results to the database")
    args = parser.parse_args()

    print(f"\n🤖 QuantPilot — researching {len(args.tickers)} ticker(s): {', '.join(args.tickers)}\n")

    try:
        agent = HypothesisAgent()
    except RuntimeError as e:
        print(f"\n❌ {e}\n")
        sys.exit(1)

    engine = BacktestEngine()
    hypotheses = []
    backtest_results = []

    for ticker in args.tickers:
        print(f"\n🔎 Researching {ticker}...")
        try:
            hyp = agent.generate_hypothesis(ticker)
            hypotheses.append(hyp)
            print_hypothesis(hyp)

            result = engine.run(hyp)
            if result:
                backtest_results.append(result)
                print_backtest(result)
            else:
                print("  (skipped backtest — neutral direction or no data)")

            if args.save_db:
                from api.db import init_db, SessionLocal, save_hypothesis, save_backtest_result
                init_db()
                db = SessionLocal()
                record = save_hypothesis(db, hyp)
                if result:
                    result.hypothesis_id = record.id
                    save_backtest_result(db, result)
                db.close()

        except Exception as e:
            print(f"  ⚠️  Error researching {ticker}: {e}")

    # ── summary ──
    if backtest_results:
        stats = compute_portfolio_stats(backtest_results)
        vs_bench = compute_vs_benchmark(backtest_results)

        print(f"\n{'═' * 60}")
        print("  PORTFOLIO SUMMARY")
        print(f"{'═' * 60}")
        print(f"  Total hypotheses backtested : {stats.total_hypotheses}")
        print(f"  Win rate                    : {stats.win_rate}%")
        print(f"  Avg PnL per trade            : {stats.avg_pnl_pct:+.2f}%")
        print(f"  Avg Sharpe                   : {stats.avg_sharpe}")
        print(f"  Best trade                   : {stats.best_trade_pnl:+.2f}%")
        print(f"  Worst trade                  : {stats.worst_trade_pnl:+.2f}%")
        if vs_bench["alpha"] is not None:
            print(f"  Alpha vs SPY (avg)           : {vs_bench['alpha']:+.2f}%")
        print(f"{'═' * 60}\n")
    else:
        print("\nNo backtestable trades were generated (all neutral or data unavailable).\n")


if __name__ == "__main__":
    main()
