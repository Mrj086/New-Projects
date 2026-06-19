"""
QuantPilot - api/main.py
FastAPI dashboard backend.

Endpoints:
  POST /hypotheses/generate   — run the agent on a ticker, persist + return result
  GET  /hypotheses            — list stored hypotheses
  GET  /hypotheses/{id}       — full detail incl. reasoning trail
  POST /backtest/run/{id}     — backtest a specific stored hypothesis
  GET  /backtest/results      — list all backtest results
  GET  /dashboard/stats       — aggregate portfolio stats
  GET  /dashboard/vs-benchmark — agent performance vs SPY
"""

import json
from typing import List
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from api.db import init_db, get_db, save_hypothesis, save_backtest_result, HypothesisRecord, BacktestRecord
from agent.schemas import TradeHypothesis, ReasoningStep, BacktestResult
from agent.hypothesis_agent import HypothesisAgent
from backtest.engine import BacktestEngine
from backtest.stats import compute_portfolio_stats, compute_vs_benchmark

app = FastAPI(
    title="QuantPilot API",
    description="LLM-driven autonomous trading research agent — backtest & audit trail API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

backtest_engine = BacktestEngine()


@app.on_event("startup")
def on_startup():
    init_db()


# ───────────────────────────────────────── hypotheses

@app.post("/hypotheses/generate")
def generate_hypothesis(ticker: str, db: Session = Depends(get_db)):
    """Run the research agent on a ticker and persist the resulting hypothesis."""
    try:
        agent = HypothesisAgent()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    hypothesis = agent.generate_hypothesis(ticker)
    record = save_hypothesis(db, hypothesis)
    return {"id": record.id, "ticker": record.ticker, "direction": record.direction,
            "thesis": record.thesis, "confidence": record.confidence}


@app.get("/hypotheses")
def list_hypotheses(db: Session = Depends(get_db)):
    records = db.query(HypothesisRecord).order_by(HypothesisRecord.created_at.desc()).all()
    return [
        {
            "id": r.id, "ticker": r.ticker, "direction": r.direction,
            "thesis": r.thesis, "confidence": r.confidence,
            "horizon_days": r.horizon_days, "created_at": r.created_at,
        }
        for r in records
    ]


@app.get("/hypotheses/{hypothesis_id}")
def get_hypothesis(hypothesis_id: int, db: Session = Depends(get_db)):
    record = db.query(HypothesisRecord).filter(HypothesisRecord.id == hypothesis_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Hypothesis not found")
    return {
        "id": record.id,
        "ticker": record.ticker,
        "direction": record.direction,
        "thesis": record.thesis,
        "confidence": record.confidence,
        "horizon_days": record.horizon_days,
        "source_refs": json.loads(record.source_refs_json),
        "reasoning_trail": json.loads(record.reasoning_trail_json),
        "created_at": record.created_at,
    }


# ───────────────────────────────────────── backtesting

@app.post("/backtest/run/{hypothesis_id}")
def run_backtest(hypothesis_id: int, db: Session = Depends(get_db)):
    record = db.query(HypothesisRecord).filter(HypothesisRecord.id == hypothesis_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Hypothesis not found")

    hypothesis = TradeHypothesis(
        id=record.id,
        ticker=record.ticker,
        direction=record.direction,
        thesis=record.thesis,
        confidence=record.confidence,
        horizon_days=record.horizon_days,
        created_at=record.created_at,
    )

    result = backtest_engine.run(hypothesis)
    if result is None:
        return {"message": "No trade to backtest (neutral direction or insufficient data)."}

    saved = save_backtest_result(db, result)
    return {
        "id": saved.id, "ticker": saved.ticker, "pnl_pct": saved.pnl_pct,
        "sharpe_ratio": saved.sharpe_ratio, "win": saved.win,
        "benchmark_pnl_pct": saved.benchmark_pnl_pct,
    }


@app.get("/backtest/results")
def list_backtest_results(db: Session = Depends(get_db)):
    records = db.query(BacktestRecord).order_by(BacktestRecord.created_at.desc()).all()
    return [
        {
            "id": r.id, "ticker": r.ticker, "direction": r.direction,
            "pnl_pct": r.pnl_pct, "sharpe_ratio": r.sharpe_ratio,
            "max_drawdown_pct": r.max_drawdown_pct, "win": r.win,
            "benchmark_pnl_pct": r.benchmark_pnl_pct,
            "entry_date": r.entry_date, "exit_date": r.exit_date,
        }
        for r in records
    ]


# ───────────────────────────────────────── dashboard

@app.get("/dashboard/stats")
def dashboard_stats(db: Session = Depends(get_db)):
    records = db.query(BacktestRecord).all()
    results = [
        BacktestResult(
            hypothesis_id=r.hypothesis_id, ticker=r.ticker, direction=r.direction,
            entry_date=r.entry_date, exit_date=r.exit_date,
            entry_price=r.entry_price, exit_price=r.exit_price,
            pnl_pct=r.pnl_pct, max_drawdown_pct=r.max_drawdown_pct,
            sharpe_ratio=r.sharpe_ratio, win=r.win,
            benchmark_pnl_pct=r.benchmark_pnl_pct,
        )
        for r in records
    ]
    return compute_portfolio_stats(results).dict()


@app.get("/dashboard/vs-benchmark")
def dashboard_vs_benchmark(db: Session = Depends(get_db)):
    records = db.query(BacktestRecord).all()
    results = [
        BacktestResult(
            hypothesis_id=r.hypothesis_id, ticker=r.ticker, direction=r.direction,
            entry_date=r.entry_date, exit_date=r.exit_date,
            entry_price=r.entry_price, exit_price=r.exit_price,
            pnl_pct=r.pnl_pct, max_drawdown_pct=r.max_drawdown_pct,
            sharpe_ratio=r.sharpe_ratio, win=r.win,
            benchmark_pnl_pct=r.benchmark_pnl_pct,
        )
        for r in records
    ]
    return compute_vs_benchmark(results)


@app.get("/health")
def health():
    return {"status": "ok"}
