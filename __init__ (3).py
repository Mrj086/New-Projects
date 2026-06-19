"""
QuantPilot - api/db.py
SQLAlchemy models + session management.

Stores every hypothesis the agent generates AND its full reasoning trail
(as JSON) — this is what makes the system auditable. Also stores backtest
results linked back to the hypothesis that produced them.

Works with PostgreSQL (production / docker-compose) or SQLite (quick local
testing — just change DATABASE_URL).
"""

import os
import json
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./quantpilot.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class HypothesisRecord(Base):
    __tablename__ = "hypotheses"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True)
    direction = Column(String)
    thesis = Column(Text)
    confidence = Column(Float)
    horizon_days = Column(Integer)
    source_refs_json = Column(Text)        # JSON-encoded list[str]
    reasoning_trail_json = Column(Text)    # JSON-encoded list[ReasoningStep]
    created_at = Column(DateTime, default=datetime.utcnow)


class BacktestRecord(Base):
    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True, index=True)
    hypothesis_id = Column(Integer, index=True)
    ticker = Column(String, index=True)
    direction = Column(String)
    entry_date = Column(DateTime)
    exit_date = Column(DateTime)
    entry_price = Column(Float)
    exit_price = Column(Float)
    pnl_pct = Column(Float)
    max_drawdown_pct = Column(Float)
    sharpe_ratio = Column(Float, nullable=True)
    win = Column(Boolean)
    benchmark_pnl_pct = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ───────────────────────────────────────── persistence helpers

def save_hypothesis(db, hypothesis) -> HypothesisRecord:
    record = HypothesisRecord(
        ticker=hypothesis.ticker,
        direction=hypothesis.direction if isinstance(hypothesis.direction, str) else hypothesis.direction.value,
        thesis=hypothesis.thesis,
        confidence=hypothesis.confidence,
        horizon_days=hypothesis.horizon_days,
        source_refs_json=json.dumps(hypothesis.source_refs),
        reasoning_trail_json=json.dumps([step.dict() for step in hypothesis.reasoning_trail], default=str),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def save_backtest_result(db, result) -> BacktestRecord:
    record = BacktestRecord(
        hypothesis_id=result.hypothesis_id,
        ticker=result.ticker,
        direction=result.direction if isinstance(result.direction, str) else result.direction.value,
        entry_date=result.entry_date,
        exit_date=result.exit_date,
        entry_price=result.entry_price,
        exit_price=result.exit_price,
        pnl_pct=result.pnl_pct,
        max_drawdown_pct=result.max_drawdown_pct,
        sharpe_ratio=result.sharpe_ratio,
        win=result.win,
        benchmark_pnl_pct=result.benchmark_pnl_pct,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record
