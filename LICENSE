"""
QuantPilot - agent/tools.py
LangChain-compatible tools the agent can call:
  - search_financial_news   : web search scoped to financial news
  - read_sec_filing         : pulls recent SEC EDGAR filings for a ticker
  - fetch_price_history     : OHLCV data via yfinance
  - fetch_fundamentals      : basic fundamentals snapshot

Each tool returns a short, agent-readable string — not raw JSON dumps —
so the LLM's context window isn't wasted on noise.
"""

import json
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from langchain.tools import tool


# ───────────────────────────────────────────────────────── news search

@tool
def search_financial_news(query: str) -> str:
    """
    Search recent financial news for a company or ticker.
    Input: a search query, e.g. "NVDA earnings outlook" or "Tesla recall 2025".
    Returns a short digest of the top headlines.
    """
    # Uses the free DuckDuckGo HTML endpoint — no API key required.
    # For production, swap in NewsAPI, Tiingo News, or Benzinga.
    try:
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": f"{query} news"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        results = soup.select(".result__title")[:5]

        if not results:
            return f"No recent news found for '{query}'."

        headlines = [r.get_text(strip=True) for r in results]
        digest = "\n".join(f"- {h}" for h in headlines)
        return f"Top headlines for '{query}':\n{digest}"
    except Exception as e:
        return f"News search failed: {e}. Proceed using price/fundamentals data only."


# ───────────────────────────────────────────────────────── SEC EDGAR

@tool
def read_sec_filing(ticker: str) -> str:
    """
    Fetch the most recent SEC filings (10-K, 10-Q, 8-K) for a ticker
    using the free SEC EDGAR full-text search API.
    Input: a stock ticker symbol, e.g. "AAPL".
    Returns a short summary of the latest filing types and dates.
    """
    try:
        # Step 1: resolve ticker -> CIK number
        headers = {"User-Agent": "QuantPilot research-agent contact@example.com"}
        tickers_resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=headers,
            timeout=10,
        )
        tickers_map = tickers_resp.json()

        cik = None
        for entry in tickers_map.values():
            if entry["ticker"].upper() == ticker.upper():
                cik = str(entry["cik_str"]).zfill(10)
                break

        if cik is None:
            return f"Could not resolve CIK for ticker '{ticker}'."

        # Step 2: fetch recent filings for that CIK
        filings_resp = requests.get(
            f"https://data.sec.gov/submissions/CIK{cik}.json",
            headers=headers,
            timeout=10,
        )
        filings = filings_resp.json()
        recent = filings.get("filings", {}).get("recent", {})

        forms = recent.get("form", [])[:5]
        dates = recent.get("filingDate", [])[:5]

        if not forms:
            return f"No recent filings found for {ticker}."

        summary_lines = [f"- {f} filed on {d}" for f, d in zip(forms, dates)]
        return f"Recent SEC filings for {ticker}:\n" + "\n".join(summary_lines)

    except Exception as e:
        return f"SEC filing lookup failed: {e}. Proceed using other available data."


# ───────────────────────────────────────────────────────── price data

@tool
def fetch_price_history(ticker: str, days: int = 90) -> str:
    """
    Fetch recent OHLCV price history for a ticker.
    Input: ticker symbol and optional lookback window in days (default 90).
    Returns a compact summary: current price, % change, volatility, volume trend.
    """
    try:
        end = datetime.now()
        start = end - timedelta(days=days)
        df = yf.download(ticker, start=start, end=end, progress=False)

        if df.empty:
            return f"No price data found for '{ticker}'."

        current_price = float(df["Close"].iloc[-1])
        start_price = float(df["Close"].iloc[0])
        pct_change = (current_price - start_price) / start_price * 100
        volatility = float(df["Close"].pct_change().std() * 100)
        avg_volume = float(df["Volume"].mean())
        recent_volume = float(df["Volume"].iloc[-5:].mean())
        volume_trend = "above average" if recent_volume > avg_volume else "below average"

        return (
            f"{ticker} price summary ({days}d window):\n"
            f"- Current price: ${current_price:.2f}\n"
            f"- {days}-day change: {pct_change:+.2f}%\n"
            f"- Daily volatility (std dev): {volatility:.2f}%\n"
            f"- Recent volume: {volume_trend} (5d avg vs {days}d avg)"
        )
    except Exception as e:
        return f"Price fetch failed for '{ticker}': {e}"


# ───────────────────────────────────────────────────────── fundamentals

@tool
def fetch_fundamentals(ticker: str) -> str:
    """
    Fetch a basic fundamentals snapshot for a ticker: P/E, market cap,
    sector, and analyst recommendation if available.
    Input: a stock ticker symbol.
    """
    try:
        info = yf.Ticker(ticker).info

        pe = info.get("trailingPE", "N/A")
        market_cap = info.get("marketCap", "N/A")
        sector = info.get("sector", "N/A")
        target = info.get("targetMeanPrice", "N/A")
        recommendation = info.get("recommendationKey", "N/A")

        if isinstance(market_cap, (int, float)):
            market_cap = f"${market_cap / 1e9:.1f}B"

        return (
            f"{ticker} fundamentals:\n"
            f"- Sector: {sector}\n"
            f"- P/E ratio: {pe}\n"
            f"- Market cap: {market_cap}\n"
            f"- Analyst mean target: {target}\n"
            f"- Analyst recommendation: {recommendation}"
        )
    except Exception as e:
        return f"Fundamentals fetch failed for '{ticker}': {e}"


ALL_TOOLS = [
    search_financial_news,
    read_sec_filing,
    fetch_price_history,
    fetch_fundamentals,
]
