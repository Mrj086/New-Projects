"""
QuantPilot - agent/hypothesis_agent.py
The core research agent. Given a ticker (or a sector to scan), it:
  1. Gathers news, filings, price action, fundamentals via tools
  2. Reasons step by step about whether a trade thesis exists
  3. Emits a structured TradeHypothesis with a full, auditable reasoning trail

Built on LangChain's agent executor (works with any LangChain-compatible
LLM — OpenAI by default, swap in Anthropic/local models easily).
"""

import os
import json
import re
from datetime import datetime
from typing import List, Optional

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage, HumanMessage

from agent.tools import ALL_TOOLS
from agent.schemas import TradeHypothesis, ReasoningStep, SignalDirection


SYSTEM_PROMPT = """You are a disciplined equity research analyst working for an internal \
research desk. Your job is NOT to give financial advice to retail users — you are \
generating and documenting testable trade hypotheses for backtesting research purposes only.

For the given ticker, use your tools to gather:
1. Recent price action and volatility
2. Recent news / catalysts
3. Recent SEC filings (if any material ones)
4. Basic fundamentals context

Then reason step-by-step and produce ONE trade hypothesis with:
- A clear direction (long / short / neutral — neutral means "no edge found, skip")
- A one-paragraph thesis explaining the reasoning
- A confidence score from 0.0 to 1.0 (be honest — most situations warrant 0.3-0.6,
  reserve >0.7 for genuinely strong multi-signal alignment)
- A suggested holding horizon in trading days

Be skeptical. If the data doesn't support a clear thesis, say direction=neutral
and confidence<0.3 rather than forcing a trade idea. Avoid overconfidence — you have
been wrong before when news sentiment was strong but price action contradicted it.

After your research, respond ONLY with a JSON object in this exact format:
{{
  "direction": "long" | "short" | "neutral",
  "thesis": "...",
  "confidence": 0.0-1.0,
  "horizon_days": integer,
  "source_refs": ["headline or filing references you used"]
}}
"""


class HypothesisAgent:
    """Wraps a LangChain tool-calling agent specialised for trade research."""

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.2):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set. "
                "QuantPilot's agent reasoning requires an OpenAI API key. "
                "See README.md > Setup for instructions."
            )

        self.llm = ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_openai_tools_agent(self.llm, ALL_TOOLS, prompt)
        self.executor = AgentExecutor(
            agent=agent,
            tools=ALL_TOOLS,
            verbose=True,
            return_intermediate_steps=True,
            max_iterations=8,
        )

    def generate_hypothesis(self, ticker: str) -> TradeHypothesis:
        """
        Run the agent end-to-end on a single ticker and return a
        structured TradeHypothesis with the full reasoning trail attached.
        """
        result = self.executor.invoke({
            "input": f"Research {ticker} and produce a trade hypothesis."
        })

        # ── parse the final JSON answer ──────────────────────────────
        raw_output = result["output"]
        parsed = self._extract_json(raw_output)

        # ── reconstruct the reasoning trail from intermediate steps ──
        reasoning_trail: List[ReasoningStep] = []
        for i, (action, observation) in enumerate(result.get("intermediate_steps", [])):
            reasoning_trail.append(ReasoningStep(
                step_number=i + 1,
                tool_used=action.tool,
                tool_input=str(action.tool_input),
                tool_output_summary=str(observation)[:500],
                thought=action.log.strip()[:500] if action.log else "",
            ))

        hypothesis = TradeHypothesis(
            ticker=ticker.upper(),
            direction=SignalDirection(parsed.get("direction", "neutral")),
            thesis=parsed.get("thesis", "No thesis extracted."),
            confidence=float(parsed.get("confidence", 0.0)),
            horizon_days=int(parsed.get("horizon_days", 5)),
            reasoning_trail=reasoning_trail,
            source_refs=parsed.get("source_refs", []),
        )
        return hypothesis

    def scan_universe(self, tickers: List[str]) -> List[TradeHypothesis]:
        """Run generate_hypothesis across a list of tickers, skipping failures."""
        results = []
        for ticker in tickers:
            try:
                hyp = self.generate_hypothesis(ticker)
                results.append(hyp)
            except Exception as e:
                print(f"[HypothesisAgent] Skipped {ticker} due to error: {e}")
        return results

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Pull the first JSON object out of the agent's final answer text."""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {"direction": "neutral", "thesis": text, "confidence": 0.0}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"direction": "neutral", "thesis": text, "confidence": 0.0}
