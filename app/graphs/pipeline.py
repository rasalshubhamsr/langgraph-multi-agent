"""
LangGraph graph definition for the multi-agent credit analysis pipeline.

Flow:
  financial_analyst
       ↓
  risk_assessor
       ↓
  reflection_agent ──── (has issues AND loops < max) ──→ financial_analyst (redo)
       ↓ (no issues OR max loops reached)
  validator
       ↓
  report_writer
       ↓
     END
"""

from langgraph.graph import StateGraph, END
from app.graphs.state import AgentState
from app.agents.nodes import (
    financial_analyst_agent,
    risk_assessor_agent,
    reflection_agent,
    validator_agent,
    report_writer_agent,
)
from app.config import get_settings
import json

settings = get_settings()


def should_reflect(state: AgentState) -> str:
    """
    Conditional edge after reflection_agent.
    Returns 'redo' if issues found and loops remaining, else 'validate'.
    """
    reflection_count = state.get("reflection_count", 0)
    max_loops = settings.max_reflection_loops
    flags = state.get("hallucination_flags", [])

    try:
        feedback_raw = state.get("reflection_feedback", "{}")
        # Check if reflection found issues via hallucination_flags
        has_issues = len(flags) > 0
    except Exception:
        has_issues = False

    if has_issues and reflection_count < max_loops:
        return "redo"
    return "validate"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("financial_analyst", financial_analyst_agent)
    graph.add_node("risk_assessor", risk_assessor_agent)
    graph.add_node("reflection", reflection_agent)
    graph.add_node("validator", validator_agent)
    graph.add_node("report_writer", report_writer_agent)

    # Entry point
    graph.set_entry_point("financial_analyst")

    # Linear edges
    graph.add_edge("financial_analyst", "risk_assessor")
    graph.add_edge("risk_assessor", "reflection")

    # Conditional edge: reflection → redo OR validate
    graph.add_conditional_edges(
        "reflection",
        should_reflect,
        {
            "redo": "financial_analyst",   # loop back
            "validate": "validator",        # proceed
        }
    )

    graph.add_edge("validator", "report_writer")
    graph.add_edge("report_writer", END)

    return graph.compile()


# Singleton compiled graph
compiled_graph = build_graph()
