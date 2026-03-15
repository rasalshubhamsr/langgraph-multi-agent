import os
from loguru import logger
from app.config import get_settings
from app.graphs.pipeline import compiled_graph
from app.graphs.state import AgentState
from app.models.schemas import (
    AnalysisRequest, CreditAnalysisResult,
    RiskLevel, ValidationStatus, AgentStep
)
import json

settings = get_settings()


def _setup_langsmith():
    """Enable LangSmith tracing if configured."""
    if settings.langchain_tracing_v2 and settings.langchain_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        logger.info(f"LangSmith tracing enabled → project: {settings.langchain_project}")
    else:
        logger.info("LangSmith tracing disabled (set LANGCHAIN_API_KEY to enable).")


_setup_langsmith()


async def run_analysis(request: AnalysisRequest) -> CreditAnalysisResult:
    """
    Entry point for the multi-agent credit analysis pipeline.
    Invokes the compiled LangGraph and returns a structured result.
    """
    rec = request.credit_record
    max_loops = request.max_loops or settings.max_reflection_loops

    logger.info(f"Starting analysis for applicant: {rec.applicant_id}")

    # Build initial state
    initial_state: AgentState = {
        "applicant_id": rec.applicant_id,
        "credit_record": rec.model_dump(),
        "financial_analysis": "",
        "risk_assessment": "",
        "validation_result": "",
        "reflection_feedback": "",
        "agent_steps": [],
        "reflection_count": 0,
        "confidence_score": 0.0,
        "hallucination_flags": [],
        "risk_level": None,
        "validation_status": None,
        "final_report": None,
        "is_complete": False,
    }

    # Run graph
    final_state = await compiled_graph.ainvoke(initial_state)

    logger.info(
        f"Analysis complete: risk={final_state['risk_level']}, "
        f"status={final_state['validation_status']}, "
        f"loops={final_state['reflection_count']}"
    )

    # Parse sub-results
    try:
        risk_data = json.loads(final_state.get("risk_assessment", "{}"))
    except Exception:
        risk_data = {}

    dti = round(
        rec.existing_debt / rec.annual_income, 4
    ) if rec.annual_income > 0 else 0.0

    return CreditAnalysisResult(
        applicant_id=rec.applicant_id,
        risk_level=RiskLevel(final_state.get("risk_level", "medium")),
        validation_status=ValidationStatus(final_state.get("validation_status", "needs_review")),
        recommendation=risk_data.get("reasoning", "See final report."),
        confidence_score=round(final_state.get("confidence_score", 0.0), 4),
        debt_to_income_ratio=dti,
        key_findings=risk_data.get("key_findings", []),
        risk_factors=risk_data.get("risk_factors", []),
        agent_steps=final_state.get("agent_steps", []),
        reflection_loops_used=final_state.get("reflection_count", 0),
        hallucination_checks_passed=max(0, 3 - len(final_state.get("hallucination_flags", []))),
        final_report=final_state.get("final_report", "Report unavailable."),
    )
