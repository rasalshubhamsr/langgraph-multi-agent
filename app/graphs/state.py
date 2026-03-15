from typing import TypedDict, List, Optional, Annotated
import operator
from app.models.schemas import AgentStep, RiskLevel, ValidationStatus


class AgentState(TypedDict):
    """
    Shared state that flows through the LangGraph agent graph.
    Each agent reads from and writes to this state.
    """
    # Input
    applicant_id: str
    credit_record: dict

    # Intermediate outputs
    financial_analysis: str
    risk_assessment: str
    validation_result: str
    reflection_feedback: str

    # Tracking
    agent_steps: Annotated[List[AgentStep], operator.add]
    reflection_count: int
    confidence_score: float
    hallucination_flags: Annotated[List[str], operator.add]

    # Final
    risk_level: Optional[str]
    validation_status: Optional[str]
    final_report: Optional[str]
    is_complete: bool
