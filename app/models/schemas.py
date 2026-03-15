from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ValidationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"


# ── Request models ────────────────────────────────────────────────────────────

class CreditRecord(BaseModel):
    applicant_id: str = Field(..., description="Unique applicant identifier")
    name: str = Field(..., min_length=2, max_length=100)
    annual_income: float = Field(..., gt=0, description="Annual income in USD")
    credit_score: int = Field(..., ge=300, le=850)
    existing_debt: float = Field(..., ge=0)
    loan_amount_requested: float = Field(..., gt=0)
    loan_purpose: str = Field(..., description="Purpose of the loan")
    employment_years: float = Field(..., ge=0)
    payment_history: str = Field(..., description="Good / Fair / Poor")


class AnalysisRequest(BaseModel):
    credit_record: CreditRecord
    enable_reflection: bool = Field(default=True, description="Enable self-reflection loops")
    max_loops: Optional[int] = Field(default=None, ge=1, le=5)


# ── Response models ───────────────────────────────────────────────────────────

class AgentStep(BaseModel):
    agent: str
    action: str
    output: str
    confidence: float
    iteration: int


class CreditAnalysisResult(BaseModel):
    applicant_id: str
    risk_level: RiskLevel
    validation_status: ValidationStatus
    recommendation: str
    confidence_score: float
    debt_to_income_ratio: float
    key_findings: List[str]
    risk_factors: List[str]
    agent_steps: List[AgentStep]
    reflection_loops_used: int
    hallucination_checks_passed: int
    final_report: str


class HealthResponse(BaseModel):
    status: str
    langsmith_enabled: bool
    model: str
