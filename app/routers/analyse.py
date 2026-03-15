from fastapi import APIRouter, HTTPException
from loguru import logger
from app.models.schemas import AnalysisRequest, CreditAnalysisResult
from app.agents.runner import run_analysis

router = APIRouter(prefix="/analyse", tags=["Credit Analysis"])


@router.post("/", response_model=CreditAnalysisResult, summary="Run multi-agent credit analysis")
async def analyse(request: AnalysisRequest) -> CreditAnalysisResult:
    """
    Submit a credit application for multi-agent analysis.

    Pipeline:
    1. **FinancialAnalystAgent** — DTI, income, credit score analysis
    2. **RiskAssessorAgent** — Risk level assignment with reasoning
    3. **ReflectionAgent** — Self-reflection loop (hallucination checks)
    4. **ValidatorAgent** — Final pass/fail decision
    5. **ReportWriterAgent** — Human-readable executive report

    Returns full analysis including agent steps, risk level, and final report.
    """
    try:
        return await run_analysis(request)
    except Exception as e:
        logger.error(f"Analysis failed for {request.credit_record.applicant_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
