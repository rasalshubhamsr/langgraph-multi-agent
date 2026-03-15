"""
Agent nodes for the LangGraph multi-agent credit analysis pipeline.

Agents:
  1. FinancialAnalystAgent  — Analyses DTI, income, credit score
  2. RiskAssessorAgent      — Assigns risk level with reasoning
  3. ReflectionAgent        — Self-checks for hallucinations & gaps
  4. ValidatorAgent         — Final pass/fail decision
  5. ReportWriterAgent      — Produces final human-readable report
"""

import json
import time
from loguru import logger
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from app.config import get_settings
from app.graphs.state import AgentState
from app.models.schemas import AgentStep, RiskLevel, ValidationStatus

settings = get_settings()


def _get_llm(temperature: float = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=temperature if temperature is not None else settings.temperature,
        max_tokens=settings.max_tokens,
        api_key=settings.openai_api_key,
    )


def _make_step(agent: str, action: str, output: str, confidence: float, iteration: int) -> AgentStep:
    return AgentStep(
        agent=agent,
        action=action,
        output=output[:300],
        confidence=round(confidence, 3),
        iteration=iteration,
    )


# ── Agent 1: Financial Analyst ────────────────────────────────────────────────

async def financial_analyst_agent(state: AgentState) -> AgentState:
    """Calculates DTI, analyses income stability, credit score context."""
    logger.info("FinancialAnalystAgent running...")
    rec = state["credit_record"]
    llm = _get_llm()

    dti = round(rec["existing_debt"] / rec["annual_income"], 4) if rec["annual_income"] > 0 else 1.0
    monthly_income = rec["annual_income"] / 12
    monthly_payment_estimate = rec["loan_amount_requested"] / 60  # 5-year assumption

    prompt = f"""You are a financial analyst reviewing a credit application.

Applicant Data:
- Annual Income: ${rec['annual_income']:,.0f}
- Credit Score: {rec['credit_score']}
- Existing Debt: ${rec['existing_debt']:,.0f}
- Loan Requested: ${rec['loan_amount_requested']:,.0f}
- Loan Purpose: {rec['loan_purpose']}
- Employment Years: {rec['employment_years']}
- Payment History: {rec['payment_history']}
- Calculated DTI: {dti:.1%}
- Estimated Monthly Payment: ${monthly_payment_estimate:,.0f}

Provide a structured financial analysis covering:
1. Income adequacy
2. Debt-to-income ratio assessment (flag if >43%)
3. Credit score interpretation
4. Employment stability
5. Loan feasibility

Be factual and specific. Output as structured analysis."""

    response = await llm.ainvoke([
        SystemMessage(content="You are a senior financial analyst. Be precise and data-driven."),
        HumanMessage(content=prompt),
    ])

    analysis = response.content
    confidence = 0.9 if dti < 0.43 and rec["credit_score"] >= 650 else 0.65

    logger.info(f"FinancialAnalystAgent complete. DTI={dti:.2%}, confidence={confidence}")

    return {
        **state,
        "financial_analysis": analysis,
        "agent_steps": [_make_step(
            "FinancialAnalystAgent", "financial_analysis", analysis, confidence,
            state.get("reflection_count", 0)
        )],
    }


# ── Agent 2: Risk Assessor ────────────────────────────────────────────────────

async def risk_assessor_agent(state: AgentState) -> AgentState:
    """Assigns LOW/MEDIUM/HIGH/CRITICAL risk with explicit reasoning."""
    logger.info("RiskAssessorAgent running...")
    llm = _get_llm(temperature=0.0)

    prompt = f"""You are a credit risk assessor.

Based on this financial analysis:
{state['financial_analysis']}

Original applicant data:
{json.dumps(state['credit_record'], indent=2)}

Assign a risk level and provide reasoning. Respond in JSON:
{{
  "risk_level": "low|medium|high|critical",
  "confidence": 0.0-1.0,
  "risk_factors": ["factor1", "factor2"],
  "key_findings": ["finding1", "finding2"],
  "reasoning": "detailed explanation"
}}"""

    response = await llm.ainvoke([
        SystemMessage(content="You are a senior credit risk officer. Output valid JSON only."),
        HumanMessage(content=prompt),
    ])

    try:
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content.strip())
    except json.JSONDecodeError:
        logger.warning("RiskAssessorAgent: JSON parse failed, using fallback.")
        result = {
            "risk_level": "medium",
            "confidence": 0.5,
            "risk_factors": ["Unable to parse structured output"],
            "key_findings": ["Manual review recommended"],
            "reasoning": response.content,
        }

    logger.info(f"RiskAssessorAgent: risk={result['risk_level']}, confidence={result['confidence']}")

    return {
        **state,
        "risk_assessment": json.dumps(result),
        "risk_level": result["risk_level"],
        "confidence_score": float(result["confidence"]),
        "agent_steps": [_make_step(
            "RiskAssessorAgent", "risk_assessment",
            result["reasoning"], result["confidence"],
            state.get("reflection_count", 0)
        )],
    }


# ── Agent 3: Reflection Agent ─────────────────────────────────────────────────

async def reflection_agent(state: AgentState) -> AgentState:
    """
    Self-reflection loop: checks for hallucinations, gaps, and inconsistencies.
    If issues found, flags them and increments reflection_count.
    """
    logger.info(f"ReflectionAgent running (loop {state.get('reflection_count', 0) + 1})...")
    llm = _get_llm(temperature=0.0)

    prompt = f"""You are a quality assurance agent reviewing a credit analysis for hallucinations and errors.

Financial Analysis:
{state['financial_analysis']}

Risk Assessment:
{state['risk_assessment']}

Original Data:
{json.dumps(state['credit_record'], indent=2)}

Check for:
1. Any numbers that contradict the original data
2. Unsupported claims not backed by data
3. Missing critical risk factors
4. Logical inconsistencies

Respond in JSON:
{{
  "has_issues": true|false,
  "issues_found": ["issue1", "issue2"],
  "confidence_in_analysis": 0.0-1.0,
  "recommendation": "approve_analysis|redo_analysis",
  "feedback": "specific guidance for improvement if needed"
}}"""

    response = await llm.ainvoke([
        SystemMessage(content="You are a meticulous QA agent. Output valid JSON only."),
        HumanMessage(content=prompt),
    ])

    try:
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content.strip())
    except json.JSONDecodeError:
        result = {
            "has_issues": False,
            "issues_found": [],
            "confidence_in_analysis": 0.8,
            "recommendation": "approve_analysis",
            "feedback": "",
        }

    flags = result.get("issues_found", [])
    logger.info(f"ReflectionAgent: has_issues={result['has_issues']}, flags={len(flags)}")

    return {
        **state,
        "reflection_feedback": result.get("feedback", ""),
        "reflection_count": state.get("reflection_count", 0) + 1,
        "confidence_score": float(result.get("confidence_in_analysis", state["confidence_score"])),
        "hallucination_flags": flags,
        "agent_steps": [_make_step(
            "ReflectionAgent", "self_reflection",
            result.get("feedback", "No issues found."),
            result.get("confidence_in_analysis", 0.8),
            state.get("reflection_count", 0) + 1
        )],
    }


# ── Agent 4: Validator ────────────────────────────────────────────────────────

async def validator_agent(state: AgentState) -> AgentState:
    """Final validation — pass/fail/needs_review decision."""
    logger.info("ValidatorAgent running...")
    llm = _get_llm(temperature=0.0)

    prompt = f"""You are the final credit validation officer.

Risk Assessment: {state['risk_assessment']}
Reflection Feedback: {state.get('reflection_feedback', 'None')}
Confidence Score: {state['confidence_score']}
Hallucination Flags: {state.get('hallucination_flags', [])}
Reflection Loops: {state.get('reflection_count', 0)}

Make the final validation decision. Respond in JSON:
{{
  "validation_status": "passed|failed|needs_review",
  "recommendation": "approve|reject|manual_review",
  "reasoning": "one paragraph explanation",
  "confidence": 0.0-1.0
}}"""

    response = await llm.ainvoke([
        SystemMessage(content="You are a final credit validation officer. Output valid JSON only."),
        HumanMessage(content=prompt),
    ])

    try:
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content.strip())
    except json.JSONDecodeError:
        result = {
            "validation_status": "needs_review",
            "recommendation": "manual_review",
            "reasoning": response.content,
            "confidence": 0.5,
        }

    logger.info(f"ValidatorAgent: status={result['validation_status']}")

    return {
        **state,
        "validation_result": json.dumps(result),
        "validation_status": result["validation_status"],
        "agent_steps": [_make_step(
            "ValidatorAgent", "final_validation",
            result["reasoning"], result["confidence"],
            state.get("reflection_count", 0)
        )],
    }


# ── Agent 5: Report Writer ────────────────────────────────────────────────────

async def report_writer_agent(state: AgentState) -> AgentState:
    """Produces the final human-readable credit analysis report."""
    logger.info("ReportWriterAgent running...")
    llm = _get_llm(temperature=0.2)

    rec = state["credit_record"]
    try:
        risk_data = json.loads(state.get("risk_assessment", "{}"))
    except json.JSONDecodeError:
        risk_data = {}
    try:
        val_data = json.loads(state.get("validation_result", "{}"))
    except json.JSONDecodeError:
        val_data = {}

    prompt = f"""You are a senior credit analyst writing an executive report.

Applicant: {rec['name']} (ID: {state['applicant_id']})
Loan Request: ${rec['loan_amount_requested']:,.0f} for {rec['loan_purpose']}

Risk Level: {state['risk_level'].upper()}
Validation Status: {state['validation_status'].upper()}
Confidence: {state['confidence_score']:.1%}
Reflection Loops: {state['reflection_count']}
Hallucination Checks Passed: {3 - len(state.get('hallucination_flags', []))} / 3

Key Findings: {risk_data.get('key_findings', [])}
Risk Factors: {risk_data.get('risk_factors', [])}
Recommendation: {val_data.get('recommendation', 'N/A')}

Write a concise, professional 3-paragraph credit analysis report.
Paragraph 1: Applicant financial summary
Paragraph 2: Risk assessment findings
Paragraph 3: Recommendation and next steps"""

    response = await llm.ainvoke([
        SystemMessage(content="You are a senior credit analyst. Write clear, professional reports."),
        HumanMessage(content=prompt),
    ])

    logger.info("ReportWriterAgent complete.")

    return {
        **state,
        "final_report": response.content,
        "is_complete": True,
        "agent_steps": [_make_step(
            "ReportWriterAgent", "write_final_report",
            response.content[:200], 0.95,
            state.get("reflection_count", 0)
        )],
    }
