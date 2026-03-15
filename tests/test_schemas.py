"""
Basic unit tests for schemas and agent state.
Run: pytest tests/ -v
"""

import pytest
from app.models.schemas import CreditRecord, AnalysisRequest, RiskLevel, ValidationStatus
from app.graphs.state import AgentState


def make_record(**kwargs):
    defaults = dict(
        applicant_id="TEST-001",
        name="Test User",
        annual_income=80000,
        credit_score=700,
        existing_debt=15000,
        loan_amount_requested=20000,
        loan_purpose="Car purchase",
        employment_years=4.0,
        payment_history="Good",
    )
    defaults.update(kwargs)
    return CreditRecord(**defaults)


def test_credit_record_valid():
    rec = make_record()
    assert rec.applicant_id == "TEST-001"
    assert rec.credit_score == 700


def test_credit_record_invalid_score():
    with pytest.raises(Exception):
        make_record(credit_score=200)  # below 300


def test_credit_record_invalid_income():
    with pytest.raises(Exception):
        make_record(annual_income=-1000)  # must be > 0


def test_analysis_request_defaults():
    rec = make_record()
    req = AnalysisRequest(credit_record=rec)
    assert req.enable_reflection is True
    assert req.max_loops is None


def test_risk_levels():
    for level in ["low", "medium", "high", "critical"]:
        assert RiskLevel(level) is not None


def test_validation_statuses():
    for status in ["passed", "failed", "needs_review"]:
        assert ValidationStatus(status) is not None


def test_dti_calculation():
    rec = make_record(annual_income=100000, existing_debt=43000)
    dti = rec.existing_debt / rec.annual_income
    assert abs(dti - 0.43) < 0.001  # exactly at threshold
