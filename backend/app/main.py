from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from .agent.agent_executor import AgentExecutor

app = FastAPI(title="LexiBot Agent API", version="0.2.0")
executor = AgentExecutor()

MODEL_VERSION = os.getenv("MODEL_VERSION", "lexibot-local")
API_TOKEN = os.getenv("LEGAL_API_TOKEN")
bearer_scheme = HTTPBearer(auto_error=False)


def verify_token(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)) -> None:
    if not API_TOKEN:
        return
    if credentials is None or credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing API token")


class AgentRequest(BaseModel):
    query: str = Field(..., description="User's natural language request")
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional structured context such as uploaded contract sections",
    )


class TraceEntry(BaseModel):
    step: Dict[str, Any]
    result: Dict[str, Any]
    tool_reasoning: str
    tool_metadata: Dict[str, Any]


class AgentResponse(BaseModel):
    answer: str
    plan: List[Dict[str, Any]]
    trace: List[TraceEntry]


AnalysisType = Literal["comprehensive", "quick", "specific"]


class ContractAnalysisRequest(BaseModel):
    text: str = Field(..., description="Contract text to analyse", min_length=1)
    analysis_type: AnalysisType = Field(
        default="comprehensive", description="Analysis depth: comprehensive, quick, or specific"
    )
    questions: Optional[List[str]] = Field(
        default=None,
        description="Custom questions to answer when analysis_type is 'specific'",
    )


class ContractComparisonRequest(BaseModel):
    contract_a: str = Field(..., description="First contract text", min_length=1)
    contract_b: str = Field(..., description="Second contract text", min_length=1)
    focus_areas: Optional[List[str]] = Field(
        default=None, description="Optional focus areas to highlight in the comparison"
    )


class ClauseExtractionRequest(BaseModel):
    text: str = Field(..., description="Contract text to search", min_length=1)
    clause_types: Optional[List[str]] = Field(
        default=None, description="Clause categories to prioritise in the extraction"
    )
    query: Optional[str] = Field(default=None, description="Free-text clause query")
    top_k: int = Field(default=5, ge=1, le=20, description="Maximum clauses to return")


class AnalysisResponse(BaseModel):
    success: bool
    analysis: Dict[str, Any]
    confidence_score: float
    processing_time: float
    model_version: str


class ClauseExtractionResponse(BaseModel):
    success: bool
    clauses: List[Dict[str, Any]]
    confidence_score: float
    processing_time: float


DEFAULT_COMPREHENSIVE_QUESTIONS = [
    "Who are the main parties in this contract?",
    "What are the key terms and obligations?",
    "Identify any termination clauses and notice periods.",
    "What are the principal risks or liabilities?",
]
QUICK_OVERVIEW_QUESTION = "Provide a concise executive summary of this contract, including obligations and key risks."


def _build_document_context(text: str) -> Dict[str, Any]:
    return {
        "documents": [
            {
                "text": text,
                "metadata": {"source": "uploaded_contract"},
            }
        ]
    }


def _resolve_questions(payload: ContractAnalysisRequest) -> List[str]:
    if payload.analysis_type == "quick":
        return [QUICK_OVERVIEW_QUESTION]
    if payload.analysis_type == "specific":
        if not payload.questions:
            raise HTTPException(
                status_code=400,
                detail="Custom questions must be provided when analysis_type is 'specific'",
            )
        cleaned = [question.strip() for question in payload.questions if question.strip()]
        if not cleaned:
            raise HTTPException(
                status_code=400,
                detail="At least one non-empty custom question is required",
            )
        return cleaned
    return DEFAULT_COMPREHENSIVE_QUESTIONS


def _format_comparison_analysis(diff_output: str, focus_areas: Optional[List[str]]) -> Dict[str, Any]:
    diff_lines = diff_output.splitlines()
    additions = [line for line in diff_lines if line.startswith("+ ")]
    removals = [line for line in diff_lines if line.startswith("- ")]
    key_differences = []
    for line in diff_lines:
        if line.startswith("+ ") or line.startswith("- "):
            key_differences.append(line)
        if len(key_differences) >= 10:
            break

    total_changes = len(additions) + len(removals)
    if total_changes > 80:
        risk_level = "HIGH"
    elif total_changes > 30:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    summary = "Generated a unified diff between Contract A and Contract B."
    if focus_areas:
        summary += " Focus areas supplied by the caller: " + ", ".join(focus_areas) + "."

    recommendations: List[str] = []
    if additions:
        recommendations.append("Review added lines (prefixed with '+') for new obligations or liabilities.")
    if removals:
        recommendations.append("Confirm removed lines (prefixed with '-') do not delete critical protections.")
    if focus_areas:
        recommendations.append("Pay special attention to the highlighted focus areas in the diff output.")
    if not recommendations:
        recommendations.append("No material textual differences detected; contracts appear aligned.")

    return {
        "summary": summary,
        "key_differences": key_differences or ["No material textual differences detected."],
        "risk_assessment": f"{risk_level} risk: {total_changes} changed lines identified across both versions.",
        "recommendations": recommendations,
        "diff": diff_output,
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/agent/run", response_model=AgentResponse)
def run_agent(request: AgentRequest) -> AgentResponse:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    result = executor.run(request.query, request.context)
    return AgentResponse(**result)


@app.post("/api/v1/analyze-contract", response_model=AnalysisResponse)
def analyze_contract(request: ContractAnalysisRequest, _: None = Depends(verify_token)) -> AnalysisResponse:
    contract_text = request.text.strip()
    if not contract_text:
        raise HTTPException(status_code=400, detail="Contract text must not be empty")

    questions = _resolve_questions(request)
    context = _build_document_context(contract_text)

    start = time.perf_counter()
    answers: Dict[str, Any] = {}
    confidences: List[float] = []
    overall_success = True

    for question in questions:
        result = executor.execute_tool("contract_qa_tool", question=question, context=context)
        answers[question] = result.content
        confidences.append(max(result.confidence, 0.0))
        if not result.success:
            overall_success = False

    elapsed = time.perf_counter() - start
    average_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return AnalysisResponse(
        success=overall_success,
        analysis=answers,
        confidence_score=round(average_confidence, 4),
        processing_time=round(elapsed, 4),
        model_version=MODEL_VERSION,
    )


@app.post("/api/v1/compare-contracts", response_model=AnalysisResponse)
def compare_contracts(
    request: ContractComparisonRequest,
    _: None = Depends(verify_token),
) -> AnalysisResponse:
    start = time.perf_counter()

    result = executor.execute_tool(
        "contract_comparison_tool",
        contracts=[
            {"text": request.contract_a},
            {"text": request.contract_b},
        ],
    )

    if not result.success:
        raise HTTPException(status_code=500, detail=str(result.content))

    diff_output = str(result.content.get("differences", ""))
    analysis = _format_comparison_analysis(diff_output, request.focus_areas)
    elapsed = time.perf_counter() - start

    return AnalysisResponse(
        success=True,
        analysis=analysis,
        confidence_score=round(result.confidence, 4),
        processing_time=round(elapsed, 4),
        model_version=MODEL_VERSION,
    )


@app.post("/api/v1/extract-clauses", response_model=ClauseExtractionResponse)
def extract_clauses(
    request: ClauseExtractionRequest,
    _: None = Depends(verify_token),
) -> ClauseExtractionResponse:
    contract_text = request.text.strip()
    if not contract_text:
        raise HTTPException(status_code=400, detail="Contract text must not be empty")

    context = _build_document_context(contract_text)
    start = time.perf_counter()

    result = executor.execute_tool(
        "clause_retrieval_tool",
        query=request.query,
        clause_types=request.clause_types,
        context=context,
        top_k=request.top_k,
    )

    if not result.success:
        raise HTTPException(status_code=404, detail=str(result.content))

    elapsed = time.perf_counter() - start
    clauses = list(result.content) if isinstance(result.content, list) else []

    return ClauseExtractionResponse(
        success=True,
        clauses=clauses,
        confidence_score=round(result.confidence, 4),
        processing_time=round(elapsed, 4),
    )
