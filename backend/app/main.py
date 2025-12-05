from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

# ==========================================
# 1. Path Setup & Imports
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import your Agent components
from .agent.agent_executor import AgentExecutor
# FIX: Use relative imports for consistency
from .tools.structured_data_extractor import StructuredDataExtractor
from .tools.contract_qa_tool import ContractQATool 
from .llm.ollama_client import OllamaClient
# from .llm.gemini_client import GeminiClient 

# ==========================================
# 2. App Configuration
# ==========================================
app = FastAPI(title="LexiBot Agent API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_VERSION = os.getenv("MODEL_VERSION", "lexibot-local")
API_TOKEN = os.getenv("LEGAL_API_TOKEN")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

bearer_scheme = HTTPBearer(auto_error=False)

# ==========================================
# 3. Agent Initialization (Singleton Pattern)
# ==========================================
def initialize_executor() -> AgentExecutor:
    """
    Initializes the AgentExecutor with the LLM client and registers dynamic tools.
    """
    # A. Initialize LLM Client
    llm_client = None
    try:
        llm_client = OllamaClient(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
        print(f"INFO: Initialized OllamaClient with model {OLLAMA_MODEL}")
    except Exception as e:
        print(f"WARNING: Failed to init LLM Client: {e}")

    # B. Initialize Executor
    executor_instance = AgentExecutor(llm_client=llm_client)

    # C. Register Dynamic Tools
    if llm_client:
        # 1. Register Data Extractor
        extractor_tool = StructuredDataExtractor(llm_client=llm_client)
        executor_instance.register_tool(extractor_tool)
        
        # 2. Register QA Tool (CRITICAL MISSING PIECE ADDED HERE)
        qa_tool = ContractQATool(llm_client=llm_client)
        executor_instance.register_tool(qa_tool)
        
        print("INFO: Registered Dynamic Tools (Extractor & QA)")
    
    return executor_instance

# Create a global instance
global_executor = initialize_executor()

def get_executor() -> AgentExecutor:
    return global_executor

def verify_token(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)) -> None:
    if not API_TOKEN:
        return
    if credentials is None or credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing API token")

# ==========================================
# 4. Pydantic Models
# ==========================================
class AgentRequest(BaseModel):
    query: str = Field(..., description="User's natural language request")
    chat_history: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Previous conversation history [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]"
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional structured context such as uploaded contract text",
    )

class TraceEntry(BaseModel):
    step: Dict[str, Any]
    result: Dict[str, Any]
    tool_metadata: Dict[str, Any]

class AgentResponse(BaseModel):
    answer: str
    plan: List[Dict[str, Any]]
    trace: List[TraceEntry]

class AnalysisResponse(BaseModel):
    success: bool
    analysis: Dict[str, Any]
    confidence_score: float
    processing_time: float
    model_version: str

# ==========================================
# 5. API Endpoints
# ==========================================

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "model": MODEL_VERSION}

@app.post("/agent/run", response_model=AgentResponse)
def run_agent(
    request: AgentRequest, 
    executor: AgentExecutor = Depends(get_executor)
) -> AgentResponse:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    # Debugging logs (Optional: Keep or remove)
    print("\n--- 🔍 DEBUGGING INCOMING REQUEST ---")
    print(f"QUERY: {request.query}")
    print(f"CONTEXT KEYS: {request.context.keys() if request.context else 'None'}")
    # print(f"FULL CONTEXT: {request.context}") # Commented out to reduce noise
    print("-------------------------------------\n")

    result = executor.run(request.query, request.context, request.chat_history)
    return AgentResponse(**result)

# Example: Updating extract-clauses to be safer
class ClauseExtractionRequest(BaseModel):
    text: str
    query: Optional[str] = None
    clause_types: Optional[List[str]] = None
    top_k: int = 5

class ClauseExtractionResponse(BaseModel):
    success: bool
    clauses: Any 
    confidence_score: float
    processing_time: float

@app.post("/api/v1/extract-clauses", response_model=ClauseExtractionResponse)
def extract_clauses(
    request: ClauseExtractionRequest,
    executor: AgentExecutor = Depends(get_executor),
    _: None = Depends(verify_token),
) -> ClauseExtractionResponse:
    start = time.perf_counter()
    
    result = executor.execute_tool(
        "clause_retrieval_tool",
        query=request.query,
        clause_types=request.clause_types,
        contract_text=request.text, 
        top_k=request.top_k,
    )

    elapsed = time.perf_counter() - start
    
    return ClauseExtractionResponse(
        success=result.success,
        clauses=result.content,
        confidence_score=getattr(result, "confidence", 0.8),
        processing_time=round(elapsed, 4),
    )