"""
FastAPI Production Inference Server
High-performance API for legal document analysis
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import uvicorn
import asyncio
import time
from pathlib import Path
import logging

# Import your existing components
from agent.agent_executor import AgentExecutor
from agent.core.planning_engine import LegalPlanningEngine
from core.model_loader import ModelLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LexiBot Legal AI API",
    description="Production API for AI-powered legal document analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global variables for model and agent
agent_executor = None
model_loader = None

class ContractAnalysisRequest(BaseModel):
    text: str = Field(..., description="Contract text to analyze", max_length=50000)
    questions: Optional[List[str]] = Field(None, description="Specific questions about the contract")
    analysis_type: str = Field("comprehensive", description="Type of analysis: comprehensive, quick, specific")

class ContractComparisonRequest(BaseModel):
    contract_a: str = Field(..., description="First contract text")
    contract_b: str = Field(..., description="Second contract text")  
    focus_areas: Optional[List[str]] = Field(None, description="Specific areas to compare")

class ClauseExtractionRequest(BaseModel):
    text: str = Field(..., description="Contract text")
    clause_types: List[str] = Field(..., description="Types of clauses to extract")

class AgentQueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query about legal documents")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    use_planning: bool = Field(True, description="Whether to use multi-step planning")

class AnalysisResponse(BaseModel):
    success: bool
    analysis: Dict[str, Any]
    confidence_score: float
    processing_time: float
    model_version: str

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None
    timestamp: float

# Authentication dependency
def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify API token - implement your authentication logic"""
    token = credentials.credentials
    # TODO: Implement actual token verification
    if token != "your-secret-token":  # Replace with real auth
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return token

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize models and agents on startup"""
    global agent_executor, model_loader
    
    try:
        logger.info("Initializing model loader...")
        model_loader = ModelLoader()
        
        logger.info("Loading fine-tuned model...")
        # Load your fine-tuned model
        model_path = "/path/to/your/fine-tuned-model"  # Update this path
        model, tokenizer = model_loader.load_model_and_tokenizer(model_path)
        
        logger.info("Initializing agent executor...")
        agent_executor = AgentExecutor()
        agent_executor.set_model(model, tokenizer)
        
        logger.info("API server ready!")
        
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        raise e

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "model_loaded": agent_executor is not None
    }

# Contract Analysis Endpoint
@app.post("/api/v1/analyze-contract", response_model=AnalysisResponse)
async def analyze_contract(
    request: ContractAnalysisRequest,
    token: str = Depends(verify_token)
):
    """
    Comprehensive contract analysis using fine-tuned model
    """
    start_time = time.time()
    
    try:
        if not agent_executor:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Use your existing contract QA tool
        if request.analysis_type == "comprehensive":
            questions = request.questions or [
                "Who are the main parties in this contract?",
                "What are the key terms and conditions?", 
                "What are the main obligations of each party?",
                "Are there any termination clauses?",
                "What are the potential risks or liabilities?"
            ]
        elif request.analysis_type == "quick":
            questions = ["Provide a brief summary of this contract."]
        else:
            questions = request.questions or ["Analyze this contract."]
        
        analysis_results = {}
        total_confidence = 0
        
        for question in questions:
            result = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: agent_executor.execute_tool("contract_qa_tool", {
                    "question": question,
                    "context": request.text
                })
            )
            
            analysis_results[question] = result.get("answer", "")
            total_confidence += result.get("confidence", 0.8)
        
        avg_confidence = total_confidence / len(questions)
        processing_time = time.time() - start_time
        
        return AnalysisResponse(
            success=True,
            analysis=analysis_results,
            confidence_score=avg_confidence,
            processing_time=processing_time,
            model_version="lexibot-v1.0"
        )
        
    except Exception as e:
        logger.error(f"Contract analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Contract Comparison Endpoint  
@app.post("/api/v1/compare-contracts", response_model=AnalysisResponse)
async def compare_contracts(
    request: ContractComparisonRequest,
    token: str = Depends(verify_token)
):
    """
    Compare two contracts and highlight differences
    """
    start_time = time.time()
    
    try:
        if not agent_executor:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: agent_executor.execute_tool("contract_comparison_tool", {
                "contract_a": request.contract_a,
                "contract_b": request.contract_b,
                "focus_areas": request.focus_areas
            })
        )
        
        processing_time = time.time() - start_time
        
        return AnalysisResponse(
            success=True,
            analysis=result,
            confidence_score=0.9,  # Comparison tool typically high confidence
            processing_time=processing_time,
            model_version="lexibot-v1.0"
        )
        
    except Exception as e:
        logger.error(f"Contract comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Clause Extraction Endpoint
@app.post("/api/v1/extract-clauses", response_model=AnalysisResponse)
async def extract_clauses(
    request: ClauseExtractionRequest,
    token: str = Depends(verify_token)
):
    """
    Extract specific types of clauses from contract text
    """
    start_time = time.time()
    
    try:
        if not agent_executor:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: agent_executor.execute_tool("clause_retrieval_tool", {
                "text": request.text,
                "clause_types": request.clause_types
            })
        )
        
        processing_time = time.time() - start_time
        
        return AnalysisResponse(
            success=True,
            analysis=result,
            confidence_score=0.85,
            processing_time=processing_time,
            model_version="lexibot-v1.0"
        )
        
    except Exception as e:
        logger.error(f"Clause extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Advanced Agent Query Endpoint
@app.post("/api/v1/agent-query", response_model=AnalysisResponse)
async def agent_query(
    request: AgentQueryRequest,
    token: str = Depends(verify_token)
):
    """
    Process complex legal queries using the planning agent
    """
    start_time = time.time()
    
    try:
        if not agent_executor:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        if request.use_planning:
            # Use the planning engine for complex queries
            planner = LegalPlanningEngine()
            plan = planner.create_action_plan(request.query, request.context)
            
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: agent_executor.execute_plan(plan)
            )
        else:
            # Direct execution without planning
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: agent_executor.execute_query(request.query, request.context)
            )
        
        processing_time = time.time() - start_time
        
        return AnalysisResponse(
            success=True,
            analysis=result,
            confidence_score=result.get("confidence", 0.8),
            processing_time=processing_time,
            model_version="lexibot-v1.0"
        )
        
    except Exception as e:
        logger.error(f"Agent query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# File Upload Endpoint
@app.post("/api/v1/upload-contract")
async def upload_contract(
    file: UploadFile = File(...),
    token: str = Depends(verify_token)
):
    """
    Upload and analyze contract files (PDF, DOCX, TXT)
    """
    try:
        # Validate file type
        allowed_types = ["text/plain", "application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Read file content
        content = await file.read()
        
        # TODO: Add document parsing logic for PDF/DOCX
        if file.content_type == "text/plain":
            text = content.decode("utf-8")
        else:
            # For now, return error for non-text files
            raise HTTPException(status_code=400, detail="PDF/DOCX parsing not implemented yet")
        
        # Analyze the uploaded contract
        analysis_request = ContractAnalysisRequest(text=text, analysis_type="comprehensive")
        result = await analyze_contract(analysis_request, token)
        
        return {
            "filename": file.filename,
            "file_size": len(content),
            "content_type": file.content_type,
            "analysis": result
        }
        
    except Exception as e:
        logger.error(f"File upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            timestamp=time.time()
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            details=str(exc),
            timestamp=time.time()
        ).dict()
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        workers=1,     # Adjust based on your needs
        access_log=True
    )