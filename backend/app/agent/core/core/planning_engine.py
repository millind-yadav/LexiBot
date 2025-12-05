import json
import logging
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field, ValidationError, validator

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 1. Data Models (Pydantic)
# ==========================================

class TaskType(str, Enum):
    CONTRACT_ANALYSIS = "contract_analysis"
    CLAUSE_EXTRACTION = "clause_extraction"
    CONTRACT_COMPARISON = "contract_comparison"
    LEGAL_RESEARCH = "legal_research"
    RISK_ASSESSMENT = "risk_assessment"
    DOCUMENT_GENERATION = "document_generation"

class AgentStep(BaseModel):
    step_id: int
    task_type: TaskType
    description: str
    tool_name: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    expected_output: str
    dependencies: List[int] = Field(default_factory=list)

    @validator('dependencies')
    def dependencies_must_be_less_than_id(cls, v, values):
        """Ensure dependencies point to previous steps."""
        if 'step_id' in values:
            for dep in v:
                if dep >= values['step_id']:
                    raise ValueError(f"Dependency {dep} must be strictly less than step_id {values['step_id']}")
        return v

class ExecutionResult(BaseModel):
    """
    Represents the result of a single step execution.
    This was missing and causing your ImportError.
    """
    step_id: int
    success: bool
    output: Any
    execution_time: float
    confidence_score: float = 0.0

class ExecutionPlan(BaseModel):
    steps: List[AgentStep]

# ==========================================
# 2. Planner Strategies
# ==========================================

class BasePlannerStrategy(ABC):
    @abstractmethod
    def generate_plan(self, query: str, context: Dict[str, Any], chat_history: Optional[List[Dict[str, str]]] = None) -> List[AgentStep]:
        pass

class RuleBasedPlanner(BasePlannerStrategy):
    """
    Fallback planner using deterministic heuristics.
    UPDATED: Supports dynamic extraction without hardcoded fields.
    """

    def generate_plan(self, query: str, context: Dict[str, Any], chat_history: Optional[List[Dict[str, str]]] = None) -> List[AgentStep]:
        logger.info("Using Rule-Based Planner")
        intent = self._analyze_intent(query)
        return self._get_template(intent, query, context)

    def _analyze_intent(self, query: str) -> str:
        q = query.lower()
        
        # 1. Comparison
        if any(w in q for w in ['compare', 'difference', 'vs ', 'versus']):
            return 'comparison'
            
        # 2. Extraction (Dynamic Data)
        # Broader keywords to catch "Who is...", "What is the date...", "Extract..."
        extraction_triggers = ['extract', 'find', 'get', 'who', 'what', 'when', 'how much', 'details', 'list']
        # We ensure it's not a generic summary request
        if any(w in q for w in extraction_triggers) and 'risk' not in q and 'summary' not in q:
            return 'extraction'
            
        # 3. Default Analysis
        return 'analysis'

    def _get_template(self, intent: str, query: str, context: Dict) -> List[AgentStep]:
        
        # Template: Comparison
        if intent == 'comparison':
            return [
                AgentStep(
                    step_id=1,
                    task_type=TaskType.CONTRACT_COMPARISON,
                    tool_name="contract_comparison_tool",
                    description="Compare provided contracts",
                    inputs={"contracts": context.get("contracts", [])},
                    expected_output="Diff report"
                ),
                AgentStep(
                    step_id=2,
                    task_type=TaskType.RISK_ASSESSMENT,
                    tool_name="contract_qa_tool",
                    description="Analyze risks in differences",
                    inputs={"question": "What are the risks in the differences identified in step 1?"},
                    expected_output="Risk summary",
                    dependencies=[1]
                )
            ]
        
        # Template: Dynamic Extraction (NO HARDCODED FIELDS)
        if intent == 'extraction':
            return [
                AgentStep(
                    step_id=1,
                    task_type=TaskType.CLAUSE_EXTRACTION,
                    tool_name="structured_data_extractor", # Points to the new dynamic tool
                    description=f"Extract data based on: {query}",
                    inputs={
                        # We pass the raw user query. The Tool will figure out the JSON schema.
                        "user_request": query, 
                        # 'contract_text' will be injected by the AgentExecutor automatically
                    },
                    expected_output="JSON Data Object"
                )
            ]

        # Template: General Analysis (Default)
        return [
            AgentStep(
                step_id=1,
                task_type=TaskType.CONTRACT_ANALYSIS,
                tool_name="contract_qa_tool",
                description="Analyze contract based on query",
                inputs={"question": query},
                expected_output="Analysis result"
            )
        ]

class LLMPlanner(BasePlannerStrategy):
    """Base class for LLM-based planners."""

    def __init__(self, client: Any, model_name: str):
        self.client = client
        self.model_name = model_name

    def generate_plan(self, query: str, context: Dict[str, Any], chat_history: Optional[List[Dict[str, str]]] = None) -> List[AgentStep]:
        prompt = self._build_prompt(query, context, chat_history)
        try:
            raw_response = self._call_llm(prompt)
            return self._parse_llm_json(raw_response)
        except Exception as e:
            logger.error(f"{self.model_name} planning failed: {e}. Falling back to Rule-Based.")
            return RuleBasedPlanner().generate_plan(query, context, chat_history)

    def _call_llm(self, prompt: str) -> str:
        if hasattr(self.client, 'generate_text'):
            return self.client.generate_text(prompt)
        return "{}" 

    def _build_prompt(self, query: str, context: Dict, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        # UPDATED PROMPT: Instructs LLM to use 'user_request' input for extraction
        history_str = ""
        if chat_history:
            history_str = "CONVERSATION HISTORY:\n" + "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in chat_history[-5:]]) + "\n\n"

        return (
            "You are a legal AI planner. Convert the user query into a JSON execution plan.\n"
            "AVAILABLE TOOLS:\n"
            "1. contract_qa_tool (For general questions, risks, summaries)\n"
            "2. structured_data_extractor (For extracting specific entities, names, dates, amounts)\n"
            "3. contract_comparison_tool (For comparing two docs)\n\n"
            "IMPORTANT FOR EXTRACTION:\n"
            "If the user asks for specific data (e.g., 'Who is the tenant?', 'Get dates'), use 'structured_data_extractor'.\n"
            "Do NOT guess the JSON keys. Pass the user's exact query into the input field 'user_request'.\n\n"
            f"{history_str}"
            "Return ONLY valid JSON with this schema: \n"
            "{ \"steps\": [ { \"step_id\": int, \"task_type\": str, \"tool_name\": str, "
            "\"description\": str, \"inputs\": dict, \"expected_output\": str, \"dependencies\": [int] } ] }\n\n"
            f"User Query: {query}"
        )

    def _parse_llm_json(self, raw_text: str) -> List[AgentStep]:
        try:
            # Regex to find JSON block
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON object found in response")
            
            clean_json = json_match.group(0)
            data = json.loads(clean_json)
            plan = ExecutionPlan(**data)
            return plan.steps
            
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"JSON Parsing/Validation Error: {e}")
            raise e

# Specific Implementations
class GeminiPlanner(LLMPlanner):
    def _call_llm(self, prompt: str) -> str:
        return self.client.generate_text(prompt)

class OllamaPlanner(LLMPlanner):
    def _call_llm(self, prompt: str) -> str:
        return self.client.generate_text(prompt)

# ==========================================
# 3. The Engine
# ==========================================

class LegalPlanningEngine:
    def __init__(
        self, 
        planner_model: str = "rule_based", 
        gemini_client: Optional[Any] = None, 
        ollama_client: Optional[Any] = None
    ):
        self.planner = self._factory(planner_model, gemini_client, ollama_client)

    def _factory(self, model: str, gemini_c, ollama_c) -> BasePlannerStrategy:
        if model == "gemini" and gemini_c:
            return GeminiPlanner(gemini_c, "Gemini")
        if model == "ollama" and ollama_c:
            return OllamaPlanner(ollama_c, "Ollama")
        return RuleBasedPlanner()

    def create_action_plan(self, user_query: str, context: Dict = None, chat_history: Optional[List[Dict[str, str]]] = None) -> List[AgentStep]:
        context = context or {}
        
        steps = self.planner.generate_plan(user_query, context, chat_history)
        
        if not self._validate_dag(steps):
            logger.warning("Cycle detected or invalid plan. Fallback to Rules.")
            steps = RuleBasedPlanner().generate_plan(user_query, context, chat_history)
            
        return steps

    def _validate_dag(self, steps: List[AgentStep]) -> bool:
        step_ids = {s.step_id for s in steps}
        adj = {s.step_id: s.dependencies for s in steps}
        
        for s in steps:
            for dep in s.dependencies:
                if dep not in step_ids: return False

        visited = set()
        recursion_stack = set()

        def has_cycle(node):
            visited.add(node)
            recursion_stack.add(node)
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor): return True
                elif neighbor in recursion_stack: return True
            recursion_stack.remove(node)
            return False

        for node in step_ids:
            if node not in visited:
                if has_cycle(node): return False
        
        return True