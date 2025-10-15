"""
Legal Planning Engine for Multi-Step Reasoning
Breaks down complex legal queries into executable steps
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

try:  # Lazy import to avoid hard dependency when used outside the backend
    from backend.app.core.config import get_settings
    from backend.app.llm.gemini_client import GeminiClient, GeminiClientError
except Exception:  # pragma: no cover - fallback for pure local usage
    get_settings = None  # type: ignore
    GeminiClient = None  # type: ignore
    GeminiClientError = Exception  # type: ignore

class TaskType(Enum):
    CONTRACT_ANALYSIS = "contract_analysis"
    CLAUSE_EXTRACTION = "clause_extraction" 
    CONTRACT_COMPARISON = "contract_comparison"
    LEGAL_RESEARCH = "legal_research"
    RISK_ASSESSMENT = "risk_assessment"
    DOCUMENT_GENERATION = "document_generation"

@dataclass
class AgentStep:
    step_id: int
    task_type: TaskType
    description: str
    tool_name: str
    inputs: Dict[str, Any]
    expected_output: str
    dependencies: List[int] = field(default_factory=list)

@dataclass
class ExecutionResult:
    step_id: int
    success: bool
    output: Any
    execution_time: float
    confidence_score: float = 0.0

class LegalPlanningEngine:
    def __init__(self, gemini_client: Optional[GeminiClient] = None):
        self.planning_templates = self._load_planning_templates()
        self._gemini_client: Optional[GeminiClient] = None
        self._planner_model = "rule_based"

        settings = get_settings() if callable(get_settings) else None
        if settings:
            self._planner_model = settings.planner_model.lower()

        if gemini_client is not None:
            self._gemini_client = gemini_client
        elif self._planner_model == "gemini" and settings and getattr(settings, "gemini_api_key", None):
            if GeminiClient is None:  # pragma: no cover - defensive
                logger.warning("Gemini client unavailable; falling back to rule-based planner")
            else:
                try:
                    self._gemini_client = GeminiClient(settings.gemini_api_key, settings.gemini_model)
                except ValueError:
                    logger.warning("Invalid Gemini configuration detected; using rule-based planner")
                    self._planner_model = "rule_based"
        elif self._planner_model == "gemini":
            logger.warning("PLANNER_MODEL set to 'gemini' but GEMINI_API_KEY is missing; using rule-based planner")
            self._planner_model = "rule_based"
    
    def create_action_plan(self, user_query: str, context: Dict = None) -> List[AgentStep]:
        """
        Break down complex legal queries into executable steps
        
        Args:
            user_query: The user's question or request
            context: Additional context (uploaded documents, conversation history)
            
        Returns:
            List of AgentStep objects representing the execution plan
        """
        context = context or {}

        if self._planner_model == "gemini" and self._gemini_client is not None:
            try:
                llm_plan = self._generate_plan_with_gemini(user_query, context)
                if llm_plan:
                    logger.info("Generated plan with Gemini model")
                    return llm_plan
                logger.warning("Gemini did not return a usable plan; falling back to rule-based templates")
            except GeminiClientError as exc:  # type: ignore[arg-type]
                logger.warning("Gemini planner failed: %s", exc)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Unexpected Gemini planner failure: %s", exc)

        # Analyze query intent via rule-based heuristics
        query_intent = self._analyze_query_intent(user_query)

        # Select appropriate planning template
        template = self._select_planning_template(query_intent)

        # Generate execution plan
        plan = self._generate_plan_from_template(template, user_query, context)
        
        logger.info(f"Created {len(plan)}-step plan for query: {user_query[:100]}...")
        return plan
    
    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze user query to determine intent and required actions"""
        query_lower = query.lower()
        
        intents = {
            'contract_review': any(term in query_lower for term in ['review', 'analyze', 'check contract']),
            'clause_extraction': any(term in query_lower for term in ['find clause', 'extract', 'locate term']),
            'comparison': any(term in query_lower for term in ['compare', 'difference', 'vs', 'versus']),
            'risk_assessment': any(term in query_lower for term in ['risk', 'liability', 'danger', 'problem']),
            'legal_research': any(term in query_lower for term in ['research', 'law', 'regulation', 'precedent']),
        }
        
        return {
            'primary_intent': max(intents.items(), key=lambda x: x[1])[0],
            'intents': intents,
            'complexity': 'high' if len([v for v in intents.values() if v]) > 2 else 'medium'
        }
    
    def _select_planning_template(self, query_intent: Dict[str, Any]) -> str:
        """Select the appropriate planning template based on query intent"""
        primary_intent = query_intent['primary_intent']
        complexity = query_intent['complexity']
        
        template_map = {
            'contract_review': 'comprehensive_contract_analysis',
            'clause_extraction': 'targeted_clause_search',
            'comparison': 'contract_comparison',
            'risk_assessment': 'risk_analysis',
            'legal_research': 'legal_research'
        }
        
        return template_map.get(primary_intent, 'general_legal_query')
    
    def _generate_plan_from_template(self, template: str, query: str, context: Dict) -> List[AgentStep]:
        """Generate specific execution plan from template"""
        
        if template == 'comprehensive_contract_analysis':
            return [
                AgentStep(
                    step_id=1,
                    task_type=TaskType.CONTRACT_ANALYSIS,
                    description="Extract key contract information and parties",
                    tool_name="contract_qa_tool",
                    inputs={"question": "Who are the main parties and what are the key terms?", "context": context},
                    expected_output="Party information and key terms summary"
                ),
                AgentStep(
                    step_id=2,
                    task_type=TaskType.CLAUSE_EXTRACTION,
                    description="Identify critical clauses (termination, liability, IP)",
                    tool_name="clause_retrieval_tool",
                    inputs={"clause_types": ["termination", "liability", "intellectual_property"]},
                    expected_output="List of critical clauses with locations",
                    dependencies=[1]
                ),
                AgentStep(
                    step_id=3,
                    task_type=TaskType.RISK_ASSESSMENT,
                    description="Analyze potential risks and liabilities",
                    tool_name="contract_qa_tool",
                    inputs={"question": "What are the main risks and liabilities in this contract?"},
                    expected_output="Risk assessment with severity levels",
                    dependencies=[1, 2]
                ),
                AgentStep(
                    step_id=4,
                    task_type=TaskType.DOCUMENT_GENERATION,
                    description="Generate executive summary with recommendations",
                    tool_name="contract_qa_tool",
                    inputs={"question": "Provide an executive summary with key recommendations"},
                    expected_output="Executive summary document",
                    dependencies=[1, 2, 3]
                )
            ]
        
        elif template == 'contract_comparison':
            return [
                AgentStep(
                    step_id=1,
                    task_type=TaskType.CONTRACT_COMPARISON,
                    description="Compare contracts and identify differences",
                    tool_name="contract_comparison_tool",
                    inputs={"contracts": context.get("contracts", [])},
                    expected_output="Detailed comparison highlighting differences"
                ),
                AgentStep(
                    step_id=2,
                    task_type=TaskType.RISK_ASSESSMENT,
                    description="Assess risks from identified differences",
                    tool_name="contract_qa_tool",
                    inputs={"question": "What risks arise from the contract differences?"},
                    expected_output="Risk analysis of contract variations",
                    dependencies=[1]
                )
            ]
        
        elif template == 'targeted_clause_search':
            return [
                AgentStep(
                    step_id=1,
                    task_type=TaskType.CLAUSE_EXTRACTION,
                    description="Search for specific clauses requested by user",
                    tool_name="clause_retrieval_tool",
                    inputs={"query": query},
                    expected_output="Matching clauses with context"
                )
            ]
        
        else:  # general_legal_query
            return [
                AgentStep(
                    step_id=1,
                    task_type=TaskType.CONTRACT_ANALYSIS,
                    description="Answer legal query using contract analysis",
                    tool_name="contract_qa_tool",
                    inputs={"question": query, "context": context},
                    expected_output="Direct answer to user query"
                )
            ]

    def _generate_plan_with_gemini(self, query: str, context: Dict[str, Any]) -> List[AgentStep]:
        if self._gemini_client is None:  # pragma: no cover - guard
            return []

        prompt = self._build_gemini_prompt(query, context)
        raw_plan = self._gemini_client.generate_text(prompt)

        try:
            plan_dict = self._extract_plan_json(raw_plan)
        except GeminiClientError as exc:  # type: ignore[arg-type]
            logger.warning("Failed to parse Gemini plan: %s", exc)
            return []

        steps_payload = plan_dict.get("steps", [])
        if not isinstance(steps_payload, list):
            logger.warning("Gemini plan missing 'steps' array")
            return []

        steps: List[AgentStep] = []
        for index, step_data in enumerate(steps_payload, start=1):
            if not isinstance(step_data, dict):
                continue

            tool_name = step_data.get("tool_name") or step_data.get("tool")
            if not isinstance(tool_name, str):
                continue
            tool_name = tool_name.strip()
            if tool_name not in {"contract_qa_tool", "clause_retrieval_tool", "contract_comparison_tool"}:
                logger.debug("Skipping unsupported tool '%s' in Gemini plan", tool_name)
                continue

            task_type_value = step_data.get("task_type") or step_data.get("task")
            task_type = self._resolve_task_type(task_type_value, tool_name)

            description = step_data.get("description") or "Execute tool step"
            expected_output = step_data.get("expected_output") or step_data.get("expected") or "Tool output"
            inputs = step_data.get("inputs") or {}
            if not isinstance(inputs, dict):
                inputs = {}
            dependencies = step_data.get("dependencies") or step_data.get("dependent_on") or []
            if not isinstance(dependencies, list):
                dependencies = []

            step_id = step_data.get("id") or step_data.get("step_id") or index

            steps.append(
                AgentStep(
                    step_id=int(step_id),
                    task_type=task_type,
                    description=str(description),
                    tool_name=tool_name,
                    inputs=inputs,
                    expected_output=str(expected_output),
                    dependencies=[int(dep) for dep in dependencies if isinstance(dep, int)]
                )
            )

        return steps

    def _build_gemini_prompt(self, query: str, context: Dict[str, Any]) -> str:
        context_hint = "Contracts are provided in the context object." if context else "No contract context provided."

        return (
            "You are the planning brain for a legal contract assistant. "
            "Create a structured plan to answer the user's request by calling specialised tools.\n"
            "Respond strictly with valid JSON using the schema: {\"steps\": ["  # noqa: E501
            "{\"id\": int, \"task_type\": str, \"tool_name\": str, \"description\": str, "
            "\"inputs\": object, \"expected_output\": str, \"dependencies\": [int]} ]}.\n"
            "Only use the following tools: "
            "contract_qa_tool (answers targeted legal questions using contract text), "
            "clause_retrieval_tool (finds clauses by type or keywords), "
            "contract_comparison_tool (compares two contracts).\n"
            "Set \"dependencies\" when a step relies on the results of an earlier step. "
            "Ensure inputs include the question or clause types to run the tool.\n"
            f"Context hint: {context_hint}\n"
            "User query: " + query
        )

    def _extract_plan_json(self, raw_plan: str) -> Dict[str, Any]:
        text = raw_plan.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise GeminiClientError("Gemini response did not contain JSON")  # type: ignore[arg-type]

        json_blob = text[start : end + 1]
        try:
            return json.loads(json_blob)
        except json.JSONDecodeError as exc:
            raise GeminiClientError("Unable to decode Gemini JSON payload") from exc  # type: ignore[arg-type]

    @staticmethod
    def _resolve_task_type(value: Optional[str], tool_name: str) -> TaskType:
        if not value:
            return LegalPlanningEngine._default_task_for_tool(tool_name)

        normalised = value.replace("-", "_").lower()
        for task in TaskType:
            if task.value == normalised or task.name.lower() == normalised:
                return task

        return LegalPlanningEngine._default_task_for_tool(tool_name)

    @staticmethod
    def _default_task_for_tool(tool_name: str) -> TaskType:
        if tool_name == "clause_retrieval_tool":
            return TaskType.CLAUSE_EXTRACTION
        if tool_name == "contract_comparison_tool":
            return TaskType.CONTRACT_COMPARISON
        return TaskType.CONTRACT_ANALYSIS
    
    def _load_planning_templates(self) -> Dict[str, Any]:
        """Load planning templates for different query types"""
        # In a full implementation, these could be loaded from files
        return {
            "comprehensive_contract_analysis": {
                "steps": ["extract_key_info", "find_critical_clauses", "assess_risks", "generate_summary"],
                "tools": ["contract_qa_tool", "clause_retrieval_tool"]
            },
            "contract_comparison": {
                "steps": ["compare_contracts", "assess_differences"],
                "tools": ["contract_comparison_tool", "contract_qa_tool"]
            }
        }

    def validate_plan(self, plan: List[AgentStep]) -> bool:
        """Validate that the execution plan is coherent and executable"""
        # Check dependencies
        step_ids = {step.step_id for step in plan}
        
        for step in plan:
            if step.dependencies:
                for dep_id in step.dependencies:
                    if dep_id not in step_ids:
                        logger.error(f"Step {step.step_id} has invalid dependency {dep_id}")
                        return False
        
        logger.info("Plan validation successful")
        return True