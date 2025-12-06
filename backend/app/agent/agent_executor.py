from __future__ import annotations

import time
import logging
import json
from typing import Any, Dict, Iterable, List, Optional

# FIX: Use relative import for internal module
from .core.planning_engine import AgentStep, ExecutionResult, LegalPlanningEngine

# Fallback imports
try:
    from ..core.config import get_settings
    from ..llm.ollama_client import OllamaClient
    from ..tools import load_default_tools
    from ..tools.base import BaseTool, ToolResult
except ImportError:
    get_settings = None
    load_default_tools = lambda: []
    class BaseTool:
        def __init__(self, name: str, description: str) -> None:
            self.name = name
            self.description = description
        def __call__(self, **kwargs: Any) -> Any:
            return self.run(**kwargs)
        def run(self, **kwargs: Any) -> Any:
            raise NotImplementedError
    class ToolResult:
        def __init__(self, success: bool, content: Any, reasoning: str = "", confidence: float = 0.0, metadata: Dict[str, Any] | None = None) -> None:
            self.success = success
            self.content = content
            self.reasoning = reasoning
            self.confidence = confidence
            self.metadata = metadata or {}


class _UnavailableTool(BaseTool):
    """Lightweight placeholder to keep plans from crashing when a tool is missing."""

    def __init__(self, name: str, reason: str) -> None:
        super().__init__(name=name, description=reason)
        self._reason = reason

    def run(self, **kwargs: Any) -> ToolResult:
        return ToolResult(
            success=False,
            content=f"Tool '{self.name}' is unavailable: {self._reason}",
            reasoning=self._reason,
        )

logger = logging.getLogger(__name__)

class AgentExecutor:
    def __init__(
        self,
        planning_engine: Optional[LegalPlanningEngine] = None,
        tools: Iterable[Any] | None = None,
        llm_client: Optional[Any] = None 
    ) -> None:
        self.planning_engine = planning_engine or LegalPlanningEngine()
        self._tool_registry: Dict[str, Any] = {}
        self.llm_client = llm_client
        self._register_startup_tools(tools)
        self.model = None
        self.tokenizer = None
        
        if not self.llm_client and get_settings:
            try:
                settings = get_settings()
                if settings.ollama_base_url:
                    self.llm_client = OllamaClient(settings.ollama_base_url, settings.ollama_model)
            except Exception:
                pass

    def register_tool(self, tool: Any) -> None:
        if isinstance(tool, str): pass 
        elif "StructuredDataExtractor" in tool.__class__.__name__:
            self._tool_registry["structured_data_extractor"] = tool
        elif "ContractQATool" in tool.__class__.__name__:
            self._tool_registry["contract_qa_tool"] = tool
        elif hasattr(tool, "name"):
            self._tool_registry[tool.name] = tool
        else:
            self._tool_registry[getattr(tool, "name", "unknown").lower()] = tool

    def create_plan(self, query: str, context: Optional[Dict[str, Any]] = None, chat_history: Optional[List[Dict[str, str]]] = None) -> List[AgentStep]:
        return self.planning_engine.create_action_plan(query, context or {}, chat_history)

    def set_model(self, model: Any, tokenizer: Any) -> None:
        """
        Attach a locally loaded model/tokenizer pair for downstream tools that
        may want direct access (e.g., for offline QA). The current executor still
        prefers llm_client.generate_text when available.
        """
        self.model = model
        self.tokenizer = tokenizer

    def execute_query(self, query: str, context: Optional[Dict[str, Any]] = None, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Compatibility wrapper expected by the legacy API."""
        return self.run(query=query, context=context, chat_history=chat_history)

    def execute_plan(self, plan: List[AgentStep], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a precomputed plan (e.g., from an external planner) using the
        same mechanics as `run` but without regenerating the plan.
        """
        if not plan:
            return {
                "answer": "I could not generate an action plan for this request.",
                "plan": [],
                "trace": [],
            }

        trace: List[Dict[str, Any]] = []
        context = context or {}
        intermediate_outputs: Dict[int, Any] = {}

        for step in plan:
            step_inputs = self._prepare_step_inputs(step, context, intermediate_outputs)
            start = time.perf_counter()
            tool = self._tool_registry.get(step.tool_name)

            if tool is None:
                result = ToolResult(success=False, content=f"Tool '{step.tool_name}' missing.")
            else:
                result = self._execute_tool(tool, step.tool_name, step_inputs)

            duration = time.perf_counter() - start

            if isinstance(result, ToolResult):
                output_content = result.content
                is_success = result.success
                metadata = getattr(result, "metadata", {})
            elif isinstance(result, dict) and "content" in result:
                output_content = result["content"]
                is_success = result.get("success", True)
                metadata = result.get("metadata", {})
            else:
                output_content = result
                is_success = True
                metadata = {}

            execution = ExecutionResult(
                step_id=step.step_id,
                success=is_success,
                output=output_content,
                execution_time=duration,
                confidence_score=0.0,
            )

            trace.append({
                "step": step.model_dump(),
                "result": execution.model_dump(),
                "tool_metadata": metadata,
            })

            if not is_success:
                break

            intermediate_outputs[step.step_id] = output_content

        # Reuse summarization path
        answer = self._build_final_answer("Pre-computed plan execution", trace)
        return {"answer": answer, "plan": [step.model_dump() for step in plan], "trace": trace}

    def run(self, query: str, context: Optional[Dict[str, Any]] = None, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        context = context or {}
        plan = self.create_plan(query, context, chat_history)

        if not plan:
            return {
                "answer": "I could not generate an action plan for this request.",
                "plan": [],
                "trace": [],
            }

        trace: List[Dict[str, Any]] = []
        intermediate_outputs: Dict[int, Any] = {}

        for step in plan:
            step_inputs = self._prepare_step_inputs(step, context, intermediate_outputs)
            start = time.perf_counter()
            tool = self._tool_registry.get(step.tool_name)
            
            if tool is None:
                result = ToolResult(success=False, content=f"Tool '{step.tool_name}' missing.")
            else:
                result = self._execute_tool(tool, step.tool_name, step_inputs)

            duration = time.perf_counter() - start
            
            # --- FIX 1: Aggressive Unwrapping ---
            # We want the pure content, not the wrapper object
            if isinstance(result, ToolResult):
                output_content = result.content
                is_success = result.success
                metadata = getattr(result, "metadata", {})
            elif isinstance(result, dict) and "content" in result:
                output_content = result["content"]
                is_success = result.get("success", True)
                metadata = result.get("metadata", {})
            else:
                output_content = result
                is_success = True
                metadata = {}

            execution = ExecutionResult(
                step_id=step.step_id,
                success=is_success,
                output=output_content,
                execution_time=duration,
                confidence_score=0.0,
            )

            trace.append({
                "step": step.model_dump(),
                "result": execution.model_dump(),
                "tool_metadata": metadata,
            })

            if not is_success:
                break

            intermediate_outputs[step.step_id] = output_content

        answer = self._build_final_answer(query, trace)
        return {"answer": answer, "plan": [step.model_dump() for step in plan], "trace": trace}

    def _prepare_step_inputs(self, step: AgentStep, context: Dict[str, Any], intermediate_outputs: Dict[int, Any]) -> Dict[str, Any]:
        prepared = dict(step.inputs)
        
        # Robust Text Finder
        text_to_inject = context.get("contract_text")
        if not text_to_inject and "documents" in context:
            docs = context["documents"]
            if isinstance(docs, list) and len(docs) > 0:
                first_doc = docs[0]
                if "text" in first_doc:
                    text_to_inject = "\n\n".join([d.get("text", "") for d in docs])
                elif "page_content" in first_doc:
                    text_to_inject = "\n\n".join([d.get("page_content", "") for d in docs])
            elif isinstance(docs, dict):
                text_to_inject = "\n\n".join([str(content) for content in docs.values() if content])

        if not text_to_inject:
             text_to_inject = context.get("text") or context.get("content")

        if text_to_inject:
            prepared["contract_text"] = text_to_inject
            prepared["context"] = text_to_inject

        if step.dependencies:
            dep_results = [intermediate_outputs[d] for d in step.dependencies if d in intermediate_outputs]
            if dep_results:
                prepared["context_data"] = dep_results[0] if len(dep_results) == 1 else dep_results
                prepared["previous_output"] = dep_results[0] 

        return prepared

    def _execute_tool(self, tool: Any, tool_name: str, inputs: Dict[str, Any]) -> Any:
        try:
            # Case A: Structured Data Extractor
            if tool_name == "structured_data_extractor":
                user_req = inputs.get("user_request") or inputs.get("query") or ""
                contract_txt = inputs.get("contract_text", "")
                
                # Smart Reroute: If asking for text/clauses, switch to QA tool
                is_clause_search = any(w in user_req.lower() for w in ["clause", "section", "provision", "text of"])
                if is_clause_search:
                    qa_tool = self._tool_registry.get("contract_qa_tool")
                    if qa_tool:
                        return qa_tool.run(question=user_req, context=contract_txt)

                if hasattr(tool, "extract_dynamic"):
                    output = tool.extract_dynamic(contract_text=contract_txt, user_request=user_req)
                    # Extractor returns Dict, wrap it in ToolResult
                    return ToolResult(success=True, content=output, reasoning="Dynamic Extraction")
            
            # Case B: Contract QA Tool
            elif tool_name == "contract_qa_tool":
                q = inputs.get("question") or inputs.get("query")
                ctx = inputs.get("context_data") or inputs.get("context") or inputs.get("contract_text")
                if isinstance(ctx, (dict, list)): ctx = str(ctx)
                
                # QA Tool returns ToolResult directly - pass it through
                if hasattr(tool, "ask_question"):
                    return tool.ask_question(question=q, context=ctx or "")
            
            # Case C: Clause Retrieval
            elif tool_name == "clause_retrieval_tool":
                 if "contract_text" in inputs: return tool.run(**inputs)

            if callable(tool):
                return tool(**inputs)
            
            return ToolResult(success=False, content=f"Tool {tool_name} not found.", reasoning="Method not found")

        except Exception as exc:
            logger.error(f"Tool execution failed: {exc}")
            return ToolResult(success=False, content=str(exc), reasoning="Tool execution exception")

    def _build_final_answer(self, query: str, trace: List[Dict[str, Any]]) -> str:
        successful_steps = [entry for entry in trace if entry["result"]["success"]]
        
        if not successful_steps:
            return "I could not find the information you requested."

        last_result = successful_steps[-1]["result"]["output"]

        # --- NEW LOGIC: Summarize Data into Natural Language ---
        
        # If we have an LLM, ask it to make the data readable
        if self.llm_client and hasattr(self.llm_client, "generate_text"):
            try:
                # If it's a huge list, truncate it slightly for the prompt context
                data_str = str(last_result)[:10000] 
                
                prompt = (
                    f"User Question: {query}\n\n"
                    f"Data Found: {data_str}\n\n"
                    "Task: Answer the user's question naturally using the Data Found. "
                    "If the data is a list of people, list them clearly. "
                    "Do not just dump the JSON. Write a professional response."
                )
                return self.llm_client.generate_text(prompt)
            except Exception as e:
                logger.warning(f"Summarization failed: {e}")

        # Fallback: If no LLM, just return the string (JSON)
        if isinstance(last_result, (dict, list)):
            return json.dumps(last_result, indent=2)
            
        return str(last_result)

    def _register_startup_tools(self, tools: Iterable[Any] | None) -> None:
        toolset = list(tools) if tools is not None else load_default_tools()
        for tool in toolset:
            self.register_tool(tool)

        # Ensure critical tool names exist to keep planner outputs runnable
        if "contract_qa_tool" not in self._tool_registry:
            if self.llm_client:
                try:
                    from ..tools.contract_qa_tool import ContractQATool
                    self.register_tool(ContractQATool(self.llm_client))
                except Exception:
                    self.register_tool(_UnavailableTool("contract_qa_tool", "LLM client not configured"))
            else:
                self.register_tool(_UnavailableTool("contract_qa_tool", "LLM client not configured"))

        if "structured_data_extractor" not in self._tool_registry:
            if self.llm_client:
                try:
                    from ..tools.structured_data_extractor import StructuredDataExtractor
                    self.register_tool(StructuredDataExtractor(self.llm_client))
                except Exception:
                    self.register_tool(_UnavailableTool("structured_data_extractor", "LLM client not configured"))
            else:
                self.register_tool(_UnavailableTool("structured_data_extractor", "LLM client not configured"))
