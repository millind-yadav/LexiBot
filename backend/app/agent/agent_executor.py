from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional

from agent.core.planning_engine import AgentStep, ExecutionResult, LegalPlanningEngine

from ..tools import load_default_tools
from ..tools.base import BaseTool, ToolExecutionError, ToolResult
from .prompt_templates import build_summary_prompt


class AgentExecutor:
    """Coordinates planning and tool execution for legal workflows."""

    def __init__(
        self,
        planning_engine: Optional[LegalPlanningEngine] = None,
        tools: Iterable[BaseTool] | None = None,
    ) -> None:
        self.planning_engine = planning_engine or LegalPlanningEngine()
        self._tool_registry: Dict[str, BaseTool] = {}
        self._register_startup_tools(tools)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register_tool(self, tool: BaseTool) -> None:
        self._tool_registry[tool.name] = tool

    def create_plan(self, query: str, context: Optional[Dict[str, Any]] = None) -> List[AgentStep]:
        return self.planning_engine.create_action_plan(query, context or {})

    def run(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        context = context or {}
        plan = self.create_plan(query, context)

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
                result = ToolResult(
                    success=False,
                    content=f"Tool '{step.tool_name}' is not registered.",
                    reasoning="Missing tool registration",
                )
            else:
                result = self._execute_tool(tool, step_inputs)

            duration = time.perf_counter() - start
            execution = ExecutionResult(
                step_id=step.step_id,
                success=result.success,
                output=result.content,
                execution_time=duration,
                confidence_score=result.confidence,
            )

            trace.append(
                {
                    "step": asdict(step),
                    "result": self._serialise_execution_result(execution),
                    "tool_reasoning": result.reasoning,
                    "tool_metadata": result.metadata,
                }
            )

            if not result.success:
                break

            intermediate_outputs[step.step_id] = result.content

        answer = self._build_final_answer(query, trace)
        return {
            "answer": answer,
            "plan": [asdict(step) for step in plan],
            "trace": trace,
        }

    def execute_tool(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """Execute a registered tool directly with the supplied keyword arguments."""

        tool = self._tool_registry.get(tool_name)
        if tool is None:
            return ToolResult(
                success=False,
                content=f"Tool '{tool_name}' is not registered.",
                reasoning="Missing tool registration",
            )

        inputs = dict(kwargs)
        return self._execute_tool(tool, inputs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _register_startup_tools(self, tools: Iterable[BaseTool] | None) -> None:
        toolset = list(tools) if tools is not None else load_default_tools()
        for tool in toolset:
            self.register_tool(tool)

    def _prepare_step_inputs(
        self,
        step: AgentStep,
        context: Dict[str, Any],
        intermediate_outputs: Dict[int, Any],
    ) -> Dict[str, Any]:
        prepared = dict(step.inputs)
        prepared.setdefault("context", context)

        if step.dependencies:
            prepared["dependencies"] = {
                dep_id: intermediate_outputs.get(dep_id) for dep_id in step.dependencies
            }
        return prepared

    def _execute_tool(self, tool: BaseTool, inputs: Dict[str, Any]) -> ToolResult:
        try:
            return tool(**inputs)
        except ToolExecutionError as exc:
            return ToolResult(success=False, content=str(exc), reasoning="Tool execution error")
        except Exception as exc:  # pragma: no cover - defensive
            return ToolResult(
                success=False,
                content=str(exc),
                reasoning="Unexpected tool failure",
                metadata={"exception_type": exc.__class__.__name__},
            )

    def _build_final_answer(self, query: str, trace: List[Dict[str, Any]]) -> str:
        successful_steps = [entry for entry in trace if entry["result"]["success"]]
        if not successful_steps:
            failure = trace[-1] if trace else None
            if failure:
                return (
                    "I could not complete the analysis. The last executed tool reported: "
                    f"{failure['result']['output']}"
                )
            return "I could not make progress on this request."

        summaries = []
        for entry in successful_steps:
            step = entry["step"]
            result = entry["result"]
            summary = (
                f"Step {step['step_id']} ({step['description']}): {result['output']}"
            )
            summaries.append(summary)

        prompt = build_summary_prompt(query, summaries)
        # We do not call an LLM here; instead return the stitched summary directly.
        return prompt + "\n\nFinal Answer:\n" + "\n".join(summaries)

    @staticmethod
    def _serialise_execution_result(result: ExecutionResult) -> Dict[str, Any]:
        return {
            "step_id": result.step_id,
            "success": result.success,
            "output": result.output,
            "execution_time": result.execution_time,
            "confidence_score": result.confidence_score,
        }
