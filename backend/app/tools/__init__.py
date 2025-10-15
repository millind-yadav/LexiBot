from __future__ import annotations

from typing import Iterable, List

from .base import BaseTool
from .clause_retrieval_tool import ClauseRetrievalTool
from .contract_comparison_tool import ContractComparisonTool
from .contract_qa_tool import ContractQATool


def load_default_tools() -> List[BaseTool]:
    """Instantiate the default toolset used by the agent executor."""

    return [
        ClauseRetrievalTool(),
        ContractQATool(),
        ContractComparisonTool(),
    ]


def register_tools(executor, tools: Iterable[BaseTool] | None = None) -> None:
    """Register tools on an executor instance following the expected interface."""
    toolset = tools or load_default_tools()
    for tool in toolset:
        executor.register_tool(tool)
