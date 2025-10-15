from __future__ import annotations

import difflib
from typing import Any, Dict, List

from .base import BaseTool, ToolExecutionError, ToolResult


class ContractComparisonTool(BaseTool):
    """Simple textual comparison between two contract versions."""

    def __init__(self) -> None:
        super().__init__(
            name="contract_comparison_tool",
            description="Compare two contract texts and highlight key differences.",
        )

    def run(
        self,
        contracts: List[Dict[str, Any]] | None = None,
        context: Dict[str, Any] | None = None,
    ) -> ToolResult:
        contracts = contracts or (context or {}).get("contracts")
        if not contracts or len(contracts) < 2:
            raise ToolExecutionError("Two contract documents are required for comparison")

        left = contracts[0].get("text") if isinstance(contracts[0], dict) else str(contracts[0])
        right = contracts[1].get("text") if isinstance(contracts[1], dict) else str(contracts[1])

        if not left or not right:
            raise ToolExecutionError("Both contract entries must include text content")

        diff = difflib.unified_diff(
            left.splitlines(),
            right.splitlines(),
            fromfile="Contract A",
            tofile="Contract B",
            lineterm="",
        )
        diff_output = "\n".join(diff)

        summary = {
            "differences": diff_output or "No textual differences detected.",
            "left_length": len(left.splitlines()),
            "right_length": len(right.splitlines()),
        }

        return ToolResult(
            success=True,
            content=summary,
            reasoning="Comparison generated using a unified diff between the two contract texts.",
            confidence=0.5,
        )
