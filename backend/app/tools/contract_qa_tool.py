from __future__ import annotations

import textwrap
from typing import Any, Dict, List

from .base import BaseTool, ToolExecutionError, ToolResult
from .clause_retrieval_tool import ClauseRetrievalTool


class ContractQATool(BaseTool):
    """Lightweight question answering over supplied contract context."""

    def __init__(self) -> None:
        super().__init__(
            name="contract_qa_tool",
            description="Answer contract-related questions using provided context snippets.",
        )
        self._retriever = ClauseRetrievalTool()

    def run(
        self,
        question: str,
        context: Dict[str, Any] | None = None,
        max_snippets: int = 3,
    ) -> ToolResult:
        if not question:
            raise ToolExecutionError("A question is required for contract QA")

        context = context or {}
        retrieval_result = self._retriever.run(query=question, context=context, top_k=max_snippets)

        if not retrieval_result.success:
            return ToolResult(
                success=False,
                content="I could not find relevant information in the provided material.",
                reasoning="Clause retrieval produced no candidates.",
                confidence=0.0,
            )

        snippets: List[Dict[str, Any]] = retrieval_result.content  # type: ignore[assignment]
        answer = self._summarise_snippets(question, snippets)
        return ToolResult(
            success=True,
            content=answer,
            reasoning="Answer generated from top-matching contract snippets.",
            confidence=min(0.8, retrieval_result.confidence + 0.1),
            metadata={"supporting_clauses": snippets},
        )

    @staticmethod
    def _summarise_snippets(question: str, snippets: List[Dict[str, Any]]) -> str:
        fallback_markers = (
            "no clause in the supplied text answers this question",
            "the answer is not present",
            "this portion of the contract does not specify",
        )

        def is_fallback(text: str) -> bool:
            lowered = text.lower()
            return any(marker in lowered for marker in fallback_markers)

        primary_snippets: List[Dict[str, Any]] = []
        fallback_snippets: List[Dict[str, Any]] = []

        for snippet in snippets:
            text = snippet.get("text", "")
            if is_fallback(text):
                fallback_snippets.append(snippet)
            else:
                primary_snippets.append(snippet)

        display_snippets = primary_snippets or fallback_snippets

        formatted_snippets = []
        for idx, snippet in enumerate(display_snippets, start=1):
            text = textwrap.shorten(snippet.get("text", ""), width=400, placeholder="...")
            metadata = snippet.get("metadata", {})
            location = metadata.get("location") or metadata.get("section") or "Unknown section"
            hints = metadata.get("keyword_hints")
            hint_suffix = f" [{', '.join(hints)}]" if hints else ""
            formatted_snippets.append(f"- Clause {idx} ({location}){hint_suffix}: \"{text}\"")

        bullet_list = "\n".join(formatted_snippets)
        return (
            f"Question: {question}\n"
            "Answer: Based on the supplied contract passages, the following excerpts warrant legal review:\n"
            f"{bullet_list}\n"
            "\nPlease confirm whether you need further analysis or a formal write-up."
        )
