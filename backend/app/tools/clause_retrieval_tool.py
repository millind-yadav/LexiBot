from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

from .base import BaseTool, ToolExecutionError, ToolResult


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def _score(text: str, query_tokens: Iterable[str]) -> float:
    if not text:
        return 0.0
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    score = 0.0
    for token in query_tokens:
        score += counts.get(token, 0)
    return score / math.sqrt(len(tokens))


class ClauseRetrievalTool(BaseTool):
    """Naïve clause retrieval over provided contract context."""

    _KEYWORD_HINTS: Dict[str, Dict[str, Any]] = {
        "exclusiv": {
            "label": "exclusivity",
            "weight": 0.85,
            "patterns": [
                r"\bexclusive\b",
                r"\bexclusiv(?:e|ity|ely)\b",
                r"\bsole (?:right|license)\b",
                r"\bexclusive (?:right|licen[cs]e)\b",
            ],
        },
        "joint ip": {
            "label": "joint intellectual property",
            "weight": 0.6,
            "patterns": [
                r"\bjoint(?:ly)? owned\b",
                r"\bjoint ownership\b",
                r"\bco-?ownership\b",
                r"\bshared (?:ip|intellectual property)\b",
            ],
        },
        "joint ownership": {
            "label": "joint intellectual property",
            "weight": 0.6,
            "patterns": [
                r"\bjoint(?:ly)? owned\b",
                r"\bjoint ownership\b",
                r"\bco-?ownership\b",
                r"\bshared (?:ip|intellectual property)\b",
            ],
        },
        "effective date": {
            "label": "effective date",
            "weight": 0.55,
            "patterns": [
                r"\beffective (?:date|upon|as of)\b",
                r"\bcomes? into force\b",
                r"\bshall be effective\b",
                r"\bcommence(?:s|ment)?\b",
            ],
        },
        "effective": {
            "label": "effective date",
            "weight": 0.4,
            "patterns": [
                r"\beffective (?:date|upon|as of)\b",
                r"\bcomes? into force\b",
                r"\bshall be effective\b",
            ],
        },
        "profit": {
            "label": "revenue/profit sharing",
            "weight": 0.65,
            "patterns": [
                r"\brevenue sharing\b",
                r"\bprofit sharing\b",
                r"\bnet profits?\b",
                r"\bshare of (?:profits|revenue)\b",
                r"\bdistribution of profits?\b",
            ],
        },
        "revenue": {
            "label": "revenue/profit sharing",
            "weight": 0.6,
            "patterns": [
                r"\brevenue sharing\b",
                r"\bnet profits?\b",
                r"\bprofit (?:distribution|allocation)\b",
            ],
        },
    }

    def __init__(self) -> None:
        super().__init__(
            name="clause_retrieval_tool",
            description="Retrieve clauses from supplied documents that match a query or clause types.",
        )

    def run(
        self,
        query: str | None = None,
        clause_types: List[str] | None = None,
        context: Dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> ToolResult:
        context = context or {}
        documents: List[Dict[str, Any]] = self._extract_documents(context)

        if not documents:
            raise ToolExecutionError("No documents provided for clause retrieval")

        tokens = _tokenize(query or "")
        keyword_hints = self._build_keyword_hints(query or "")
        matches: List[Tuple[Dict[str, Any], float, List[str]]] = []

        for doc in documents:
            text = doc.get("text") or ""
            meta = doc.get("metadata", {})

            if clause_types:
                doc_clause_type = meta.get("clause_type", "").lower()
                if doc_clause_type and any(ct.lower() in doc_clause_type for ct in clause_types):
                    matches.append((doc, 1.0, []))
                    continue

            score = _score(text, tokens) if tokens else 0.0
            matched_labels: List[str] = []

            if keyword_hints:
                boost = 0.0
                lowered = text.lower()
                for hint in keyword_hints:
                    if any(pattern.search(lowered) for pattern in hint["patterns"]):
                        boost += hint["weight"]
                        matched_labels.append(hint["label"])
                score += boost

            if score > 0:
                matches.append((doc, score, sorted(set(matched_labels))))

        matches.sort(key=lambda item: item[1], reverse=True)
        selected = [self._format_match(doc, score, hints) for doc, score, hints in matches[:top_k]]

        if not selected:
            return ToolResult(
                success=False,
                content="No matching clauses were found.",
                reasoning="The retrieval heuristic could not match the query against supplied documents.",
                confidence=0.0,
            )

        return ToolResult(
            success=True,
            content=selected,
            reasoning="Returned clauses ranked by lexical overlap with the query and requested clause types.",
            confidence=0.65,
            metadata={"match_count": len(selected)},
        )

    @staticmethod
    def _extract_documents(context: Dict[str, Any]) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        if isinstance(context.get("documents"), list):
            docs.extend(context["documents"])
        if isinstance(context.get("contracts"), list):
            for contract in context["contracts"]:
                sections = contract.get("sections") if isinstance(contract, dict) else None
                if isinstance(sections, list):
                    docs.extend(sections)
        return docs

    @classmethod
    def _format_match(cls, doc: Dict[str, Any], score: float, hints: List[str]) -> Dict[str, Any]:
        metadata = dict(doc.get("metadata", {}))
        if hints:
            metadata["keyword_hints"] = hints
        return {
            "text": doc.get("text", ""),
            "metadata": metadata,
            "confidence": round(min(score, 1.0), 3),
        }

    @classmethod
    def _build_keyword_hints(cls, query: str) -> List[Dict[str, Any]]:
        lowered = (query or "").lower()
        hints: List[Dict[str, Any]] = []
        if not lowered:
            return hints

        added_labels: set[str] = set()
        for trigger, config in cls._KEYWORD_HINTS.items():
            if trigger in lowered and config["label"] not in added_labels:
                patterns = [re.compile(pattern, re.IGNORECASE) for pattern in config["patterns"]]
                hints.append({"label": config["label"], "weight": config["weight"], "patterns": patterns})
                added_labels.add(config["label"])
        return hints
