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
        top_k: int = 10,
    ) -> ToolResult:
        context = context or {}
        documents: List[Dict[str, Any]] = self._extract_documents(context)

        if not documents:
            raise ToolExecutionError("No documents provided for clause retrieval")
            
        # Debug logging
        total_len = sum(len(d.get("text", "")) for d in documents)
        print(f"DEBUG: ClauseRetrievalTool received {len(documents)} docs, total length: {total_len} chars")
        if total_len > 0:
            print(f"DEBUG: Start: {documents[0].get('text', '')[:100]}")
            print(f"DEBUG: End: {documents[-1].get('text', '')[-100:]}")

        try:
            from ..core.rag import rag_service
            
            vectorstore = rag_service.create_index(documents)
            if not vectorstore:
                 return ToolResult(success=False, content="Could not create index from documents.")
            
            full_query = query or ""
            
            # Augment query with synonyms for better retrieval
            # Check both the raw query and the clause types
            combined_text = (full_query + " " + " ".join(clause_types or [])).lower()
            
            if "termination" in combined_text or "terminate" in combined_text:
                full_query += " ending break clause notice period quit surrender expiration"
            if "liability" in combined_text:
                full_query += " indemnity damages responsible"
            if "tenant" in combined_text:
                full_query += " occupier resident lessee"
            if "viewing" in combined_text or "view" in combined_text:
                full_query += " show access inspect visit entry"
            
            if clause_types:
                full_query += " " + " ".join(clause_types)

            results = rag_service.query(vectorstore, full_query, k=top_k)
            
            if not results:
                 return ToolResult(
                    success=False,
                    content="No matching clauses were found.",
                    reasoning="The retrieval heuristic could not match the query against supplied documents.",
                    confidence=0.0,
                )
                 
            return ToolResult(
                success=True,
                content=results,
                reasoning="Retrieved relevant clauses using FAISS vector search.",
                confidence=0.85,
                metadata={"match_count": len(results)}
            )
        except Exception as e:
            raise ToolExecutionError(f"RAG Retrieval failed: {str(e)}")

    @staticmethod
    def _extract_documents(context: Dict[str, Any]) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        
        # Handle dictionary format from frontend (filename -> content)
        if isinstance(context.get("documents"), dict):
            for filename, content in context["documents"].items():
                docs.append({"text": content, "metadata": {"source": filename}})
        
        # Handle list format
        elif isinstance(context.get("documents"), list):
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
