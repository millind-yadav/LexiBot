from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .base import BaseTool, ToolExecutionError, ToolResult

logger = logging.getLogger(__name__)

class ContractQATool(BaseTool):
    """
    Answers natural language questions about a contract using an LLM.
    Replaces the old snippet-only approach with actual reasoning.
    """

    def __init__(self, llm_client: Any) -> None:
        super().__init__(
            name="contract_qa_tool",
            description="Answer contract-related questions using the provided text.",
        )
        self.client = llm_client

    def run(
        self, 
        question: str, 
        context: Dict[str, Any] | str | None = None, 
        **kwargs: Any
    ) -> ToolResult:
        """
        Executes the QA. 
        Accepts **kwargs to safely ignore extra arguments (like 'contract_text') 
        passed by the AgentExecutor.
        """
        if not question:
            return ToolResult(success=False, content="No question provided.")

        # --- 1. Resolve Contract Text ---
        # The text might come in 'context' (as dict or str) or in 'kwargs'
        doc_text = ""
        
        # Case A: context is the text string (Simple)
        if isinstance(context, str):
            doc_text = context
            
        # Case B: context is a dict (Standard)
        elif isinstance(context, dict):
            doc_text = context.get("contract_text") or context.get("text", "")
            # Fallback: Check inside 'documents' list
            if not doc_text and "documents" in context:
                docs = context["documents"]
                if isinstance(docs, dict):
                    doc_text = "\n".join([str(v) for v in docs.values()])
                elif isinstance(docs, list):
                    doc_text = "\n".join([d.get("text", "") for d in docs if isinstance(d, dict)])

        # Case C: Passed explicitly in kwargs (from AgentExecutor injection)
        if not doc_text and "contract_text" in kwargs:
            doc_text = kwargs["contract_text"]

        # --- 2. Safety Check ---
        if not doc_text:
            return ToolResult(
                success=False, 
                content="I cannot answer this because I don't have the contract text.",
                reasoning="Missing contract context."
            )

        # --- 3. Generate Answer ---
        try:
            return self.ask_question(question, doc_text)
        except Exception as e:
            logger.error(f"QA Tool Failed: {e}")
            return ToolResult(success=False, content=f"Error analyzing contract: {str(e)}")

    def ask_question(self, question: str, context: str) -> ToolResult:
        # Truncate context to ~15k chars to prevent token overflow on smaller models
        safe_context = context[:15000]
        
        prompt = (
            "You are an expert Legal AI. Answer the user's question based STRICTLY on the contract text provided below.\n"
            "Rules:\n"
            "1. If the answer is found, quote the specific clause number or section if possible.\n"
            "2. If the answer is NOT in the text, state clearly: 'The contract does not mention this.'\n"
            "3. Be concise and professional.\n\n"
            f"--- CONTRACT TEXT ---\n{safe_context}\n"
            f"---------------------\n\n"
            f"User Question: {question}"
        )
        
        # Call the LLM (Ollama or Gemini)
        response = self.client.generate_text(prompt)
        
        return ToolResult(
            success=True,
            content=response,
            reasoning="LLM Analysis completed successfully.",
            confidence=0.9
        )