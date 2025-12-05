import json
import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class StructuredDataExtractor:
    """
    Dynamic Tool: Generates its own schema based on user requests, 
    then extracts data. No hardcoded fields!
    """
    def __init__(self, llm_client: Any):
        self.client = llm_client

    def extract_dynamic(self, contract_text: str, user_request: str) -> Dict[str, Any]:
        logger.info(f"Dynamic Extraction Request: {user_request}")
        
        # Step 1: Ask LLM what fields to look for
        target_fields = self._generate_schema(user_request)
        if not target_fields:
            return {"error": "Could not determine fields to extract."}
            
        logger.info(f"Generated Schema: {target_fields}")

        # Step 2: Extract those specific fields
        return self._perform_extraction(contract_text, target_fields)

    def _generate_schema(self, user_request: str) -> List[str]:
        """Translates natural language request into JSON keys."""
        prompt = (
            "You are a Data Schema Architect. Convert the user's request into a list of specific JSON keys.\n"
            "Rules: Return ONLY a JSON list of strings. Use snake_case.\n"
            "Example: 'Who signed?' -> [\"signatory_name\", \"signing_date\"]\n"
            f"User Request: \"{user_request}\""
        )
        response = self.client.generate_text(prompt)
        return self._parse_json_list(response)

    def _perform_extraction(self, text: str, fields: List[str]) -> Dict[str, Any]:
        """Extracts data for the generated keys."""
        schema_example = {field: "value_from_text_or_null" for field in fields}
        prompt = (
            f"Extract specific values from the contract text below.\n"
            f"Return valid JSON matching this structure: {json.dumps(schema_example)}\n"
            f"Contract Text: {text[:10000]}..." 
        )
        response = self.client.generate_text(prompt)
        return self._parse_json_dict(response)

    def _parse_json_list(self, text: str) -> List[str]:
        try:
            match = re.search(r'\[.*\]', text, re.DOTALL)
            return json.loads(match.group(0)) if match else []
        except:
            return []

    def _parse_json_dict(self, text: str) -> Dict:
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            return json.loads(match.group(0)) if match else {}
        except:
            return {}

class ContractQATool:
    """Standard RAG tool for general questions."""
    def __init__(self, llm_client: Any):
        self.client = llm_client

    def ask_question(self, question: str, context: str = "") -> str:
        prompt = f"Context: {context}\n\nAnswer this legal question: {question}"
        return self.client.generate_text(prompt)