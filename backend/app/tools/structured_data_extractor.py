import json
import logging
import re
from typing import List, Dict, Any
from .base import BaseTool

logger = logging.getLogger(__name__)

class StructuredDataExtractor(BaseTool):
    def __init__(self, llm_client: Any):
        super().__init__(
            name="structured_data_extractor",
            description="Extracts entities based on natural language requests."
        )
        self.client = llm_client

    def extract_dynamic(self, contract_text: str, user_request: str) -> Any:
        logger.info(f"Dynamic Extraction: {user_request}")
        
        if not contract_text or len(contract_text.strip()) < 20:
            logger.error("❌ ERROR: Contract text is empty or too short.")
            return {"error": "No contract text provided."}
        
        target_fields = self._generate_schema(user_request)
        if not target_fields:
            return {"error": "Could not determine fields to extract."}
        logger.info(f"Target Fields: {target_fields}")

        return self._perform_extraction(contract_text, target_fields)

    def _generate_schema(self, user_request: str) -> List[str]:
        # --- FIX 1: SIMPLER SCHEMA PROMPT ---
        prompt = (
            f"You are a Data Schema Architect. Convert this request: '{user_request}' "
            "into a list of simple JSON keys (snake_case).\n\n"
            "RULES:\n"
            "1. Keep keys simple (e.g. use 'address' instead of 'street', 'city', 'state').\n"
            "2. Do not nest objects.\n"
            "3. Return JSON List ONLY.\n\n"
            "Example: [\"tenant_name\", \"address\", \"rent_amount\"]\n"
        )
        response = self.client.generate_text(prompt)
        return self._parse_json_list(response)

    def _perform_extraction(self, text: str, fields: List[str]) -> Any:
        schema_obj = {field: "value_found_or_null" for field in fields}
        
        # --- FIX 2: STRONGER LIST INSTRUCTION ---
        prompt = (
            f"Extract values from the contract text for these fields: {fields}\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. **Look for Lists**: If the text lists multiple people (e.g. under 'The Tenant(s):'), you MUST return a JSON LIST of objects, one for each person.\n"
            "2. **Capture Everyone**: Do not stop after the first match.\n"
            "3. **Simple Text**: Copy the exact text from the document. Do not try to split addresses.\n\n"
            f"Expected JSON Format: [{json.dumps(schema_obj)}, ...]\n\n"
            "RULES:\n"
            "- Return JSON ONLY. No markdown.\n"
            "- If a value is missing, use null.\n\n"
            f"Contract Text: {text[:15000]}..." 
        )
        
        response = self.client.generate_text(prompt)
        logger.info(f"LLM Raw Response: {response}")
        
        data = self._parse_json_result(response)
        
        if data is None:
            logger.error("Failed to parse JSON from LLM response.")
            return {"parsing_error": "Could not parse JSON", "raw_output": response}
            
        return data

    def _parse_json_list(self, text: str) -> List[str]:
        try:
            text = self._clean_llm_output(text)
            match = re.search(r'\[.*\]', text, re.DOTALL)
            return json.loads(match.group(0)) if match else []
        except Exception:
            return []

    def _parse_json_result(self, text: str) -> Any:
        try:
            text = self._clean_llm_output(text)
            # Priority: List -> Object
            match_list = re.search(r'\[.*\]', text, re.DOTALL)
            if match_list: return json.loads(match_list.group(0))
            
            match_obj = re.search(r'\{.*\}', text, re.DOTALL)
            if match_obj: return json.loads(match_obj.group(0))
            return None
        except Exception as e:
            logger.warning(f"Extraction Parse Error: {e}")
            return None

    def _clean_llm_output(self, text: str) -> str:
        text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```", "", text)
        return text.strip()

    def run(self, **kwargs):
        return self.extract_dynamic(kwargs.get("contract_text", ""), kwargs.get("user_request", ""))