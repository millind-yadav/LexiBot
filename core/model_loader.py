from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Minimal model loader for the legacy API. Loads a local Hugging Face model
    from disk so the service can run fully offline.
    """

    def load_model_and_tokenizer(self, model_path: str | Path) -> Tuple[Any, Any]:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

        logger.info("Loading model from %s", model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")

        device_map = "auto" if torch.cuda.is_available() else None
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )

        return model, tokenizer
