import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

try:
    import wandb  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    wandb = None  # type: ignore
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# -----------------------------
# Constants & Helpers
# -----------------------------
NEGATIVE_RESPONSE_VARIANTS = {
    "the provided section does not address this question.",
    "this portion of the contract does not specify that information.",
    "no clause in the supplied text answers this question.",
    "the answer is not present in the quoted contract excerpt.",
    "this clause is not present in the contract.",
}

NEGATIVE_RESPONSE_PATTERNS = (
    re.compile(r"no clause in the supplied text answers this question", re.IGNORECASE),
    re.compile(r"the answer is not present in the (?:quoted )?contract excerpt", re.IGNORECASE),
    re.compile(r"provided section does not address this question", re.IGNORECASE),
    re.compile(r"portion of the contract does not specify", re.IGNORECASE),
    re.compile(r"clause is not present in the contract", re.IGNORECASE),
)

LLAMA3_CHAT_TEMPLATE = """<|begin_of_text|>{% for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"""

LOGGER = logging.getLogger("eval")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


@dataclass
class EvaluationConfig:
    model_path: str
    dataset_path: str
    batch_size: int = 1
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 0.9
    top_k: int = 0
    repetition_penalty: float = 1.1
    load_in_8bit: bool = True
    load_in_4bit: bool = False
    wandb_project: str = "lexibot-evals"
    wandb_run_name: Optional[str] = None
    wandb_group: Optional[str] = None
    disable_wandb: bool = False
    misclassified_output: Optional[str] = None


# -----------------------------
# Core evaluation utilities
# -----------------------------

def _normalize_for_matching(text: str) -> str:
    lowered = text.strip().lower()
    lowered = re.sub(r"[\"“”]", "", lowered)
    return lowered


def _detect_negative_variant(text: str) -> Optional[str]:
    if not text:
        return None
    normalized = _normalize_for_matching(text)
    for variant in NEGATIVE_RESPONSE_VARIANTS:
        if variant in normalized:
            return variant
    for pattern in NEGATIVE_RESPONSE_PATTERNS:
        if pattern.search(normalized):
            return pattern.pattern
    return None


def _has_negative_answer(text: str) -> int:
    return 0 if _detect_negative_variant(text) else 1


def _extract_keyword_hints(text: str) -> List[str]:
    if not text:
        return []
    hints = re.findall(r"\[([^\]]+)\]", text)
    return [hint.strip() for hint in hints if hint.strip()]


def _strip_negative_responses(text: str) -> str:
    cleaned = text
    for variant in NEGATIVE_RESPONSE_VARIANTS:
        cleaned = re.sub(re.escape(variant), "", cleaned, flags=re.IGNORECASE)
    for pattern in NEGATIVE_RESPONSE_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    # Remove common leading markers the model sometimes prepends
    cleaned = re.sub(r"(?i)the answer is not present in the supplied text", "", cleaned)
    cleaned = re.sub(r"(?i)no relevant clause was found", "", cleaned)
    return cleaned


def _has_substantive_content(text: str, min_length: int = 120) -> bool:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) < min_length:
        return False
    return bool(re.search(r"[a-zA-Z0-9]", normalized))


def _has_clause_evidence(
    text: str,
    keyword_hints: Optional[List[str]] = None,
) -> bool:
    if not text:
        return False
    if keyword_hints is None:
        keyword_hints = _extract_keyword_hints(text)
    if keyword_hints:
        return True
    cleaned = _strip_negative_responses(text)
    if not _has_substantive_content(cleaned):
        return False

    bullet_like = bool(re.search(r"(^|\n)\s*[-•\d][\.)]?\s", cleaned))
    clause_markers = bool(re.search(r"\b(section|clause|article|shall|hereby|agreement)\b", cleaned, re.IGNORECASE))
    multi_sentence = cleaned.count(".") >= 2 or cleaned.count("\n") >= 1

    if bullet_like and clause_markers:
        return True
    if clause_markers and multi_sentence:
        return True
    return False


def _strip_assistant_segment(text: str) -> str:
    if not text:
        return ""
    marker = "Answer:"
    if marker in text:
        return text.split(marker, 1)[-1].strip()
    return text.strip()


def _build_quant_config(cfg: EvaluationConfig) -> Optional[BitsAndBytesConfig]:
    if cfg.load_in_4bit:
        return BitsAndBytesConfig(load_in_4bit=True)
    if cfg.load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def ensure_chat_template(tokenizer: AutoTokenizer) -> None:
    if getattr(tokenizer, "chat_template", None):
        return
    LOGGER.warning("Tokenizer missing chat_template; applying default LLaMA 3 template for evaluation")
    tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE


def _build_wandb_table(rows: List[Dict[str, str]]) -> Optional[Any]:
    if not rows or wandb is None:
        return None
    columns = list(rows[0].keys())
    data = [[row.get(col, "") for col in columns] for row in rows]
    return wandb.Table(columns=columns, data=data)


def load_model_and_tokenizer(cfg: EvaluationConfig):
    LOGGER.info("Loading tokenizer from %s", cfg.model_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ensure_chat_template(tokenizer)

    quant_config = _build_quant_config(cfg)
    LOGGER.info("Loading model (quant=%s)...", "8bit" if cfg.load_in_8bit else ("4bit" if cfg.load_in_4bit else "fp16/bf16"))
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=(torch.float16 if not cfg.load_in_4bit else None),
    )
    model.eval()
    LOGGER.info("Model device map: %s", model.device)
    return model, tokenizer


def format_prompts(tokenizer, batch_messages: List[List[Dict[str, str]]]) -> Dict[str, torch.Tensor]:
    prompts: List[str] = []
    for messages in batch_messages:
        prompts.append(
            tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        )
    tokenized = tokenizer(
        prompts,
        padding=True,
        return_tensors="pt",
    )
    return tokenized


def generate_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    inputs: Dict[str, torch.Tensor],
    cfg: EvaluationConfig,
) -> List[str]:
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask.to(model.device)

    do_sample = cfg.temperature > 0 and cfg.top_p < 1.0

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature if do_sample else None,
            top_p=cfg.top_p if do_sample else None,
            top_k=cfg.top_k if do_sample else None,
            do_sample=do_sample,
            repetition_penalty=cfg.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_lengths = attention_mask.sum(dim=1)
    generations: List[str] = []
    for idx in range(outputs.size(0)):
        start = input_lengths[idx].item()
        generated = outputs[idx, start:].detach().cpu()
        decoded = tokenizer.decode(generated, skip_special_tokens=True)
        generations.append(_strip_assistant_segment(decoded))
    return generations


def evaluate(cfg: EvaluationConfig) -> Dict[str, float]:
    model, tokenizer = load_model_and_tokenizer(cfg)

    dataset = load_dataset("json", data_files=cfg.dataset_path, split="train")
    LOGGER.info("Loaded validation dataset: %s (%d examples)", cfg.dataset_path, len(dataset))

    preds: List[int] = []
    labels: List[int] = []

    run = None
    if wandb is None and not cfg.disable_wandb:
        LOGGER.warning("wandb is not installed; disabling wandb logging for this run")
        cfg.disable_wandb = True

    if not cfg.disable_wandb and wandb is not None:
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            group=cfg.wandb_group,
            config={
                "model_path": cfg.model_path,
                "dataset_path": cfg.dataset_path,
                "max_new_tokens": cfg.max_new_tokens,
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "top_k": cfg.top_k,
                "repetition_penalty": cfg.repetition_penalty,
                "load_in_8bit": cfg.load_in_8bit,
                "load_in_4bit": cfg.load_in_4bit,
            },
        )
        run = wandb.run

    table_examples: List[Dict[str, str]] = []
    false_negatives: List[Dict[str, str]] = []
    false_positives: List[Dict[str, str]] = []
    contradiction_examples: List[Dict[str, str]] = []
    misclassified_records: List[Dict[str, str]] = []
    batch_records: List[Dict[str, str]] = []
    batch_messages: List[List[Dict[str, str]]] = []
    contradiction_count = 0

    progress = tqdm(total=len(dataset), desc="Evaluating")

    def process_batch():
        nonlocal batch_records, batch_messages, contradiction_count
        if not batch_records:
            return

        inputs = format_prompts(tokenizer, batch_messages)
        generated_texts = generate_responses(model, tokenizer, inputs, cfg)

        for meta, generated_text in zip(batch_records, generated_texts):
            gold_text = meta["gold_text"]
            keyword_hints = _extract_keyword_hints(generated_text)
            has_clause_evidence = _has_clause_evidence(generated_text, keyword_hints)
            matched_negative_variant = _detect_negative_variant(generated_text)
            pred_label = 0 if matched_negative_variant and not has_clause_evidence else 1
            gold_label = meta["gold_label"]
            gold_negative_variant = meta.get("gold_negative_variant")
            contradiction = bool(matched_negative_variant and has_clause_evidence)

            preds.append(pred_label)
            labels.append(gold_label)

            is_false_negative = gold_label == 1 and pred_label == 0
            is_false_positive = gold_label == 0 and pred_label == 1

            record_base = {
                "user": meta["user_text"],
                "gold": gold_text,
                "prediction": generated_text,
                "gold_label": str(gold_label),
                "prediction_label": str(pred_label),
                "gold_negative_variant": gold_negative_variant or "",
                "matched_negative_variant": matched_negative_variant or "",
                "keyword_hints": ", ".join(keyword_hints),
                "contradiction": str(contradiction),
            }

            if is_false_negative:
                record = dict(record_base)
                if len(false_negatives) < 50:
                    false_negatives.append(record)
                misclassified_records.append({**record, "type": "false_negative"})

            if is_false_positive:
                record = dict(record_base)
                if len(false_positives) < 50:
                    false_positives.append(record)
                misclassified_records.append({**record, "type": "false_positive"})

            if contradiction:
                contradiction_count += 1
                if len(contradiction_examples) < 50:
                    contradiction_examples.append(record_base)

            if run and len(table_examples) < 25:
                table_examples.append(record_base)

        progress.update(len(batch_records))
        batch_records = []
        batch_messages = []

    for record in dataset:
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            progress.update(1)
            continue

        user_turns = messages[:-1]
        gold_message = messages[-1]

        gold_text = gold_message.get("content", "")
        gold_label = _has_negative_answer(gold_text)

        batch_messages.append(user_turns)
        batch_records.append(
            {
                "gold_text": gold_text,
                "gold_label": gold_label,
                "user_text": user_turns[-1].get("content", ""),
                "gold_negative_variant": _detect_negative_variant(gold_text),
            }
        )

        if len(batch_records) >= max(cfg.batch_size, 1):
            process_batch()

    process_batch()

    progress.close()

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(np.mean(np.array(preds) == np.array(labels))),
    }

    confusion = {
        "false_negatives": int(sum(1 for p, g in zip(preds, labels) if p == 0 and g == 1)),
        "false_positives": int(sum(1 for p, g in zip(preds, labels) if p == 1 and g == 0)),
        "true_positives": int(sum(1 for p, g in zip(preds, labels) if p == 1 and g == 1)),
        "true_negatives": int(sum(1 for p, g in zip(preds, labels) if p == 0 and g == 0)),
        "contradictions": contradiction_count,
    }

    LOGGER.info("Evaluation metrics: %s", {**metrics, **confusion})

    if cfg.misclassified_output and misclassified_records:
        output_path = Path(cfg.misclassified_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for record in misclassified_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        LOGGER.info("Wrote %d misclassified examples to %s", len(misclassified_records), output_path)

    if run and wandb is not None:
        wandb.log({**metrics, **confusion})

        samples_table = _build_wandb_table(table_examples)
        if samples_table is not None:
            wandb.log({"samples": samples_table})

        fn_table = _build_wandb_table(false_negatives)
        if fn_table is not None:
            wandb.log({"false_negatives": fn_table})

        fp_table = _build_wandb_table(false_positives)
        if fp_table is not None:
            wandb.log({"false_positives": fp_table})

        contradiction_table = _build_wandb_table(contradiction_examples)
        if contradiction_table is not None:
            wandb.log({"contradictions": contradiction_table})
        wandb.finish()

    return metrics


def parse_args() -> EvaluationConfig:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model on the validation dataset with wandb logging.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model directory.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the validation dataset JSONL file.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation (increase to speed up if memory allows).")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum tokens to generate per sample.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature (0 for greedy).")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for nucleus sampling.")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling parameter.")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty for generation.")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model weights in 8-bit mode (default).")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model weights in 4-bit mode.")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging.")
    parser.add_argument("--wandb_project", type=str, default="lexibot-evals", help="Weights & Biases project name.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases run name.")
    parser.add_argument("--wandb_group", type=str, default=None, help="Weights & Biases group name.")
    parser.add_argument("--misclassified_output", type=str, default=None, help="Optional path to write misclassified examples as JSONL.")

    args = parser.parse_args()

    if args.load_in_4bit:
        args.load_in_8bit = False

    return EvaluationConfig(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
    batch_size=max(1, args.batch_size),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        load_in_8bit=args.load_in_8bit or not args.load_in_4bit,
        load_in_4bit=args.load_in_4bit,
        disable_wandb=args.disable_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_group=args.wandb_group,
        misclassified_output=args.misclassified_output,
    )


def main():
    cfg = parse_args()
    metrics = evaluate(cfg)
    LOGGER.info("Finished evaluation. F1=%.4f", metrics.get("f1", 0.0))


if __name__ == "__main__":
    main()
