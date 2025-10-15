import unsloth
import os
import json
import logging
import argparse
import time
from pathlib import Path
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
from peft import PeftModel
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
import wandb
from dataclasses import dataclass
import concurrent.futures

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Stores model evaluation metrics."""
    precision: float
    recall: float
    f1: float
    accuracy: float
    confusion_matrix: np.ndarray
    evaluation_time: float
    examples_processed: int
    avg_inference_time: float

def format_prompt(messages: list[dict], tokenizer: any, max_seq_length: int) -> str:
    """
    Formats a list of messages, including a system message, into a single
    Llama-style prompt string. Excludes the final assistant message and truncates
    the system context to fit within max_seq_length.
    """
    prompt_str = ""
    system_message = next((msg['content'] for msg in messages if msg['role'] == 'system'), None)
    conversation = [msg for msg in messages if msg['role'] in ('user', 'assistant')]

    # Estimate token usage for the non-system-message parts of the prompt
    # (BOS, INST, SYS tags, user query, etc.) to reserve space.
    # This is a rough estimate; a more precise method would tokenize each part.
    reserved_space = 256 
    for msg in conversation:
        if msg['role'] == 'user':
            # Add estimate for user message and instruction tags
            reserved_space += len(msg['content']) // 2 # Rough char-to-token
    
    max_context_len = max_seq_length - reserved_space

    # Truncate the system message if it's too long
    if system_message and len(system_message) > max_context_len:
        system_message = system_message[:max_context_len]
        logger.debug("Truncated system message to fit max_seq_length.")

    # Start with the system prompt if it exists
    if system_message:
        prompt_str += f"{tokenizer.bos_token}[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n"
    else:
        prompt_str += f"{tokenizer.bos_token}[INST] "

    # Process conversation, excluding the last assistant message (the label)
    for i, msg in enumerate(conversation):
        if msg['role'] == 'user':
            if system_message or i > 0:
                 prompt_str += f"[INST] {msg['content']} [/INST]"
            else:
                 prompt_str += f"{msg['content']} [/INST]"
        elif msg['role'] == 'assistant':
            if i < len(conversation) - 1:
                prompt_str += f" {msg['content']}{tokenizer.eos_token}"
                
    return prompt_str

def is_negative_response(text: str) -> bool:
    """
    Checks if the model's output or a label should be considered a negative response.
    A response is considered negative if it's empty, "no", or indicates the answer isn't in the text.
    """
    if not text:
        return True
    
    lower_text = text.strip().lower()
    
    if lower_text == "no":
        return True
    
    # Check for phrases indicating the information was not found
    if "cannot be found" in lower_text or "not found" in lower_text:
        return True
        
    return False

def evaluate_model(
    model: any,
    tokenizer: any,
    test_dataset: any,
    batch_size: int = 8,
    max_new_tokens: int = 150,
) -> EvaluationMetrics:
    """
    Evaluate the model on a test dataset with comprehensive metrics.
    """
    start_time = time.time()
    model.eval()
    predictions, labels = [], []
    total_inference_time = 0
    
    with torch.no_grad():
        num_batches = (len(test_dataset) + batch_size - 1) // batch_size
        for i in range(0, len(test_dataset), batch_size):
            logger.info(f"Processing batch {i//batch_size + 1}/{num_batches}...")
            batch = test_dataset[i:min(i + batch_size, len(test_dataset))]
            
            batch_prompts = []
            batch_labels = []

            # The batch is a dictionary of lists, e.g., {'messages': [...], 'label': [...]}
            num_items_in_batch = len(next(iter(batch.values())))

            for j in range(num_items_in_batch):
                try:
                    messages = batch["messages"][j]
                    
                    # --- CORRECTED LOGIC FOR EXTRACTIVE QA ---
                    # 1. Determine Ground Truth Label
                    assistant_message = next((msg['content'] for msg in messages if msg['role'] == 'assistant'), "")
                    
                    # A negative response (label 0) indicates the answer is not in the contract.
                    # A positive response (label 1) is an extraction.
                    label = 0 if is_negative_response(assistant_message) else 1
                    
                    # 2. Format prompt and add to batch
                    prompt = format_prompt(messages, tokenizer)
                    if not prompt:
                        logger.warning(f"Skipping item with no user messages at index {j} in batch")
                        continue
                        
                    batch_prompts.append(prompt)
                    batch_labels.append(label)
                    # --- END CORRECTED LOGIC ---

                except (KeyError, IndexError, TypeError) as e:
                    logger.warning(f"Skipping malformed item at index {j} in batch, error: {e}")
                    continue

            if not batch_prompts:
                continue
            
            # Tokenize inputs
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=model.config.max_position_embeddings
            ).to(model.device)
            
            # Generate predictions
            inference_start = time.time()
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
            total_inference_time += time.time() - inference_start
            
            # Decode and process outputs
            pred_texts = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Log first batch for debugging
            if i == 0:
                for k in range(min(len(batch_prompts), 5)): # Log up to 5 examples
                    logger.info(f"--- DEBUG: Batch 0, Example {k} ---")
                    logger.info(f"PROMPT:\n{batch_prompts[k]}")
                    logger.info(f"GENERATED:\n{pred_texts[k]}")
                    logger.info(f"GROUND TRUTH LABEL: {batch_labels[k]}")
                    logger.info(f"------------------------------------")

            # --- CORRECTED PREDICTION LOGIC ---
            # Use the same logic as for labels to classify predictions.
            batch_preds = [0 if is_negative_response(pred) else 1 for pred in pred_texts]
            # --- END CORRECTED PREDICTION LOGIC ---
            
            predictions.extend(batch_preds)
            labels.extend(batch_labels)

    if not labels:
        raise ValueError("No valid examples found in the test dataset to evaluate.")

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    cm = confusion_matrix(labels, predictions)
    accuracy = accuracy_score(labels, predictions)
    
    evaluation_time = time.time() - start_time
    avg_inference_time = total_inference_time / len(test_dataset) if len(test_dataset) > 0 else 0
    
    return EvaluationMetrics(
        precision=precision, recall=recall, f1=f1, accuracy=accuracy,
        confusion_matrix=cm, evaluation_time=evaluation_time,
        examples_processed=len(test_dataset), avg_inference_time=avg_inference_time
    )

def main(args: argparse.Namespace):
    """Main function to run the evaluation."""
    try:
        # Initialize wandb
        wandb.init(project="lexibot-evaluation", config=vars(args), name=f"eval-{Path(args.model_path).name}-{time.strftime('%Y%m%d_%H%M')}")
        
        logger.info(f"Loading base model: {args.base_model_name}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.base_model_name,
            max_seq_length=args.max_seq_length,
            load_in_4bit=True,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
        
        logger.info(f"Loading adapter from: {args.model_path}")
        model = PeftModel.from_pretrained(model, args.model_path)

        logger.info(f"Loading test dataset from: {args.test_dataset_path}")
        test_dataset = load_dataset("json", data_files=args.test_dataset_path, split="train")

        logger.info("Starting evaluation...")
        global start_time
        start_time = time.time()
        metrics = evaluate_model(model, tokenizer, test_dataset, batch_size=args.batch_size)
        
        # Log metrics
        logger.info("--- Evaluation Results ---")
        logger.info(f"Precision: {metrics.precision:.4f}")
        logger.info(f"Recall: {metrics.recall:.4f}")
        logger.info(f"F1 Score: {metrics.f1:.4f}")
        logger.info(f"Accuracy: {metrics.accuracy:.4f}")
        logger.info(f"Confusion Matrix:\n{metrics.confusion_matrix}")
        logger.info(f"Total evaluation time: {metrics.evaluation_time:.2f}s")
        
        # Log to wandb
        wandb.log({
            "eval/precision": metrics.precision,
            "eval/recall": metrics.recall,
            "eval/f1": metrics.f1,
            "eval/accuracy": metrics.accuracy,
            "eval/avg_inference_time": metrics.avg_inference_time,
            "eval/confusion_matrix": wandb.plot.confusion_matrix(
                y_true=metrics.confusion_matrix.flatten(), preds=metrics.confusion_matrix.flatten()
            )
        })
        
        logger.info("Evaluation completed successfully.")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
    finally:
        if wandb.run:
            wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned language model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned adapter model directory.")
    parser.add_argument("--base_model_name", type=str, default="unsloth/llama-3.2-1b-bnb-4bit", help="The base model to load before applying the adapter.")
    parser.add_argument("--test_dataset_path", type=str, required=True, help="Path to the JSONL test dataset file.")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Maximum sequence length.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation.")
    
    args = parser.parse_args()
    main(args)
