from unsloth import FastLanguageModel
import os
import time
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
import concurrent.futures
from functools import partial

# NOTE: Delaying Unsloth import until AFTER dataset formatting to avoid
# fork-after-CUDA / multiprocessing deadlocks when using datasets.map with num_proc>1.
# We'll import `FastLanguageModel` right before model loading.
FastLanguageModel = None  # type: ignore
import torch
import torch.cuda.profiler as profiler

# Other imports
import psutil
import numpy as np
try:
    from memory_profiler import profile  # type: ignore[import]
except Exception:  # Optional dependency; provide no-op decorator if missing
    def profile(func=None, *args, **kwargs):
        if func is None:
            def _decorator(f):
                return f
            return _decorator
        return func
from sklearn.metrics import precision_recall_fscore_support
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, EarlyStoppingCallback, AutoTokenizer
from trl import SFTTrainer, SFTConfig  # type: ignore[import]
import wandb

# Configure torch to use a different attention mechanism since FA2 is not compatible
os.environ["PYTORCH_ATTENTION_MODE"] = "xformers"  # Fallback to xformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid warnings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataProcessingError(Exception):
    """Raised when there's an error processing the dataset."""
    pass

class ModelConfigurationError(Exception):
    """Raised when there's an error configuring the model."""
    pass

def get_gpu_memory_info() -> Dict[str, float]:
    info = {"available_gb": 0.0, "total_gb": 0.0, "reserved_gb": 0.0, "allocated_gb": 0.0}
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        info["available_gb"] = free_bytes / (1024**3)
        info["total_gb"] = total_bytes / (1024**3)
        info["reserved_gb"] = torch.cuda.memory_reserved(0) / (1024**3)
        info["allocated_gb"] = torch.cuda.memory_allocated(0) / (1024**3)
    return info

def validate_gpu_memory(min_free_gb: float = 2.0) -> None:
    if not torch.cuda.is_available():
        logger.warning("CUDA not available; training will run on CPU if attempted.")
        return
    info = get_gpu_memory_info()
    logger.info(f"GPU memory â€” free: {info['available_gb']:.2f}GB, total: {info['total_gb']:.2f}GB, reserved: {info['reserved_gb']:.2f}GB, allocated: {info['allocated_gb']:.2f}GB")
    if info["available_gb"] < min_free_gb:
        logger.warning(f"Low free GPU memory ({info['available_gb']:.2f}GB). Consider lowering batch size or seq length.")

@dataclass
class ProcessingMetrics:
    """Stores metrics about data processing."""
    total_examples: int
    chunked_examples: int
    max_chunk_size: int
    avg_chunk_size: float
    processing_time: float

@contextmanager
def timer(name):
    start = time.time()
    yield
    end = time.time()
    print(f"{name} took {end - start:.2f} seconds")

def log_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} memory: {torch.cuda.memory_allocated(i) / 1024 / 1024:.2f} MB allocated, "
                  f"{torch.cuda.memory_reserved(i) / 1024 / 1024:.2f} MB reserved")

@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    max_length: int
    overlap_size: int = 100
    min_chunk_size: int = 50
    break_chars: List[str] = None

    def __post_init__(self):
        if self.break_chars is None:
            self.break_chars = ["\n\n", "\n", ". ", "! ", "? "]
        if self.overlap_size >= self.max_length:
            raise ValueError("overlap_size must be less than max_length")

def find_break_point(text: str, max_pos: int, break_chars: List[str]) -> int:
    """Find the best position to break the text, searching backwards from max_pos."""
    for char in break_chars:
        # Search backwards for a clean break
        pos = text.rfind(char, 0, max_pos)
        if pos != -1:
            return pos + len(char)  # Return the position after the break character
    return -1  # Return -1 if no break point is found

def chunk_long_text(
    text: str,
    tokenizer: Any,
    config: ChunkingConfig
) -> Tuple[List[str], ProcessingMetrics]:
    """
    Split long text into overlapping chunks that fit within the model's context window.
    Ultra-fast version that handles ANY size document without failing.
    """
    try:
        start_time = time.time()
        
        # Always use character-based chunking for maximum speed and reliability
        chunks = []
        text_len = len(text)
        
        # If text is very short, return as-is without tokenization
        if text_len < 1000:
            try:
                tokens = tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) <= config.max_length:
                    return [text], ProcessingMetrics(
                        total_examples=1,
                        chunked_examples=1,
                        max_chunk_size=len(tokens),
                        avg_chunk_size=float(len(tokens)),
                        processing_time=time.time() - start_time
                    )
            except:
                # If tokenization fails, just return the text anyway
                return [text], ProcessingMetrics(
                    total_examples=1,
                    chunked_examples=1,
                    max_chunk_size=text_len // 4,  # Rough estimate
                    avg_chunk_size=text_len // 4,
                    processing_time=time.time() - start_time
                )
        
        # Fast character-based chunking for all sizes
        char_per_token = 3.2  # Conservative estimate
        target_tokens = min(config.max_length - 100, 3500)  # Leave buffer
        max_char_chunk = int(target_tokens * char_per_token)
        overlap_chars = min(800, max_char_chunk // 10)  # Small overlap for speed
        
        current_pos = 0
        chunk_count = 0
        max_processing_time = 3.0  # Max 3 seconds per document
        
        while current_pos < text_len:
            # Time check - if processing takes too long, use faster method
            if time.time() - start_time > max_processing_time:
                logger.debug(f"Switching to fast mode for large document ({text_len} chars)")
                # Simple character-based split for the rest
                remaining_text = text[current_pos:]
                for i in range(0, len(remaining_text), max_char_chunk):
                    chunk = remaining_text[i:i + max_char_chunk].strip()
                    if len(chunk) >= config.min_chunk_size:
                        chunks.append(chunk)
                break
            
            # Calculate chunk boundaries
            end_pos = min(current_pos + max_char_chunk, text_len)
            
            # Find natural break points (simple and fast)
            if end_pos < text_len:
                # Look for good break points in order of preference
                for break_str in ["\n\n", "\n", ". ", "! ", "? ", ", ", " "]:
                    search_start = max(current_pos, end_pos - 1000)  # Look in last 1000 chars
                    break_pos = text.rfind(break_str, search_start, end_pos)
                    if break_pos > current_pos:
                        end_pos = break_pos + len(break_str)
                        break
            
            chunk_text = text[current_pos:end_pos].strip()
            
            # Accept any reasonable-sized chunk
            if len(chunk_text) >= config.min_chunk_size:
                chunks.append(chunk_text)
                chunk_count += 1
            
            # Move to next position
            if end_pos >= text_len:
                break
            
            # Calculate next starting position with overlap
            next_pos = max(end_pos - overlap_chars, current_pos + max_char_chunk // 2)
            current_pos = next_pos
        
        # Ensure we always have at least one chunk
        if not chunks:
            # Take first reasonable portion
            chunk_size = min(max_char_chunk, text_len)
            chunk_text = text[:chunk_size].strip()
            if chunk_text:
                chunks.append(chunk_text)
            else:
                # Even if text is weird, take something
                chunks.append(text[:1000] if len(text) > 1000 else text)
        
        # Post-process: estimate token counts efficiently
        chunk_sizes = []
        for chunk in chunks:
            # For very large chunks, estimate tokens; for smaller ones, count exactly
            if len(chunk) > 10000:
                # Use character-based estimate for large chunks to save time
                estimated_tokens = min(int(len(chunk) / char_per_token), config.max_length)
                chunk_sizes.append(estimated_tokens)
            else:
                # Actually tokenize smaller chunks
                try:
                    actual_tokens = len(tokenizer.encode(chunk, add_special_tokens=False))
                    chunk_sizes.append(min(actual_tokens, config.max_length))
                except:
                    # Fallback to estimate if tokenization fails
                    chunk_sizes.append(int(len(chunk) / char_per_token))
        
        # Calculate metrics
        processing_time = time.time() - start_time
        
        metrics = ProcessingMetrics(
            total_examples=1,
            chunked_examples=len(chunks),
            max_chunk_size=max(chunk_sizes) if chunk_sizes else 0,
            avg_chunk_size=float(np.mean(chunk_sizes)) if chunk_sizes else 0.0,
            processing_time=processing_time
        )
        
        # Log slow processing for monitoring
        if processing_time > 2.0:
            logger.debug(f"Slow chunking: {processing_time:.2f}s for {text_len:,} chars -> {len(chunks)} chunks")
        
        return chunks, metrics
    
    except Exception as e:
        # Ultimate fallback - never fail, always return something
        logger.debug(f"Chunking exception, using emergency fallback: {str(e)}")
        try:
            # Simple character-based split as last resort
            max_char_chunk = 12000  # Conservative size
            simple_chunks = []
            for i in range(0, len(text), max_char_chunk):
                chunk = text[i:i + max_char_chunk].strip()
                if chunk:
                    simple_chunks.append(chunk)
            
            if not simple_chunks:
                simple_chunks = [text[:5000] if len(text) > 5000 else text]
            
            # Estimate metrics
            avg_size = sum(len(c) for c in simple_chunks) / len(simple_chunks) / 3.2
            
            return simple_chunks, ProcessingMetrics(
                total_examples=1,
                chunked_examples=len(simple_chunks),
                max_chunk_size=int(max(len(c) for c in simple_chunks) / 3.2),
                avg_chunk_size=avg_size,
                processing_time=time.time() - start_time
            )
        except Exception as final_error:
            # Absolute last resort - return the text as-is
            logger.warning(f"Complete chunking failure, returning text as-is: {str(final_error)}")
            return [text], ProcessingMetrics(
                total_examples=1,
                chunked_examples=1,
                max_chunk_size=len(text) // 4,
                avg_chunk_size=len(text) // 4,
                processing_time=time.time() - start_time
            )

@dataclass
class FormatterConfig:
    """Configuration for data formatting."""
    max_seq_length: int
    chunking_config: ChunkingConfig
    num_workers: int = 4
    batch_size: int = 32
    special_tokens: Dict[str, str] = None

    def __post_init__(self):
        if self.special_tokens is None:
            self.special_tokens = {
                "begin": "<|begin_of_text|>",
                "end": "<|end_of_text|>",
                "header_start": "<|start_header_id|>",
                "header_end": "<|end_header_id|>",
                "eot": "<|eot_id|>"
            }

class DataFormatter:
    """Handles dataset formatting and processing with parallel execution."""
    
    def __init__(self, tokenizer: Any, config: FormatterConfig):
        """
        Initialize the DataFormatter.
        
        Args:
            tokenizer: The tokenizer to use for text processing
            config: FormatterConfig object with formatting parameters
        """
        self.tokenizer = tokenizer
        self.config = config
        self.metrics = []
        # Optional cache for repeated contract texts (enabled via CLI flag)
        self.chunk_cache = {}
        self.enable_cache = False  # Will be set externally
        
    def format_single_example(
        self,
        messages: List[Dict[str, str]]
    ) -> Tuple[List[str], ProcessingMetrics]:
        """
        Format a single example, handling long documents by chunking them appropriately.
        This now correctly handles system, user, and assistant roles.
        """
        try:
            texts = []
            system_content = ""
            user_content = ""
            assistant_content = ""

            # Extract content by role
            for message in messages:
                if not isinstance(message, dict) or "role" not in message or "content" not in message:
                    raise DataProcessingError("Invalid message format")
                
                if message["role"] == "system":
                    system_content = message["content"]
                elif message["role"] == "user":
                    user_content = message["content"]
                elif message["role"] == "assistant":
                    assistant_content = message["content"]
            
            if not user_content or not assistant_content:
                raise DataProcessingError("Missing user or assistant content")

            # Determine which field contains the long contract text to be chunked.
            contract_to_chunk = system_content if system_content else user_content

            # Compute token budget for the static parts (question + answer)
            static_content = user_content + assistant_content
            static_len = len(self.tokenizer.encode(static_content))

            # Create a per-example chunking config without mutating shared state
            safe_margin = 128
            available = max(self.config.max_seq_length - static_len - safe_margin, 256)
            example_chunk_cfg = ChunkingConfig(
                max_length=available,
                overlap_size=self.config.chunking_config.overlap_size,
                min_chunk_size=self.config.chunking_config.min_chunk_size,
                break_chars=self.config.chunking_config.break_chars,
            )

            cache_key = None
            if self.enable_cache and contract_to_chunk:
                # Hash only first 4096 chars to reduce hashing cost but keep uniqueness
                cache_key = f"{len(contract_to_chunk)}:" + str(hash(contract_to_chunk[:4096]))
                if cache_key in self.chunk_cache:
                    contract_chunks, chunk_metrics = self.chunk_cache[cache_key]
                else:
                    contract_chunks, chunk_metrics = chunk_long_text(
                        contract_to_chunk,
                        self.tokenizer,
                        example_chunk_cfg
                    )
                    self.chunk_cache[cache_key] = (contract_chunks, chunk_metrics)
            else:
                contract_chunks, chunk_metrics = chunk_long_text(
                    contract_to_chunk,
                    self.tokenizer,
                    example_chunk_cfg
                )

            # Create formatted examples for each chunk of the contract
            for chunk in contract_chunks:
                text = self.config.special_tokens["begin"]

                # System prompt with the contract chunk
                text += f"{self.config.special_tokens['header_start']}system{self.config.special_tokens['header_end']}\n\n{chunk}{self.config.special_tokens['eot']}"

                # User question
                text += f"{self.config.special_tokens['header_start']}user{self.config.special_tokens['header_end']}\n\n{user_content}{self.config.special_tokens['eot']}"

                # Assistant's answer
                text += f"{self.config.special_tokens['header_start']}assistant{self.config.special_tokens['header_end']}\n\n{assistant_content}{self.config.special_tokens['eot']}"

                texts.append(text)

            return texts, chunk_metrics
            
        except Exception as e:
            raise DataProcessingError(f"Error formatting example: {str(e)}") from e

    def format_batch(
        self,
        examples: Dict[str, List[Any]],
        parallel: bool = True
    ) -> List[str]:
        """
        Format a batch of examples with optional parallel processing.
        
        Args:
            examples: Dictionary containing a list of examples under 'messages' key
            parallel: Whether to use parallel processing for batch formatting
            
        Returns:
            List of formatted text strings
            
        Raises:
            DataProcessingError: If input format is invalid or processing fails
        """
        if not isinstance(examples.get("messages", []), list):
            raise DataProcessingError("Invalid examples format")
            
        try:
            if parallel and len(examples["messages"]) > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
                    # Submit jobs
                    futures = []
                    for message in examples["messages"]:
                        if isinstance(message, (dict, list)):
                            messages_to_process = [message] if isinstance(message, dict) else message
                            futures.append(executor.submit(self.format_single_example, messages_to_process))

                    # Collect results in main thread to avoid shared-state contention
                    texts = []
                    collected_metrics = []
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            batch_texts, metrics = future.result()
                            texts.extend(batch_texts)
                            collected_metrics.append(metrics)
                        except Exception as e:
                            logger.warning(f"Failed to process batch example: {str(e)}")
                    # Single extension of shared metrics to improve thread-safety
                    self.metrics.extend(collected_metrics)
                    return texts
            else:
                # Sequential processing
                texts = []
                messages = examples["messages"]
                if isinstance(messages[0], dict):
                    # Single example with multiple messages
                    formatted_texts, metrics = self.format_single_example(messages)
                    texts.extend(formatted_texts)
                    self.metrics.append(metrics)
                else:
                    # Multiple examples
                    for message_list in messages:
                        formatted_texts, metrics = self.format_single_example(message_list)
                        texts.extend(formatted_texts)
                        self.metrics.append(metrics)
                return texts
                
        except Exception as e:
            raise DataProcessingError(f"Error in batch processing: {str(e)}") from e
            
    def get_metrics_summary(self) -> Dict[str, float]:
        """Get summary statistics of processing metrics."""
        if not self.metrics:
            return {}
            
        return {
            "total_examples": sum(m.total_examples for m in self.metrics),
            "total_chunks": sum(m.chunked_examples for m in self.metrics),
            "avg_chunk_size": np.mean([m.avg_chunk_size for m in self.metrics]),
            "max_chunk_size": max(m.max_chunk_size for m in self.metrics),
            "total_processing_time": sum(m.processing_time for m in self.metrics)
        }

def _extract_assistant_segment(text: str) -> str:
    try:
        start_tag = "<|start_header_id|>assistant<|end_header_id|>"
        end_tag = "<|eot_id|>"
        start = text.rfind(start_tag)
        if start == -1:
            return text
        start += len(start_tag)
        end = text.find(end_tag, start)
        if end == -1:
            end = len(text)
        return text[start:end].strip()
    except Exception:
        return text


NEGATIVE_RESPONSE_VARIANTS = {
    "the provided section does not address this question.",
    "this portion of the contract does not specify that information.",
    "no clause in the supplied text answers this question.",
    "the answer is not present in the quoted contract excerpt.",
    "this clause is not present in the contract.",
}


def compute_metrics(eval_pred, tokenizer):
    """
    Computes precision, recall, and F1-score for clause extraction.
    Handles both logits and generated token predictions.
    """
    predictions, labels = eval_pred

    # Handle different prediction formats: logits (batch, seq, vocab) or token ids (batch, seq)
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    if hasattr(predictions, "ndim") and predictions.ndim == 3:
        predicted_ids = np.argmax(predictions, axis=-1)
    else:
        predicted_ids = predictions

    # Prepare labels: replace -100 with a valid pad id for decoding
    labels_copy = np.array(labels)
    pad_id = tokenizer.pad_token_id if getattr(tokenizer, "pad_token_id", None) is not None else (
        tokenizer.eos_token_id if getattr(tokenizer, "eos_token_id", None) is not None else 0
    )
    labels_copy[labels_copy == -100] = pad_id

    # Decode sequences
    decoded_preds = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels_copy, skip_special_tokens=True)

    # Focus on assistant segments only
    decoded_preds = [_extract_assistant_segment(t) for t in decoded_preds]
    decoded_labels = [_extract_assistant_segment(t) for t in decoded_labels]

    def has_negative_answer(text: Optional[str]) -> int:
        if not text:
            return 0
        lowered = text.strip().lower()
        return 0 if any(variant in lowered for variant in NEGATIVE_RESPONSE_VARIANTS) else 1

    binary_preds = [has_negative_answer(pred) for pred in decoded_preds]
    binary_labels = [has_negative_answer(lab) for lab in decoded_labels]

    precision, recall, f1, _ = precision_recall_fscore_support(
        binary_labels,
        binary_preds,
        average="binary",
        zero_division=0,
    )

    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def check_sequence_lengths(dataset, tokenizer, max_seq_length, data_formatter):
    filtered_dataset = []
    # Process examples in batches for better efficiency
    batch_size = 32
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        
        # Use the data_formatter to get the text
        texts = []
        for example in batch:
            formatted_texts, _ = data_formatter.format_single_example(example["messages"])
            texts.extend(formatted_texts)
        
        # Batch tokenize all texts at once
        batch_tokens = tokenizer(texts, return_length=True)["length"]
        
        for j, (example, tokens) in enumerate(zip(batch, batch_tokens)):
            if tokens > max_seq_length:
                idx = i + j
                print(f"Warning: Example {idx} with {tokens} tokens exceeds max_seq_length={max_seq_length}, truncating...")
                # This part is complex as truncation needs to happen carefully inside the formatted string
                # For now, we just warn and the SFTTrainer will handle truncation.
                # A more robust solution would re-format with truncation.
            filtered_dataset.append(example)
            
    return filtered_dataset


def log_dataset_samples(dataset: Optional[Dataset], num_samples: int = 3) -> None:
    if not dataset:
        return
    logger.info("--- Sample of formatted data being fed to the model ---")
    limit = min(num_samples, len(dataset))
    for i in range(limit):
        record = dataset[i]
        text = record.get("text") if isinstance(record, dict) else None
        logger.info("Example %d:\n%s", i + 1, text)
        logger.info("-" * 40)
    logger.info("--- End of sample ---")

def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    if args.max_seq_length > 8192:
        raise ValueError("max_seq_length cannot exceed 8192")
    if args.per_device_train_batch_size < 1:
        raise ValueError("Batch size must be positive")
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")
    if args.disable_eval:
        return
    if args.validation_dataset_path and not os.path.exists(args.validation_dataset_path):
        raise FileNotFoundError(f"Validation dataset not found: {args.validation_dataset_path}")

@contextmanager
def training_session():
    """Context manager for training session setup and cleanup."""
    try:
        # Setup wandb
        if wandb.run is None:
            wandb.init(project="lexibot-training", config={})
        
        yield
        
    except Exception as e:
        logger.error(f"Training session failed: {str(e)}")
        raise
        
    finally:
        # Cleanup
        if wandb.run is not None:
            wandb.finish()
        torch.cuda.empty_cache()

def setup_wandb(args: argparse.Namespace) -> None:
    """
    Initialize and configure Weights & Biases for experiment tracking.
    
    Args:
        args: Command line arguments containing training configuration
    """
    try:
        # Check if WANDB_API_KEY is already set in environment
        if "WANDB_API_KEY" not in os.environ:
            logger.warning("WANDB_API_KEY not found in environment. Checking for API key file...")
            # Try to read from a key file (common in cloud environments)
            key_file_paths = [
                "/etc/wandb_api_key",  # System-wide location
                os.path.expanduser("~/.wandb/api_key"),  # User's home directory
                "wandb_api_key.txt"  # Current directory
            ]
            
            for key_path in key_file_paths:
                if os.path.exists(key_path):
                    with open(key_path, 'r') as f:
                        os.environ["WANDB_API_KEY"] = f.read().strip()
                    logger.info(f"Found wandb API key in {key_path}")
                    break
        
        if "WANDB_API_KEY" not in os.environ:
            logger.warning("No wandb API key found. Proceeding in offline mode.")
            os.environ["WANDB_MODE"] = "offline"
    
    except Exception as e:
        logger.warning(f"Error setting up wandb API key: {str(e)}. Proceeding in offline mode.")
        os.environ["WANDB_MODE"] = "offline"
    
    # Create a unique run name
    run_name = f"lexibot-{args.model_name.split('/')[-1]}-{time.strftime('%Y%m%d_%H%M%S')}"
    
    # Add GCP-specific metadata if running on Google Cloud
    try:
        import requests
        metadata_server = "http://metadata.google.internal"
        metadata_header = {"Metadata-Flavor": "Google"}
        
        # Get GCP instance information
        instance_name = requests.get(f"{metadata_server}/computeMetadata/v1/instance/name",
                                   headers=metadata_header, timeout=2).text
        zone = requests.get(f"{metadata_server}/computeMetadata/v1/instance/zone",
                          headers=metadata_header, timeout=2).text.split('/')[-1]
        machine_type = requests.get(f"{metadata_server}/computeMetadata/v1/instance/machine-type",
                                  headers=metadata_header, timeout=2).text.split('/')[-1]
        
        # Add GCP metadata to tags
        additional_tags = ["gcp", f"zone-{zone}", f"machine-{machine_type}"]
        logger.info(f"Running on GCP instance: {instance_name} in zone {zone}")
        
    except Exception as e:
        logger.debug(f"Not running on GCP or couldn't fetch metadata: {str(e)}")
        additional_tags = []
    
    # Configure wandb
    wandb_config = {
        # Model configuration
        "model_name": args.model_name,
        "max_seq_length": args.max_seq_length,
        "load_in_4bit": True,
        
        # LoRA configuration
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "use_lora_plus": True,
        
        # Training configuration
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": 0.05,
        "warmup_ratio": 0.03,
        
        # Dataset details
        "train_dataset": args.dataset_path,
        "val_dataset": args.validation_dataset_path,
        
        # Hardware info
        "gpu_count": torch.cuda.device_count(),
        "gpu_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        
        # Advanced training settings
        "gradient_checkpointing": True,
        "torch_compile": True,
        "neftune_noise_alpha": 5,
    }
    
    # Initialize wandb with GCP-specific settings
    try:
        wandb.init(
            project="lexibot-training",
            name=run_name,
            config=wandb_config,
            notes=f"Training run for {args.model_name} on CUAD dataset",
            tags=["qlora", "legal-ai", "contract-analysis"] + additional_tags,
            group="lexibot-development",
            settings={
                "sync_tensorboard": True,  # Sync TensorBoard logs
                "save_code": True,  # Save code snapshot
                "_disable_stats": True,  # Disable system stats on GCP
                "silent": os.environ.get("WANDB_SILENT", "true").lower() == "true",  # Reduce console output
            }
        )
        
        # Log GCP-specific system info if available
        if additional_tags and "gcp" in additional_tags:
            wandb.config.update({
                "environment": "gcp",
                "instance_type": machine_type,
                "zone": zone,
                "instance_name": instance_name
            })
            
        logger.info(f"Successfully initialized wandb run: {wandb.run.name}")
        
    except Exception as e:
        logger.warning(f"Error during wandb initialization: {str(e)}")
        logger.warning("Will continue training without wandb logging")
        os.environ["WANDB_MODE"] = "offline"
    
    # Log system info
    if wandb.run:
        wandb.run.log_code(".")  # Log code snapshot
    
    logger.info(f"Initialized wandb run: {run_name}")

def main(args: argparse.Namespace) -> None:
    """
    Main function to run the Unsloth-optimized QLoRA fine-tuning script.
    """
    try:
        # Validate arguments
        validate_args(args)
        
        # Setup logging
        log_file = Path(args.output_dir) / "training.log"
        file_handler = logging.FileHandler(log_file)
        logger.addHandler(file_handler)
        
        logger.info("Starting training session")
        start_time = time.time()

        # --- Load and format datasets BEFORE model/W&B to avoid multiprocess fork+Cuda issues ---
        logger.info(f"Loading training dataset from: {args.dataset_path}")
        
        # Check for pre-formatted datasets first
        if args.preformatted_train and os.path.exists(args.preformatted_train):
            logger.info(f"Loading pre-formatted training dataset: {args.preformatted_train}")
            train_dataset = load_dataset("json", data_files=args.preformatted_train, split="train")
            logger.info(f"Pre-formatted training dataset size: {len(train_dataset)}")
            raw_train_dataset = None  # Not needed for pre-formatted
        else:
            raw_train_dataset = load_dataset("json", data_files=args.dataset_path, split="train")
            logger.info(f"Initial training dataset size: {len(raw_train_dataset)}")
            train_dataset = None  # Will be formatted later

        eval_dataset = None
        raw_eval_dataset = None
        if not args.disable_eval:
            if args.preformatted_val and os.path.exists(args.preformatted_val):
                logger.info(f"Loading pre-formatted validation dataset: {args.preformatted_val}")
                eval_dataset = load_dataset("json", data_files=args.preformatted_val, split="train")
                logger.info(f"Pre-formatted validation dataset size: {len(eval_dataset)}")
            elif args.validation_dataset_path:
                logger.info(f"Loading validation dataset from: {args.validation_dataset_path}")
                raw_eval_dataset = load_dataset("json", data_files=args.validation_dataset_path, split="train")

        # Only load tokenizer and format if we need to format datasets
        if train_dataset is None or eval_dataset is None:
            # Use a lightweight tokenizer for formatting to avoid loading the full model
            logger.info("Loading tokenizer for formatting only...")
            light_tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, token=os.getenv("HF_TOKEN"))
            if getattr(light_tokenizer, "pad_token", None) is None and getattr(light_tokenizer, "eos_token", None) is not None:
                light_tokenizer.pad_token = light_tokenizer.eos_token
            if hasattr(light_tokenizer, "padding_side"):
                light_tokenizer.padding_side = "right"

        def format_flatten_dataset(raw_ds, max_seq_length: int, tokenizer_for_formatting):
            if raw_ds is None:
                return None
            if "text" in raw_ds.column_names:
                logger.info("Dataset already has 'text' column; skipping formatting stage.")
                return raw_ds

            import multiprocessing
            logger.info(f"Formatting dataset with max_seq_length={max_seq_length}")
            if getattr(args, "format_num_proc", 1) > 1:
                logger.info(
                    "Using multi-process formatting: num_proc=%d, start_method=%s, cpu_count=%s", 
                    getattr(args, "format_num_proc", 1),
                    multiprocessing.get_start_method(allow_none=True) or "default",
                    os.cpu_count(),
                )
            chunking_config = ChunkingConfig(max_length=max_seq_length - 512)
            formatter_config = FormatterConfig(
                max_seq_length=max_seq_length,
                chunking_config=chunking_config
            )
            df = DataFormatter(tokenizer=tokenizer_for_formatting, config=formatter_config)
            if getattr(args, "format_cache", False):
                df.enable_cache = True
                logger.info("Formatting chunk cache ENABLED (--format_cache)")

            # Closure state for progress logging when using datasets.map
            processed_counter = {"n": 0}

            def batch_format(batch):
                messages_list = batch["messages"]
                texts = []
                
                # Batch timing for performance tracking
                batch_start_time = time.time()
                batch_successes = 0
                
                for messages in messages_list:
                    try:
                        formatted_texts, _ = df.format_single_example(messages)
                        texts.extend(formatted_texts)
                        batch_successes += 1
                    except Exception as exc:
                        logger.warning(f"Formatting failure for an example: {exc}")
                        # Continue processing other examples
                
                processed_counter["n"] += len(messages_list)
                
                # Enhanced progress logging with timing and ETA
                if processed_counter["n"] % max(1, getattr(args, "format_log_every", 200)) == 0:
                    batch_time = time.time() - batch_start_time
                    examples_per_sec = len(messages_list) / max(batch_time, 0.001)
                    total_throughput = examples_per_sec * getattr(args, "format_num_proc", 1)
                    remaining_examples = len(raw_ds) - processed_counter["n"]
                    eta_minutes = remaining_examples / max(total_throughput, 1) / 60
                    progress_pct = 100.0 * processed_counter["n"] / len(raw_ds)
                    
                    logger.info(
                        "Formatting progress: %d / %d (%.1f%%) | batch: %dâ†’%d rows in %.2fs (%.1f ex/s) | cache: %d | ETA: %.1f min",
                        processed_counter["n"], len(raw_ds), progress_pct,
                        len(messages_list), len(texts), batch_time, examples_per_sec,
                        len(getattr(df, 'chunk_cache', {})), eta_minutes
                    )
                
                return {"text": texts}

            try:
                formatted = raw_ds.map(
                    batch_format,
                    batched=True,
                    batch_size=getattr(args, "format_batch_size", 32),
                    num_proc=getattr(args, "format_num_proc", 1),
                    remove_columns=raw_ds.column_names,
                    desc=f"Formatting dataset with map() (num_proc={getattr(args, 'format_num_proc', 1)})",
                )
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                if getattr(args, "format_num_proc", 1) > 1:
                    logger.warning(
                        "Multi-process formatting failed (%s). Falling back to single process. (Set --format_num_proc 1 to suppress this.)",
                        str(e),
                    )
                    formatted = raw_ds.map(
                        batch_format,
                        batched=True,
                        batch_size=getattr(args, "format_batch_size", 32),
                        num_proc=1,
                        remove_columns=raw_ds.column_names,
                        desc="Formatting dataset with map() (fallback single process)",
                    )
                else:
                    raise
            return formatted

        # Format training dataset if needed
        if train_dataset is None:
            train_dataset = format_flatten_dataset(raw_train_dataset, args.max_seq_length, light_tokenizer)
            logger.info(f"Final training dataset size (after chunking): {len(train_dataset)}")

        # Format validation dataset if needed
        if not args.disable_eval and eval_dataset is None and args.validation_dataset_path:
            eval_dataset = format_flatten_dataset(raw_eval_dataset, args.max_seq_length, light_tokenizer)
            logger.info(f"Final validation dataset size: {len(eval_dataset)}")

        # Log a few examples of the formatted data
        log_dataset_samples(train_dataset)
        if eval_dataset is not None:
            log_dataset_samples(eval_dataset)

        # Initialize wandb AFTER formatting to avoid wandb-core spawning in workers
        setup_wandb(args)

        # Initial system state and GPU after heavy CPU formatting
        logger.info("Initial system state:")
        log_memory_usage()
        validate_gpu_memory()
        
    # --- Load Model and Tokenizer with Memory Optimizations ---
        with timer("Model loading"):
            logger.info(f"Loading base model: {args.model_name}")
            try:
                # Clear CUDA cache before loading model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                
                logger.info("Configuring model loading parameters... (importing Unsloth now)")
                # Delayed import of Unsloth to avoid CUDA initialization before multiprocessing formatting
                global FastLanguageModel
                if FastLanguageModel is None:
                    from unsloth import FastLanguageModel  # type: ignore
                model_kwargs = {
                    "model_name": args.model_name,
                    "max_seq_length": args.max_seq_length,
                    "load_in_4bit": True,
                    "token": os.getenv("HF_TOKEN"),  # Add HF token if needed
                    "device_map": "auto",
                    "low_cpu_mem_usage": True,
                    "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                }
                
                logger.info("Loading model and tokenizer...")
                model, tokenizer = FastLanguageModel.from_pretrained(**model_kwargs)
                
                # Ensure tokenizer has proper EOS token
                if tokenizer.eos_token is None:
                    logger.warning("Tokenizer missing EOS token, using pad_token or setting to </s>")
                    if tokenizer.pad_token is not None:
                        tokenizer.eos_token = tokenizer.pad_token
                    else:
                        tokenizer.eos_token = "</s>"
                        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
                
                logger.info(f"Tokenizer EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
                
                # Enable gradient checkpointing early
                if hasattr(model, "gradient_checkpointing_enable"):
                    model.gradient_checkpointing_enable()
                
                logger.info("Successfully loaded model and tokenizer")
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                raise ModelConfigurationError(f"Failed to load model: {str(e)}") from e
        
        logger.info("Model loading complete")
        log_memory_usage()

        # Validate GPU memory before proceeding
        validate_gpu_memory()

        # --- Configure QLoRA adapter with optimizations ---
        logger.info("Configuring QLoRA adapter with Unsloth...")
        try:
            # Set dropout to 0 for better Unsloth compatibility
            lora_config = {
                "r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": 0.0,  # Set to 0 for Unsloth optimization
                "bias": "none",
                "use_gradient_checkpointing": True,
                "random_state": 42,
                "target_modules": [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                "modules_to_save": ["lm_head"],
                "fan_in_fan_out": True,
                "init_lora_weights": "gaussian",
                "rank_pattern": {
                    "q_proj": args.lora_r,
                    "k_proj": args.lora_r,
                    "v_proj": args.lora_r,
                    "o_proj": args.lora_r * 2,
                    "gate_proj": args.lora_r // 2,
                    "up_proj": args.lora_r * 2,
                    "down_proj": args.lora_r // 2,
                }
            }

            logger.info("Applying QLoRA configuration...")
            model = FastLanguageModel.get_peft_model(
                model,
                **lora_config
            )

            # Additional memory optimizations
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

            # Clear unused memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Successfully configured QLoRA adapter")
        except Exception as e:
            logger.error(f"Failed to configure QLoRA adapter: {str(e)}")
            raise ModelConfigurationError(f"Failed to configure QLoRA adapter: {str(e)}") from e

        # At this point, datasets are formatted and we can proceed
            
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        raise RuntimeError(f"Training setup failed: {str(e)}") from e
        
    # Rest of function continues...    

    print("Defining training arguments and starting training attempts...")

    def auto_tune_initial_params(bs: int, ga: int, msl: int, enabled: bool) -> Tuple[int, int, int]:
        if not enabled or not torch.cuda.is_available():
            return bs, ga, msl
        info = get_gpu_memory_info()
        free = info.get("available_gb", 0.0)
        tuned_bs, tuned_ga, tuned_msl = bs, ga, msl
        # Heuristics for 8B class models - much more aggressive tuning for speed
        if free >= 20:
            tuned_bs = max(6, bs)  # Very large batch size
            tuned_ga = max(2, ga // 8)  # Minimal gradient accumulation
        elif free >= 15:  # Your case with 15.29GB
            tuned_bs = max(4, bs)  # Batch size of 4
            tuned_ga = max(4, ga // 8)  # Much smaller GA = 4
        elif free >= 12:
            tuned_bs = max(2, bs)
            tuned_ga = max(8, ga // 4)
            tuned_msl = min(msl, 2048)
        else:
            tuned_bs = 1
            tuned_ga = max(16, ga // 2)
            tuned_msl = min(msl, 1536)
        logger.info(f"Auto batch tuning based on {free:.2f}GB free: bs={tuned_bs}, ga={tuned_ga}, msl={tuned_msl}")
        return tuned_bs, tuned_ga, tuned_msl

    def build_deepspeed_config_path():
        # Minimal ZeRO-2 config, optional CPU offload. Persist to JSON and return path.
        if not args.use_deepspeed:
            return None
        import json as _json
        stage = args.deepspeed_stage
        zero = {
            "zero_optimization": {
                "stage": stage,
                # Avoid explicit None in DS config, only include offload when enabled
                **({
                    "offload_param": {"device": "cpu", "pin_memory": True}
                } if args.zero_offload else {})
            },
            "bf16": {"enabled": True},
            "train_micro_batch_size_per_gpu": 1,
        }
        ds_path = Path(args.output_dir) / "deepspeed_zero.json"
        ds_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ds_path, "w") as f:
            _json.dump(zero, f, indent=2)
        logger.info(f"DeepSpeed config written to {ds_path}")
        return str(ds_path)

    def build_training_args(
        output_dir: str,
        num_train_epochs: int,
        per_device_train_batch_size: int,
        per_device_eval_batch_size: int,
        gradient_accumulation_steps: int,
        learning_rate: float,
        max_steps: int,
    ) -> TrainingArguments:
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=0.03,
            learning_rate=learning_rate,
            fp16=False,
            bf16=True,
            logging_steps=args.logging_steps,
            optim="adamw_torch_fused",
            weight_decay=0.05,
            lr_scheduler_type="cosine",
            seed=42,
            eval_strategy="no" if eval_dataset is None else "steps",
            eval_steps=None if eval_dataset is None else 500,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=2,
            load_best_model_at_end=False if eval_dataset is None else True,
            metric_for_best_model=None if eval_dataset is None else "eval_f1",
            greater_is_better=True,
            max_grad_norm=1.0,
            report_to=(["wandb"] if wandb.run else []),
            dataloader_pin_memory=True,
            remove_unused_columns=True,
            gradient_checkpointing=True,
            torch_compile=False,
            ddp_find_unused_parameters=False,
            group_by_length=False,
            hub_model_id=None,
            push_to_hub=False,
            full_determinism=True,
            max_steps=(max_steps if max_steps is not None else -1),
            deepspeed=(build_deepspeed_config_path()),
        )

    class WandbTrainingCallback(EarlyStoppingCallback):
        """Custom callback for detailed wandb logging during training."""

        def __init__(self, early_stopping_patience: int = 3, early_stopping_threshold: float = 0.01):
            super().__init__(early_stopping_patience, early_stopping_threshold)
            self.training_tracker = {
                "step": 0,
                "epoch": 0,
                "best_loss": float("inf"),
                "samples_processed": 0,
            }

        def on_step_end(self, args, state, control, **kwargs):
            """Log detailed metrics after each step."""
            if state.global_step % args.logging_steps == 0:
                optimizer = kwargs.get("optimizer")
                if optimizer and wandb.run:
                    for group_id, group in enumerate(optimizer.param_groups):
                        wandb.log({f"lr/group_{group_id}": group["lr"]}, step=state.global_step)

                if torch.cuda.is_available() and wandb.run:
                    for i in range(torch.cuda.device_count()):
                        wandb.log(
                            {
                                f"gpu_{i}/memory_allocated": torch.cuda.memory_allocated(i) / 1024**2,
                                f"gpu_{i}/memory_reserved": torch.cuda.memory_reserved(i) / 1024**2,
                            },
                            step=state.global_step,
                        )

                if state.log_history and wandb.run:
                    last_log = state.log_history[-1]
                    metrics = {
                        "train/step": state.global_step,
                        "train/epoch": state.epoch,
                        "train/samples_processed": state.global_step
                        * args.per_device_train_batch_size
                        * args.gradient_accumulation_steps,
                        "train/loss": last_log.get("loss", getattr(state, "loss", None)),
                        "train/learning_rate": last_log.get("learning_rate", None),
                        "train/grad_norm": getattr(state, "grad_norm", None),
                    }
                    wandb.log({k: v for k, v in metrics.items() if v is not None}, step=state.global_step)

            return super().on_step_end(args, state, control, **kwargs)

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            """Log evaluation metrics."""
            if metrics and wandb.run:
                eval_metrics = {f"eval/{k}": v for k, v in metrics.items()}
                wandb.log(eval_metrics, step=state.global_step)

                if "eval_loss" in metrics and metrics["eval_loss"] < self.training_tracker["best_loss"]:
                    self.training_tracker["best_loss"] = metrics["eval_loss"]
                    wandb.run.summary["best_eval_loss"] = metrics["eval_loss"]

                    model_artifact = wandb.Artifact(
                        name=f"model-{wandb.run.id}", type="model", description="Best model checkpoint based on eval loss"
                    )
                    model_artifact.add_dir(args.output_dir)
                    wandb.log_artifact(model_artifact)

            return super().on_evaluate(args, state, control, metrics, **kwargs)

    # --- Initialize SFTTrainer ---
    logger.info("Starting training attempts with OOM auto-recovery...")
    compute_metrics_with_tokenizer = partial(compute_metrics, tokenizer=tokenizer)

    # Optionally auto-tune initial params from available GPU memory
    current_bs, current_ga, current_msl = auto_tune_initial_params(
        args.per_device_train_batch_size,
        args.gradient_accumulation_steps,
        args.max_seq_length,
        getattr(args, "auto_batch_tune", False),
    )
    max_ga_limit = 128

    # Cache formatted datasets by max_seq_length to avoid redundant work across retries
    dataset_cache = {}
    dataset_cache[current_msl] = (train_dataset, eval_dataset)

    last_error = None
    for attempt in range(args.max_oom_retries + 1):
        logger.info(f"Attempt {attempt+1}: bs={current_bs}, ga={current_ga}, max_seq_length={current_msl}")
        # If seq length changed, reformat datasets (with caching) - but only if not using pre-formatted
        if current_msl not in dataset_cache:
            if args.preformatted_train and os.path.exists(args.preformatted_train):
                # For pre-formatted datasets, we can't easily reformat, so use original
                logger.warning("Using pre-formatted dataset; cannot adjust max_seq_length for OOM recovery.")
                reformatted_train = load_dataset("json", data_files=args.preformatted_train, split="train")
            else:
                reformatted_train = format_flatten_dataset(raw_train_dataset, current_msl, light_tokenizer)
            
            reformatted_eval = None
            if not args.disable_eval:
                if args.preformatted_val and os.path.exists(args.preformatted_val):
                    reformatted_eval = load_dataset("json", data_files=args.preformatted_val, split="train")
                elif args.validation_dataset_path:
                    reformatted_eval = format_flatten_dataset(raw_eval_dataset, current_msl, light_tokenizer)
            
            dataset_cache[current_msl] = (reformatted_train, reformatted_eval)
        train_dataset, eval_dataset = dataset_cache[current_msl]

        try:
            # Create SFTConfig that replaces TrainingArguments for newer TRL versions
            sft_config = SFTConfig(
                # Training parameters
                output_dir=args.output_dir,
                num_train_epochs=args.num_train_epochs,
                per_device_train_batch_size=current_bs,
                per_device_eval_batch_size=current_bs,
                gradient_accumulation_steps=current_ga,
                warmup_ratio=0.03,
                learning_rate=args.learning_rate,
                fp16=False,
                bf16=True,
                logging_steps=args.logging_steps,
                optim="adamw_torch_fused",
                weight_decay=0.05,
                lr_scheduler_type="cosine",
                seed=42,
                eval_strategy="no" if eval_dataset is None else "steps",
                eval_steps=None if eval_dataset is None else 250,  # More frequent evaluation for better monitoring
                save_strategy="steps",
                save_steps=250,  # Save checkpoints more frequently
                save_total_limit=3,  # Keep more checkpoints for safety
                load_best_model_at_end=False if eval_dataset is None else True,
                metric_for_best_model=None if eval_dataset is None else "eval_f1",
                greater_is_better=True,
                max_grad_norm=1.0,
                report_to=(["wandb"] if wandb.run else []),
                dataloader_pin_memory=True,
                remove_unused_columns=True,
                gradient_checkpointing=True,
                torch_compile=False,
                ddp_find_unused_parameters=False,
                group_by_length=False,
                hub_model_id=None,
                push_to_hub=False,
                full_determinism=True,
                max_steps=(args.max_steps if args.max_steps is not None else -1),
                # SFT-specific parameters
                dataset_num_proc=1,
                packing=False,
                dataset_text_field="text",
            )
            
            trainer = SFTTrainer(
                model=model,
                processing_class=tokenizer,  # Pass the tokenizer object, not its eos_token_id
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                args=sft_config,  # Pass SFTConfig as args, not config
                compute_metrics=compute_metrics_with_tokenizer if eval_dataset is not None else None,
                callbacks=[
                    WandbTrainingCallback(
                        early_stopping_patience=3,
                        early_stopping_threshold=0.01
                    )
                ] if eval_dataset is not None else [],
            )
        except Exception as e:
            logger.error(f"Failed to initialize SFTTrainer: {e}", exc_info=True)
            raise e

        FastLanguageModel.for_training(model)
        try:
            logger.info("Starting training...")
            trainer.train()
            logger.info("Training complete.")
            last_error = None
            break
        except RuntimeError as e:
            msg = str(e).lower()
            if ("out of memory" in msg) or ("cuda" in msg and "oom" in msg):
                logger.warning("CUDA OOM detected. Adjusting hyperparameters and retrying...")
                torch.cuda.empty_cache()
                if current_bs > 1:
                    current_bs = 1
                elif current_ga < max_ga_limit:
                    current_ga = min(max_ga_limit, current_ga * 2)
                elif current_msl > 1024:
                    current_msl = max(1024, current_msl // 2)
                else:
                    last_error = e
                    break
            else:
                last_error = e
                break
        except Exception as e:
            last_error = e
            break

    if last_error is not None:
        logger.error(f"Training failed after retries: {last_error}")
        raise last_error

    # --- Save Final Model ---
    logger.info(f"Saving final adapter model to {args.output_dir}")
    
    # Save model and create wandb artifact
    final_model_dir = Path(args.output_dir) / "final_model"
    final_model_dir.mkdir(exist_ok=True)
    
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    # Create and log final model artifact
    if wandb.run:
        final_artifact = wandb.Artifact(
            name=f"final-model-{wandb.run.id}",
            type="model",
            description="Final trained model with all checkpoints"
        )
        final_artifact.add_dir(str(final_model_dir))
        wandb.log_artifact(final_artifact)
    
    logger.info("Model saved successfully")

    logger.info("Training completed successfully")
    if wandb.run:
        wandb.run.summary["status"] = "completed"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a language model with Unsloth-optimized QLoRA.")
    
    # Model and Data arguments
    # Model and Data arguments
    parser.add_argument("--model_name", type=str, default="unsloth/llama-3.1-8b-bnb-4bit", help="The Unsloth model to fine-tune.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the JSONL training dataset file.")
    parser.add_argument("--validation_dataset_path", type=str, default=None, help="Path to the JSONL validation dataset file (optional).")
    parser.add_argument("--preformatted_train", type=str, default=None, help="Path to pre-formatted training dataset (skips formatting).")
    parser.add_argument("--preformatted_val", type=str, default=None, help="Path to pre-formatted validation dataset (skips formatting).")
    parser.add_argument("--output_dir", type=str, default="./lexibot-1b-adapter", help="Directory to save the fine-tuned model.")  # Updated default
    parser.add_argument("--full_finetune", action="store_true", help="Enable full fine-tuning instead of LoRA.")  # Added
    parser.add_argument("--use_lora_plus", action="store_true", help="Enable LoRA+ for better performance.")  # Added
    
    # Training hyperparameters
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device during training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32, help="Number of updates steps to accumulate.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="The initial learning rate.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")  # More frequent logging
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length to use.")
    parser.add_argument("--auto_batch_tune", action="store_true", help="Auto-tune batch size/GA/seq length based on free GPU memory.")
    parser.add_argument("--max_steps", type=int, default=None, help="Total training steps. Overrides epochs if set.")
    parser.add_argument("--max_oom_retries", type=int, default=3, help="Retries with auto-tuned params on OOM.")
    parser.add_argument("--use_deepspeed", action="store_true", help="Enable DeepSpeed training.")
    parser.add_argument("--deepspeed_stage", type=int, default=2, choices=[1,2,3], help="DeepSpeed ZeRO stage.")
    parser.add_argument("--zero_offload", action="store_true", help="Enable ZeRO CPU offload.")
    parser.add_argument("--format_num_proc", type=int, default=1, help="Parallel processes for datasets.map during formatting.")
    parser.add_argument("--format_batch_size", type=int, default=32, help="Batch size for datasets.map during formatting.")
    parser.add_argument("--format_log_every", type=int, default=200, help="Log a progress message every N base examples during formatting.")
    parser.add_argument("--format_cache", action="store_true", help="Enable caching of chunked contract texts to speed formatting when many QA pairs share identical context.")
    parser.add_argument("--disable_eval", action="store_true", help="Skip validation dataset loading and disable evaluation during training.")

    # LoRA specific arguments
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA attention dimension (more conservative for 8B).")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha (more conservative for 8B).")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout for stability on 8B.")

    args = parser.parse_args()
    main(args)


 #   python3 train.py  --dataset_path ./processed/finetune_dataset.jsonl --validation_dataset_path ./processed/val_finetune_dataset.jsonl --output_dir ./lexibot-1b-adapter-new --lora_dropout 0.1 --learning_rate 1e-5 --num_train_epochs 2 --per_device_train_batch_size 4 --gradient_accumulation_steps 2 --lora_r 8 --lora_alpha 32