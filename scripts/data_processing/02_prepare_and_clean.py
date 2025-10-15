import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Optional

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

from tqdm import tqdm

from transformers import AutoTokenizer

# --- NLTK Downloader (Robust Version) ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("─" * 50)
    print("INFO: Downloading required NLTK tokenizers ('punkt' and 'punkt_tab')...")
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    print("INFO: Download complete.")
    print("─" * 50)

# --- 1. DATASET ANALYSIS FUNCTION ---
def analyze_dataset(squad_data):
    """Analyzes the raw SQuAD-style data and prints a summary."""
    num_contracts = len(squad_data['data'])
    total_qas = 0
    impossible_qas = 0
    context_word_counts = []

    for article in squad_data['data']:
        for paragraph in article['paragraphs']:
            context = ' '.join(paragraph['context'].split())
            context_word_counts.append(len(word_tokenize(context)))
            num_qas_in_para = len(paragraph['qas'])
            total_qas += num_qas_in_para
            for qa in paragraph['qas']:
                if qa['is_impossible']:
                    impossible_qas += 1

    avg_word_count = int(sum(context_word_counts) / len(context_word_counts)) if context_word_counts else 0
    max_word_count = max(context_word_counts) if context_word_counts else 0
    min_word_count = min(context_word_counts) if context_word_counts else 0

    print("\n" + "="*50)
    print("🔍 DATASET ANALYSIS REPORT")
    print("="*50)
    print(f"Total Unique Contracts: {num_contracts}")
    print(f"Total Question/Answer Pairs: {total_qas}")
    print(f"   -> Answerable Questions: {total_qas - impossible_qas}")
    print(f"  -> 'Impossible' Questions: {impossible_qas}")
    print("-" * 50)
    print("📜 Context Length Statistics (in words):")
    print(f"  -> Average Contract Length: {avg_word_count} words")
    print(f"  -> Longest Contract: {max_word_count} words")
    print(f"  -> Shortest Contract: {min_word_count} words")
    print("="*50 + "\n")


# --- 2. HELPER AND FORMATTING FUNCTIONS (Unchanged) ---
def build_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    return tokenizer


def answer_aware_chunking(context: str,
                          tokenizer: AutoTokenizer,
                          max_tokens: int,
                          answer_span: tuple[int, int] | None = None) -> str | None:
    tokenized = tokenizer(
        context,
        return_offsets_mapping=True,
        add_special_tokens=False
    )

    input_ids = tokenized["input_ids"]
    offsets = tokenized["offset_mapping"]

    if len(input_ids) <= max_tokens:
        return context

    if answer_span is not None:
        answer_start, answer_end = answer_span
        answer_token_index = None
        for idx, (start, end) in enumerate(offsets):
            if start <= answer_start < end:
                answer_token_index = idx
                break

        if answer_token_index is None:
            return None

        half_window = max_tokens // 2
        start_idx = max(0, answer_token_index - half_window)
        end_idx = min(len(input_ids), start_idx + max_tokens)

        if end_idx - start_idx < max_tokens:
            start_idx = max(0, end_idx - max_tokens)
    else:
        start_idx = 0
        end_idx = max_tokens

    selected_offsets = offsets[start_idx:end_idx]
    if not selected_offsets:
        return None

    char_start = selected_offsets[0][0]
    char_end = selected_offsets[-1][1]
    return context[char_start:char_end]


NEGATIVE_RESPONSE = "Answer not present in the excerpt."


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


SentenceSpan = tuple[int, int, str]


def _sentence_spans(text: str) -> list[SentenceSpan]:
    spans: list[SentenceSpan] = []
    start = 0
    for sent in sent_tokenize(text):
        idx = text.find(sent, start)
        if idx == -1:
            continue
        spans.append((idx, idx + len(sent), sent))
        start = idx + len(sent)
    return spans


def expand_answer_to_sentences(context: str, answer_span: tuple[int, int], min_chars: int = 60) -> tuple[str, tuple[int, int]] | None:
    answer_start, answer_end = answer_span
    if answer_start < 0 or answer_end <= answer_start:
        return None

    spans = _sentence_spans(context)
    if not spans:
        return None

    selected_sentences: list[str] = []
    span_start = None
    span_end = None

    for idx, (sent_start, sent_end, sentence) in enumerate(spans):
        if sent_start <= answer_start < sent_end:
            selected_sentences.append(sentence)
            span_start = sent_start
            span_end = sent_end

            # pull in previous sentence if too short
            if len(" ".join(selected_sentences)) < min_chars and idx > 0:
                prev_start, prev_end, prev_sentence = spans[idx - 1]
                selected_sentences.insert(0, prev_sentence)
                span_start = prev_start

            # pull in next sentence if still short
            if len(" ".join(selected_sentences)) < min_chars and idx + 1 < len(spans):
                next_start, next_end, next_sentence = spans[idx + 1]
                selected_sentences.append(next_sentence)
                span_end = next_end

            break

    if not selected_sentences or span_start is None or span_end is None:
        return None

    expanded = " ".join(selected_sentences).strip()
    return expanded, (span_start, span_end)


def _flatten_keywords(question: str, question_type: str) -> list[str]:
    groups = _keyword_groups_for_question(question, question_type)
    flattened: set[str] = set()
    for group in groups:
        for keyword in group:
            flattened.add(keyword.lower())
    return sorted(flattened)


def build_positive_context(context: str,
                           answer_span: tuple[int, int],
                           min_chars: int = 120,
                           max_chars: int = 800,
                           spans: Optional[list[SentenceSpan]] = None) -> tuple[str, tuple[int, int]] | None:
    spans = spans if spans is not None else _sentence_spans(context)
    if not spans:
        return None

    answer_start, _ = answer_span
    sentence_index = None
    for idx, (sent_start, sent_end, _) in enumerate(spans):
        if sent_start <= answer_start < sent_end:
            sentence_index = idx
            break

    if sentence_index is None:
        return None

    start_idx = sentence_index
    end_idx = sentence_index
    pieces = [spans[sentence_index][2].strip()]

    def _length() -> int:
        return len(" ".join(pieces))

    while (_length() < min_chars or _length() > max_chars) and (start_idx > 0 or end_idx < len(spans) - 1):
        if _length() < min_chars and start_idx > 0:
            start_idx -= 1
            pieces.insert(0, spans[start_idx][2].strip())
        if _length() < min_chars and end_idx < len(spans) - 1:
            end_idx += 1
            pieces.append(spans[end_idx][2].strip())

        while _length() > max_chars and len(pieces) > 1:
            if len(pieces[0]) >= len(pieces[-1]):
                pieces.pop(0)
                start_idx += 1
            else:
                pieces.pop()
                end_idx -= 1

    excerpt = normalize_whitespace(" ".join(pieces))
    span_start = spans[start_idx][0]
    span_end = spans[end_idx][1]
    return excerpt, (span_start, span_end)


def build_negative_context(context: str,
                           question: str,
                           question_type: str,
                           max_sentences: int = 4,
                           max_chars: int = 800,
                           spans: Optional[list[SentenceSpan]] = None) -> str | None:
    spans = spans if spans is not None else _sentence_spans(context)
    if not spans:
        return None

    keywords = _flatten_keywords(question, question_type)
    selected: list[str] = []

    if keywords:
        for _, _, sentence in spans:
            sentence_clean = sentence.strip()
            lower_sentence = sentence_clean.lower()
            if any(keyword in lower_sentence for keyword in keywords):
                selected.append(sentence_clean)
            if len(selected) >= max_sentences:
                break

    if not selected:
        selected = [sentence.strip() for _, _, sentence in spans[:max_sentences]]

    if not selected:
        return None

    excerpt = normalize_whitespace(" ".join(selected))
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars].rsplit(" ", 1)[0].strip()
    return excerpt or None


def _keyword_groups_for_question(question: str, question_type: str) -> list[list[str]]:
    question_lower = question.lower()

    groups: list[list[str]] = []

    type_based = {
        "governing_law": [["law"], ["govern"], ["jurisdiction"], ["state"], ["country"]],
        "termination": [["terminate"], ["termination"], ["cancel"]],
        "payment": [["pay"], ["payment"], ["fee"], ["compensation"], ["revenue"], ["royalt"], ["amount"]],
        "liability": [["liability"], ["indemn"]],
        "confidentiality": [["confidential"], ["non-disclosure"], ["nda"]],
        "non_compete": [["non-compete"], ["exclusive"], ["exclusivity"], ["solicit"], ["competition"], ["restrict"]],
    }

    groups.extend(type_based.get(question_type, []))

    if "license" in question_lower:
        groups.append(["license"])
        groups.append(["licensor", "licensee"])
    if "royalt" in question_lower or "profit" in question_lower or "revenue" in question_lower:
        groups.append(["royalt", "profit", "revenue", "share", "percentage"])
    if "parties" in question_lower:
        groups.append(["inc.", "llc", "ltd", "company", "corporation", "limited", "parties"])
    if "change of control" in question_lower:
        groups.append(["change of control", "merger", "acquisition", "transfer", "assign"])
    if "anti-assignment" in question_lower or "assignment" in question_lower:
        groups.append(["assign", "assignment", "transfer"])

    return groups


def _answer_satisfies_keywords(answer: str, question: str, question_type: str) -> bool:
    groups = _keyword_groups_for_question(question, question_type)
    if not groups:
        return True

    answer_lower = answer.lower()
    for group in groups:
        if any(keyword in answer_lower for keyword in group):
            return True
    return False


def format_chatml(context: str, question: str, answer: str, metadata: dict) -> dict:
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are an expert contract analyst. Answer using only the provided context."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}"
            },
            {
                "role": "assistant",
                "content": answer
            }
        ],
        "metadata": metadata
    }

# --- 3. MAIN PROCESSING FUNCTION (With tqdm Progress Bar) ---
def guess_question_type(question: str) -> str:
    lowered = question.lower()
    if "termination" in lowered:
        return "termination"
    if "payment" in lowered or "compensation" in lowered or "$" in question:
        return "payment"
    if "governing law" in lowered or "jurisdiction" in lowered:
        return "governing_law"
    if "liability" in lowered or "indemn" in lowered:
        return "liability"
    if "confidential" in lowered or "non-disclosure" in lowered:
        return "confidentiality"
    if "non-compete" in lowered or "no-solicit" in lowered:
        return "non_compete"
    return "other"


def _process_contract(
    idx: int,
    article: dict,
    max_examples_per_contract: int,
    max_tokens: int,
    min_answer_chars: int
) -> tuple[list[dict], list[dict], Counter]:
    contract_id = article.get('title') or f"contract_{idx}"

    stats = Counter()
    positive_examples: list[dict] = []
    negative_examples: list[dict] = []

    if not article['paragraphs']:
        stats['contracts_without_paragraphs'] += 1
        return positive_examples, negative_examples, stats

    original_context = ' '.join(article['paragraphs'][0]['context'].split())
    qas = article['paragraphs'][0]['qas']
    sentence_spans = _sentence_spans(original_context)

    used_examples = 0

    for qa in qas:
        if used_examples >= max_examples_per_contract:
            stats['skipped_contract_limit'] += 1
            continue

        question = ' '.join(qa['question'].split())
        question_type = guess_question_type(question)
        qa_id = qa.get('id') or f"{contract_id}_qa_{used_examples}"

        metadata = {
            "contract_id": contract_id,
            "question_id": qa_id,
            "question_type": question_type,
            "is_impossible": bool(qa.get('is_impossible', False))
        }

        if qa.get('is_impossible', False):
            context_excerpt = build_negative_context(
                original_context,
                question,
                question_type,
                max_chars=max_tokens // 2,
                spans=sentence_spans
            )
            if not context_excerpt:
                stats['skipped_no_chunk'] += 1
                continue

            example = format_chatml(context_excerpt, question, NEGATIVE_RESPONSE, metadata)
            negative_examples.append(example)
            used_examples += 1
            stats['negatives_kept'] += 1
            continue

        answers = qa.get('answers', [])
        if not answers:
            stats['skipped_no_answers'] += 1
            continue

        ans = answers[0]
        raw_answer_text = normalize_whitespace(ans.get('text', ''))
        if len(raw_answer_text) < min_answer_chars:
            stats['skipped_short_answer'] += 1
            continue

        answer_start = ans.get('answer_start', -1)
        if answer_start < 0:
            stats['skipped_missing_answer_start'] += 1
            continue

        answer_end = answer_start + len(ans.get('text', ''))

        positive_context = build_positive_context(
            original_context,
            (answer_start, answer_end),
            min_chars=min_answer_chars,
            max_chars=max_tokens,
            spans=sentence_spans
        )
        if positive_context is None:
            stats['skipped_sentence_expand'] += 1
            continue

        context_excerpt, _ = positive_context
        token_count = len(word_tokenize(raw_answer_text))

        MIN_ANSWER_TOKENS = 15
        SHORT_WHITELIST = ("royalt", "payment", "license", "parties", "anti-assign", "assignment")
        lowered_answer = raw_answer_text.lower()
        if token_count < MIN_ANSWER_TOKENS and not any(token in lowered_answer for token in SHORT_WHITELIST):
            stats['skipped_short_answer'] += 1
            continue

        if not _answer_satisfies_keywords(raw_answer_text, question, question_type):
            stats['skipped_answer_keyword_mismatch'] += 1
            continue

        context_excerpt = normalize_whitespace(context_excerpt)
        answer_lower = raw_answer_text.lower()
        context_lower = context_excerpt.lower()

        if answer_lower not in context_lower:
            stats['skipped_answer_outside_chunk'] += 1
            continue

        relative_start = context_lower.find(answer_lower)
        metadata = {
            **metadata,
            "answer_length_chars": len(raw_answer_text),
            "answer_tokens": token_count,
            "answer_start": relative_start if relative_start >= 0 else None,
            "context_length_chars": len(context_excerpt)
        }

        example = format_chatml(context_excerpt, question, raw_answer_text, metadata)
        positive_examples.append(example)
        used_examples += 1
        stats['positives_kept'] += 1

    return positive_examples, negative_examples, stats


def process_to_chat_format(
    input_filepath: Path,
    output_filepath: Path,
    tokenizer: AutoTokenizer,
    max_tokens: int,
    max_examples_per_contract: int,
    max_negative_ratio: float,
    min_answer_chars: int,
    seed: int,
    val_output: Optional[Path],
    val_fraction: float,
    workers: int
):
    print("✅ Stage 1: Loading raw dataset file...")
    with open(input_filepath, 'r', encoding='utf-8') as f:
        squad_data = json.load(f)
    print(f"  -> Successfully loaded '{input_filepath}'.")

    print("\n✅ Stage 2: Analyzing dataset structure...")
    analyze_dataset(squad_data)

    print("✅ Stage 3: Processing and converting data...")
    start_time = time.time()
    rng = random.Random(seed)

    positive_examples: list[dict] = []
    negative_examples: list[dict] = []

    stats = Counter()

    articles = list(enumerate(squad_data['data']))
    rng.shuffle(articles)  # Shuffle contracts so progress bar advances early

    if workers <= 1:
        iterator = tqdm(
            (
                _process_contract(idx, article, max_examples_per_contract, max_tokens, min_answer_chars)
                for idx, article in articles
            ),
            total=len(articles),
            desc="Processing contracts",
            dynamic_ncols=True,
            file=sys.stdout,
        )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            mapped = executor.map(
                _process_contract,
                (idx for idx, _ in articles),
                (article for _, article in articles),
                repeat(max_examples_per_contract),
                repeat(max_tokens),
                repeat(min_answer_chars),
            )

            iterator = tqdm(
                mapped,
                total=len(articles),
                desc="Processing contracts",
                dynamic_ncols=True,
                file=sys.stdout,
            )

    for positives, negatives, contract_stats in iterator:
        positive_examples.extend(positives)
        negative_examples.extend(negatives)
        stats.update(contract_stats)

    end_time = time.time()
    print(f"\n  -> Processing complete. Took {end_time - start_time:.2f} seconds.")
    print(f"  -> Positive examples collected: {len(positive_examples)}")
    print(f"  -> Negative examples collected: {len(negative_examples)}")

    target_negatives = math.floor(len(positive_examples) * max_negative_ratio)
    if target_negatives and len(negative_examples) > target_negatives:
        rng.shuffle(negative_examples)
        negative_examples = negative_examples[:target_negatives]
        stats['negatives_downsampled'] = 1

    combined = positive_examples + negative_examples
    rng.shuffle(combined)

    def write_jsonl(records: list[dict], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            for item in records:
                f.write(json.dumps(item) + '\n')

    val_examples: list[dict] = []
    train_examples: list[dict] = combined

    if val_output and val_fraction > 0 and len(combined) >= 2:
        clamped_fraction = min(max(val_fraction, 0.0), 0.5)
        val_count = max(1, int(round(len(combined) * clamped_fraction)))
        if val_count >= len(combined):
            val_count = len(combined) - 1
        val_examples = combined[:val_count]
        train_examples = combined[val_count:]

    print(f"\n✅ Stage 4: Saving formatted dataset to '{output_filepath}'...")
    write_jsonl(train_examples, output_filepath)

    if val_output and val_examples:
        print(f"  -> Writing validation split to '{val_output}' ({len(val_examples)} examples)")
        write_jsonl(val_examples, val_output)

    print("\n" + "="*50)
    print("🎉 All Done! 🎉")
    print(f"  -> Total examples created: {len(combined)}")
    print(f"  -> Train examples: {len(train_examples)} (saved to {output_filepath})")
    if val_output and val_examples:
        print(f"  -> Validation examples: {len(val_examples)} (saved to {val_output})")
    print("  -> Positive examples: {}/{}".format(len(positive_examples), len(combined)))
    print("  -> Negative examples: {}/{}".format(len(negative_examples), len(combined)))
    print("  -> Skip stats: {}".format(dict(stats)))
    print("="*50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CUAD-style QA data into ChatML format with quality filters.")
    parser.add_argument("--input", required=True, type=Path, help="Path to the raw CUAD dataset (JSON).")
    parser.add_argument("--output", required=True, type=Path, help="Where to write the processed training JSONL dataset.")
    parser.add_argument("--val_output", type=Path, default=None, help="Optional path to write a validation JSONL dataset.")
    parser.add_argument("--val_fraction", type=float, default=0.1, help="Fraction of examples to reserve for validation when --val_output is supplied (default: 0.1, clamped to [0, 0.5]).")
    parser.add_argument("--tokenizer", default="unsloth/Llama-3.2-3B-bnb-4bit", help="Tokenizer/model name for token counting.")
    parser.add_argument("--max_tokens", type=int, default=1600, help="Maximum tokens per context chunk (default: 1600).")
    parser.add_argument("--max_examples_per_contract", type=int, default=25, help="Cap on total examples kept per contract.")
    parser.add_argument("--max_negative_ratio", type=float, default=0.3, help="Maximum negatives as a fraction of positives (default: 0.3).")
    parser.add_argument("--min_answer_chars", type=int, default=60, help="Minimum length for positive answers before expansion.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes to run in parallel (default: 1).")
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = build_tokenizer(args.tokenizer)

    process_to_chat_format(
        input_filepath=args.input,
        output_filepath=args.output,
        tokenizer=tokenizer,
        max_tokens=args.max_tokens,
        max_examples_per_contract=args.max_examples_per_contract,
        max_negative_ratio=args.max_negative_ratio,
        min_answer_chars=args.min_answer_chars,
        seed=args.seed,
        val_output=args.val_output,
        val_fraction=args.val_fraction,
        workers=args.workers
    )


if __name__ == "__main__":
    main()


