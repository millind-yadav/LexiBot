#!/usr/bin/env python3
"""
CUAD-Specific Smart Preprocessor - FINAL VERSION
Handles answer-aware chunking for legal contracts with proper validation
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datasets import load_dataset, Dataset
from tqdm import tqdm
import re
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MAX_CONTEXT_CHARS = 8000  # ~2000 tokens - safe for most models
MIN_CONTEXT_CHARS = 100   # Minimum viable context


def extend_to_sentence_boundary(text: str, start: int, end: int) -> tuple:
    """
    Extend boundaries to complete sentences, avoiding mid-word cuts.
    """
    # Sentence endings
    sentence_endings = ['. ', '.\n', '! ', '?\n', '? ']
    
    # Extend start backward to sentence beginning
    if start > 0:
        # Look back up to 200 chars for sentence start
        search_start = max(0, start - 200)
        for ending in sentence_endings:
            pos = text.rfind(ending, search_start, start)
            if pos != -1:
                start = pos + len(ending)
                break
    
    # Extend end forward to sentence ending
    if end < len(text):
        # Look forward up to 200 chars for sentence end
        search_end = min(len(text), end + 200)
        for ending in sentence_endings:
            pos = text.find(ending, end, search_end)
            if pos != -1:
                end = pos + len(ending)
                break
    
    return start, end


def extract_section(context: str, answer_start: int, answer_length: int, 
                   context_window: int = 2000) -> Optional[str]:
    """
    Extract a section around the answer, ensuring complete sentences.
    Returns None if extraction fails.
    """
    if not context or answer_start < 0:
        return None
    
    # Calculate initial boundaries
    start = max(0, answer_start - context_window)
    end = min(len(context), answer_start + answer_length + context_window)
    
    # Extend to sentence boundaries
    start, end = extend_to_sentence_boundary(context, start, end)
    
    section = context[start:end].strip()
    
    # Validate section
    if len(section) < MIN_CONTEXT_CHARS:
        logger.warning(f"Section too short: {len(section)} chars")
        return None
    
    # If still too long, truncate at sentence boundary
    if len(section) > MAX_CONTEXT_CHARS:
        # Find last sentence ending within limit
        truncate_pos = MAX_CONTEXT_CHARS
        for ending in ['. ', '.\n', '! ', '? ']:
            pos = section.rfind(ending, 0, MAX_CONTEXT_CHARS)
            if pos != -1:
                truncate_pos = pos + len(ending)
                break
        section = section[:truncate_pos].strip()
    
    return section


def process_cuad_example(example: Dict[str, Any], idx: int, 
                        negative_ratio: float = 1.0, 
                        include_metadata: bool = False,
                        max_examples_per_contract: int = None) -> List[Dict[str, Any]]:
    """
    Process CUAD format with proper multi-answer handling and validation.
    Added max_examples_per_contract to prevent overfitting on specific contracts.
    """
    processed_examples = []
    examples_from_contract = 0  # Track examples from this contract
    
    if 'paragraphs' not in example:
        return []
    
    for paragraph in example['paragraphs']:
        context = paragraph.get('context', '')
        
        if not context or len(context) < MIN_CONTEXT_CHARS:
            logger.warning(f"Contract {idx}: Context too short or empty")
            continue
        
        for qa in paragraph.get('qas', []):
            question = qa.get('question', '')
            answers = qa.get('answers', [])
            question_type = extract_question_type(question)
            
            if not question:
                continue
            
            # === POSITIVE EXAMPLES (with answers) ===
            if answers and len(answers) > 0:
                
                # Multi-answer questions: Parties, Document Name
                if len(answers) > 1 and question_type in ['Parties', 'Document Name']:
                    # Combine all answers
                    all_answers = [ans.get('text', '').strip() 
                                  for ans in answers if ans.get('text', '')]
                    
                    if not all_answers:
                        continue
                    
                    combined_answer = ', '.join(all_answers)
                    
                    # Find section containing ALL or MOST answers
                    first_answer_start = answers[0].get('answer_start', 0)
                    last_answer = answers[-1]
                    last_answer_start = last_answer.get('answer_start', first_answer_start)
                    last_answer_text = last_answer.get('text', '')
                    
                    # Calculate span covering all answers
                    answer_span_start = first_answer_start
                    answer_span_end = last_answer_start + len(last_answer_text)
                    answer_span_length = answer_span_end - answer_span_start
                    
                    section = extract_section(context, answer_span_start, 
                                            answer_span_length, context_window=2500)
                    
                    if section and len(section) >= MIN_CONTEXT_CHARS:
                        # Check if we've hit the limit for this contract
                        if max_examples_per_contract and examples_from_contract >= max_examples_per_contract:
                            logger.debug(f"Contract {idx}: Hit max examples limit ({max_examples_per_contract})")
                            return processed_examples
                            
                            example_data = {
                                'messages': [
                                    {'role': 'system', 'content': section},
                                    {'role': 'user', 'content': question},
                                    {'role': 'assistant', 'content': combined_answer}
                                ]
                            }
                            
                            if include_metadata:
                                example_data['metadata'] = {
                                    'has_answer': True,
                                    'answer_length': len(combined_answer),
                                    'context_length': len(section),
                                    'original_context_length': len(context),
                                    'question_type': question_type,
                                    'multi_answer': True,
                                    'num_answers': len(all_answers),
                                    'contract_id': idx
                                }
                            
                            processed_examples.append(example_data)
                            examples_from_contract += 1
                            
                # Single-answer questions OR questions with multiple instances
                else:
                    for answer_obj in answers:
                        answer_text = answer_obj.get('text', '').strip()
                        answer_start = answer_obj.get('answer_start', -1)
                        
                        if not answer_text or answer_start < 0:
                            continue
                        
                        section = extract_section(context, answer_start, 
                                                len(answer_text), context_window=2000)
                        
                        if section and len(section) >= MIN_CONTEXT_CHARS:
                            # Check if we've hit the limit for this contract
                            if max_examples_per_contract and examples_from_contract >= max_examples_per_contract:
                                logger.debug(f"Contract {idx}: Hit max examples limit ({max_examples_per_contract})")
                                return processed_examples
                            
                            example_data = {
                                'messages': [
                                    {'role': 'system', 'content': section},
                                    {'role': 'user', 'content': question},
                                    {'role': 'assistant', 'content': answer_text}
                                ]
                            }
                            
                            if include_metadata:
                                example_data['metadata'] = {
                                    'has_answer': True,
                                    'answer_length': len(answer_text),
                                    'context_length': len(section),
                                    'original_context_length': len(context),
                                    'question_type': question_type,
                                    'contract_id': idx
                                }
                            
                            processed_examples.append(example_data)
                            examples_from_contract += 1
                
                # Create negative examples from same contract
                if negative_ratio > 1.0:
                    num_negatives = int(negative_ratio - 1)
                    
                    for i in range(num_negatives):
                        import random
                        random.seed(idx * 100 + i)
                        
                        # Extract random section
                        max_start = max(0, len(context) - 3000)
                        rand_start = random.randint(0, max_start)
                        rand_end = min(rand_start + 3000, len(context))
                        
                        # Extend to sentence boundaries
                        rand_start, rand_end = extend_to_sentence_boundary(
                            context, rand_start, rand_end)
                        
                        section = context[rand_start:rand_end].strip()
                        
                        # Verify it doesn't contain any answer
                        contains_answer = False
                        for ans_obj in answers:
                            ans_text = ans_obj.get('text', '')
                            if ans_text and ans_text.lower() in section.lower():
                                contains_answer = True
                                break
                        
                        if not contains_answer and len(section) >= MIN_CONTEXT_CHARS:
                            # Check if we've hit the limit for this contract
                            if max_examples_per_contract and examples_from_contract >= max_examples_per_contract:
                                logger.debug(f"Contract {idx}: Hit max examples limit ({max_examples_per_contract}) during negative generation")
                                return processed_examples
                            
                            # Cap length
                            if len(section) > MAX_CONTEXT_CHARS:
                                section = section[:MAX_CONTEXT_CHARS]
                                # Re-extend to sentence boundary
                                _, end_pos = extend_to_sentence_boundary(
                                    section, 0, len(section))
                                section = section[:end_pos].strip()
                                
                                example_data = {
                                    'messages': [
                                        {'role': 'system', 'content': section},
                                        {'role': 'user', 'content': question},
                                        {'role': 'assistant', 'content': 'Not specified in this section.'}
                                    ]
                                }
                                
                                if include_metadata:
                                    example_data['metadata'] = {
                                        'has_answer': False,
                                        'context_length': len(section),
                                        'original_context_length': len(context),
                                        'question_type': question_type,
                                        'negative_type': 'from_positive_contract'
                                    }
                                
                                processed_examples.append(example_data)
                                examples_from_contract += 1
                            
            # === NEGATIVE EXAMPLES (no answers) ===
            else:
                # Take beginning of contract
                section = context[:3000].strip()
                
                # Extend to sentence boundary
                _, end = extend_to_sentence_boundary(context, 0, min(3000, len(context)))
                section = context[:end].strip()
                
                # Cap at max length
                if len(section) > MAX_CONTEXT_CHARS:
                    section = section[:MAX_CONTEXT_CHARS]
                    _, end_pos = extend_to_sentence_boundary(section, 0, len(section))
                    section = section[:end_pos].strip()
                
                if len(section) >= MIN_CONTEXT_CHARS:
                    # Check if we've hit the limit for this contract
                    if max_examples_per_contract and examples_from_contract >= max_examples_per_contract:
                        logger.debug(f"Contract {idx}: Hit max examples limit ({max_examples_per_contract}) during negative generation")
                        return processed_examples
                    
                    example_data = {
                        'messages': [
                            {'role': 'system', 'content': section},
                            {'role': 'user', 'content': question},
                            {'role': 'assistant', 'content': 'Not specified in this section.'}
                        ]
                    }
                    
                    if include_metadata:
                        example_data['metadata'] = {
                            'has_answer': False,
                            'context_length': len(section),
                            'original_context_length': len(context),
                            'question_type': question_type,
                            'negative_type': 'original_negative'
                        }
                    
                    processed_examples.append(example_data)
                    examples_from_contract += 1
    
    return processed_examples


def extract_question_type(question: str) -> str:
    """Extract question type from CUAD question format"""
    match = re.search(r'related to ["\']([^"\']+)["\']', question)
    if match:
        return match.group(1)
    return "Unknown"


def calculate_statistics(processed_examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate comprehensive dataset statistics"""
    
    stats = {
        'total_examples': len(processed_examples),
        'positive_examples': 0,
        'negative_examples': 0,
        'question_types': defaultdict(lambda: {'positive': 0, 'negative': 0}),
        'context_lengths': [],
        'answer_lengths': [],
        'original_context_lengths': [],
        'negative_types': defaultdict(int),
        'multi_answer_count': 0
    }
    
    for example in processed_examples:
        metadata = example.get('metadata', {})
        
        # If no metadata, infer from content
        if not metadata:
            messages = example.get('messages', [])
            assistant_msg = next((msg for msg in messages if msg.get('role') == 'assistant'), {})
            assistant_content = assistant_msg.get('content', '')
            
            has_answer = assistant_content != 'Not specified in this section.'
            
            system_msg = next((msg for msg in messages if msg.get('role') == 'system'), {})
            context_content = system_msg.get('content', '')
            
            if has_answer:
                stats['positive_examples'] += 1
                stats['answer_lengths'].append(len(assistant_content))
            else:
                stats['negative_examples'] += 1
                stats['negative_types']['inferred'] += 1
            
            stats['context_lengths'].append(len(context_content))
            stats['question_types']['Unknown']['positive' if has_answer else 'negative'] += 1
        else:
            # Use metadata if available
            has_answer = metadata.get('has_answer', False)
            
            if has_answer:
                stats['positive_examples'] += 1
                stats['answer_lengths'].append(metadata.get('answer_length', 0))
                if metadata.get('multi_answer'):
                    stats['multi_answer_count'] += 1
            else:
                stats['negative_examples'] += 1
                neg_type = metadata.get('negative_type', 'unknown')
                stats['negative_types'][neg_type] += 1
            
            # Question type distribution
            q_type = metadata.get('question_type', 'Unknown')
            if has_answer:
                stats['question_types'][q_type]['positive'] += 1
            else:
                stats['question_types'][q_type]['negative'] += 1
            
            # Length statistics
            stats['context_lengths'].append(metadata.get('context_length', 0))
            stats['original_context_lengths'].append(metadata.get('original_context_length', 0))
    
    # Calculate averages
    if stats['context_lengths']:
        stats['avg_context_length'] = sum(stats['context_lengths']) / len(stats['context_lengths'])
        stats['max_context_length'] = max(stats['context_lengths'])
        stats['min_context_length'] = min(stats['context_lengths'])
    
    if stats['answer_lengths']:
        stats['avg_answer_length'] = sum(stats['answer_lengths']) / len(stats['answer_lengths'])
        stats['max_answer_length'] = max(stats['answer_lengths'])
    
    if stats['original_context_lengths']:
        stats['avg_original_length'] = sum(stats['original_context_lengths']) / len(stats['original_context_lengths'])
    
    return stats


def print_statistics(stats: Dict[str, Any], output_path: str):
    """Print comprehensive statistics"""
    
    print("\n" + "="*80)
    print("📊 DATASET STATISTICS")
    print("="*80)
    
    # Overall statistics
    print(f"\n📈 Overall Statistics:")
    print(f"   Total examples: {stats['total_examples']:,}")
    
    if stats['total_examples'] == 0:
        print("   ⚠️  WARNING: No examples were processed!")
        print("   Check your input data format.")
        return
    
    print(f"   Positive examples: {stats['positive_examples']:,} ({stats['positive_examples']/stats['total_examples']*100:.1f}%)")
    print(f"   Negative examples: {stats['negative_examples']:,} ({stats['negative_examples']/stats['total_examples']*100:.1f}%)")
    print(f"   Positive:Negative ratio: 1:{stats['negative_examples']/max(stats['positive_examples'], 1):.2f}")
    print(f"   Multi-answer examples: {stats.get('multi_answer_count', 0):,}")
    
    # Negative types breakdown
    if stats['negative_types']:
        print(f"\n📋 Negative Example Types:")
        for neg_type, count in stats['negative_types'].items():
            print(f"   {neg_type}: {count:,}")
    
    # Context length statistics
    print(f"\n📏 Context Length Statistics:")
    print(f"   Average context: {stats.get('avg_context_length', 0):,.0f} characters (~{stats.get('avg_context_length', 0)//4:.0f} tokens)")
    print(f"   Min context: {stats.get('min_context_length', 0):,} characters")
    print(f"   Max context: {stats.get('max_context_length', 0):,} characters (~{stats.get('max_context_length', 0)//4:.0f} tokens)")
    print(f"   Average original contract: {stats.get('avg_original_length', 0):,.0f} characters")
    
    # Validate context lengths
    if stats.get('min_context_length', 0) < MIN_CONTEXT_CHARS:
        print(f"   ⚠️  WARNING: Some contexts below minimum ({MIN_CONTEXT_CHARS} chars)")
    if stats.get('max_context_length', 0) > MAX_CONTEXT_CHARS:
        print(f"   ⚠️  WARNING: Some contexts exceed maximum ({MAX_CONTEXT_CHARS} chars)")
    else:
        print(f"   ✅ All contexts within safe limits (100-8000 chars)")
    
    # Answer length statistics
    if stats['answer_lengths']:
        print(f"\n✍️  Answer Length Statistics:")
        print(f"   Average answer: {stats.get('avg_answer_length', 0):,.0f} characters")
        print(f"   Max answer: {stats.get('max_answer_length', 0):,} characters")
    
    # Question type distribution
    print(f"\n❓ Question Type Distribution (Top 20):")
    print(f"   {'Question Type':<40} {'Positive':<12} {'Negative':<12} {'Total':<12}")
    print(f"   {'-'*40} {'-'*12} {'-'*12} {'-'*12}")
    
    sorted_types = sorted(stats['question_types'].items(), 
                         key=lambda x: x[1]['positive'] + x[1]['negative'], 
                         reverse=True)
    
    for q_type, counts in sorted_types[:20]:
        total = counts['positive'] + counts['negative']
        print(f"   {q_type[:40]:<40} {counts['positive']:<12} {counts['negative']:<12} {total:<12}")
    
    if len(sorted_types) > 20:
        print(f"   ... and {len(sorted_types) - 20} more question types")
    
    # Save statistics to JSON
    stats_path = output_path.replace('.jsonl', '_statistics.json')
    
    json_stats = {
        'total_examples': stats['total_examples'],
        'positive_examples': stats['positive_examples'],
        'negative_examples': stats['negative_examples'],
        'multi_answer_count': stats.get('multi_answer_count', 0),
        'avg_context_length': stats.get('avg_context_length', 0),
        'max_context_length': stats.get('max_context_length', 0),
        'min_context_length': stats.get('min_context_length', 0),
        'avg_answer_length': stats.get('avg_answer_length', 0),
        'question_types': dict(stats['question_types']),
        'negative_types': dict(stats['negative_types'])
    }
    
    with open(stats_path, 'w') as f:
        json.dump(json_stats, f, indent=2)
    
    print(f"\n💾 Statistics saved to: {stats_path}")
    print("="*80 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="CUAD Dataset Preprocessor - Final Version")
    parser.add_argument("--input_dataset", type=str, required=True,
                       help="Path to CUAD dataset (JSON or JSONL)")
    parser.add_argument("--output_dataset", type=str, required=True,
                       help="Output path for processed dataset")
    parser.add_argument("--negative_ratio", type=float, default=1.0, 
                       help="Ratio of negative to positive examples (1.0-2.0 recommended)")
    parser.add_argument("--max_examples_per_contract", type=int, default=None,
                       help="Maximum examples per contract to prevent overfitting (10-20 recommended)")
    parser.add_argument("--sample_ratio", type=float, default=None,
                       help="Sample ratio of original dataset (0.0-1.0)")
    parser.add_argument("--validation_split", type=float, default=0.15,
                       help="Validation set proportion (default: 0.15)")
    parser.add_argument("--create_validation", action="store_true",
                       help="Create separate train/validation splits")
    parser.add_argument("--include_metadata", action="store_true",
                       help="Include metadata in output (for analysis only)")
    parser.add_argument("--clean_format", action="store_true",
                       help="Output clean format without metadata (recommended for training)")
    
    args = parser.parse_args()
    
    # Set default behavior: clean format unless metadata explicitly requested
    if not args.include_metadata and not args.clean_format:
        args.include_metadata = False  # Default to clean format for training
    elif args.clean_format:
        args.include_metadata = False
    
    logger.info(f"CUAD Preprocessor - Config:")
    logger.info(f"  Max context: {MAX_CONTEXT_CHARS} chars (~{MAX_CONTEXT_CHARS//4} tokens)")
    logger.info(f"  Min context: {MIN_CONTEXT_CHARS} chars")
    logger.info(f"  Negative ratio: {args.negative_ratio}")
    logger.info(f"  Include metadata: {args.include_metadata}")
    logger.info(f"  Clean format: {not args.include_metadata}")
    
    if not args.include_metadata:
        logger.info("  → Generating CLEAN format for training (no metadata)")
    else:
        logger.info("  → Generating format with metadata for analysis")
    
    # Load dataset
    logger.info(f"\nLoading dataset: {args.input_dataset}")
    
    if args.input_dataset.endswith('.jsonl'):
        dataset = load_dataset("json", data_files=args.input_dataset, split="train")
        original_size = len(dataset)
        logger.info(f"Loaded {original_size:,} contracts from JSONL")
    else:
        try:
            with open(args.input_dataset, 'r') as f:
                raw_data = json.load(f)
            
            if isinstance(raw_data, dict) and 'data' in raw_data:
                logger.info("Detected CUAD format with nested 'data' field")
                contracts = raw_data['data']
                logger.info(f"Found {len(contracts)} contracts")
                
                import tempfile
                temp_path = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False).name
                with open(temp_path, 'w') as f:
                    for contract in contracts:
                        f.write(json.dumps(contract) + '\n')
                
                dataset = load_dataset("json", data_files=temp_path, split="train")
                original_size = len(dataset)
            else:
                dataset = load_dataset("json", data_files=args.input_dataset, split="train")
                original_size = len(dataset)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return
    
    # Sample if requested
    if args.sample_ratio and args.sample_ratio < 1.0:
        sample_size = int(original_size * args.sample_ratio)
        dataset = dataset.shuffle(seed=42).select(range(sample_size))
        logger.info(f"Sampled: {len(dataset):,} contracts ({args.sample_ratio:.1%})")
    
    # Process examples
    all_processed = []
    failed_count = 0
    
    logger.info("\nProcessing examples...")
    for idx, example in enumerate(tqdm(dataset, desc="Processing")):
        try:
            processed = process_cuad_example(example, idx, args.negative_ratio, 
                                           include_metadata=args.include_metadata,
                                           max_examples_per_contract=args.max_examples_per_contract)
            all_processed.extend(processed)
        except Exception as e:
            failed_count += 1
            if failed_count <= 5:
                logger.error(f"Error processing example {idx}: {e}")
    
    if failed_count > 0:
        logger.warning(f"Failed to process {failed_count} examples")
    
    logger.info(f"Created {len(all_processed):,} training examples from {len(dataset):,} contracts")
    
    if len(all_processed) == 0:
        logger.error("No examples created! Check your data format and logs.")
        return
    
    # Calculate statistics
    logger.info("Calculating statistics...")
    stats = calculate_statistics(all_processed)
    
    # Create dataset
    processed_dataset = Dataset.from_list(all_processed)
    
    # Split if requested
    if args.create_validation:
        split = processed_dataset.train_test_split(
            test_size=args.validation_split, seed=42)
        
        train_path = args.output_dataset.replace('.jsonl', '_train.jsonl')
        val_path = args.output_dataset.replace('.jsonl', '_val.jsonl')
        
        split['train'].to_json(train_path)
        split['test'].to_json(val_path)
        
        logger.info(f"\n✅ Saved train: {train_path} ({len(split['train'])} examples)")
        logger.info(f"✅ Saved val: {val_path} ({len(split['test'])} examples)")
        
        # Calculate and print statistics
        train_stats = calculate_statistics(split['train'])
        val_stats = calculate_statistics(split['test'])
        
        print("\n" + "="*80)
        print("📚 TRAINING SET STATISTICS")
        print_statistics(train_stats, train_path)
        
        print("\n" + "="*80)
        print("🔍 VALIDATION SET STATISTICS")
        print_statistics(val_stats, val_path)
        
    else:
        processed_dataset.to_json(args.output_dataset)
        logger.info(f"\n✅ Saved: {args.output_dataset} ({len(processed_dataset)} examples)")
        print_statistics(stats, args.output_dataset)


if __name__ == "__main__":
    main()