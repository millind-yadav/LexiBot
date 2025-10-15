#!/usr/bin/env python3
"""
Dataset Analysis Script for Identifying Problematic Examples

This script analyzes a JSONL dataset to identify examples that might cause
formatting timeouts or other issues during training preparation.
"""

import json
import argparse
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_example(example: Dict[str, Any], tokenizer: Any, idx: int) -> Dict[str, Any]:
    """Analyze a single example for potential issues."""
    analysis = {
        'idx': idx,
        'has_system': False,
        'has_user': False,
        'has_assistant': False,
        'message_count': 0,
        'system_chars': 0,
        'user_chars': 0,
        'assistant_chars': 0,
        'total_chars': 0,
        'system_tokens': 0,
        'user_tokens': 0,
        'assistant_tokens': 0,
        'total_tokens': 0,
        'max_single_message_chars': 0,
        'max_single_message_tokens': 0,
        'tokenization_time': 0,
        'issues': [],
        'risk_level': 'low'
    }
    
    try:
        messages = example.get('messages', [])
        analysis['message_count'] = len(messages)
        
        if not messages:
            analysis['issues'].append('no_messages')
            analysis['risk_level'] = 'high'
            return analysis
        
        start_time = time.time()
        
        for msg in messages:
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                analysis['issues'].append('invalid_message_format')
                continue
                
            role = msg['role']
            content = str(msg['content'])
            char_count = len(content)
            
            # Update role-specific stats
            if role == 'system':
                analysis['has_system'] = True
                analysis['system_chars'] = char_count
            elif role == 'user':
                analysis['has_user'] = True
                analysis['user_chars'] = char_count
            elif role == 'assistant':
                analysis['has_assistant'] = True
                analysis['assistant_chars'] = char_count
            
            # Track maximum single message size
            analysis['max_single_message_chars'] = max(analysis['max_single_message_chars'], char_count)
            
            # Tokenize content to get token count
            try:
                tokens = tokenizer.encode(content, add_special_tokens=False)
                token_count = len(tokens)
                
                if role == 'system':
                    analysis['system_tokens'] = token_count
                elif role == 'user':
                    analysis['user_tokens'] = token_count
                elif role == 'assistant':
                    analysis['assistant_tokens'] = token_count
                
                analysis['max_single_message_tokens'] = max(analysis['max_single_message_tokens'], token_count)
                
            except Exception as e:
                analysis['issues'].append(f'tokenization_failed_{role}')
                logger.debug(f"Tokenization failed for example {idx}, role {role}: {e}")
        
        # Calculate totals
        analysis['total_chars'] = analysis['system_chars'] + analysis['user_chars'] + analysis['assistant_chars']
        analysis['total_tokens'] = analysis['system_tokens'] + analysis['user_tokens'] + analysis['assistant_tokens']
        analysis['tokenization_time'] = time.time() - start_time
        
        # Identify potential issues
        if not analysis['has_user'] or not analysis['has_assistant']:
            analysis['issues'].append('missing_required_roles')
            analysis['risk_level'] = 'high'
        
        if analysis['total_chars'] > 100000:  # Very large examples
            analysis['issues'].append('very_large_content')
            analysis['risk_level'] = 'high'
        elif analysis['total_chars'] > 50000:
            analysis['issues'].append('large_content')
            analysis['risk_level'] = 'medium'
        
        if analysis['total_tokens'] > 20000:  # Would require many chunks
            analysis['issues'].append('very_high_token_count')
            analysis['risk_level'] = 'high'
        elif analysis['total_tokens'] > 10000:
            analysis['issues'].append('high_token_count')
            analysis['risk_level'] = 'medium'
        
        if analysis['tokenization_time'] > 1.0:  # Slow tokenization
            analysis['issues'].append('slow_tokenization')
            if analysis['risk_level'] == 'low':
                analysis['risk_level'] = 'medium'
        
        # Check for extremely unbalanced content
        max_content = max(analysis['system_chars'], analysis['user_chars'], analysis['assistant_chars'])
        if max_content > 0:
            min_content = min(c for c in [analysis['system_chars'], analysis['user_chars'], analysis['assistant_chars']] if c > 0)
            if max_content / max(min_content, 1) > 100:  # One message 100x larger than others
                analysis['issues'].append('highly_unbalanced_content')
                if analysis['risk_level'] == 'low':
                    analysis['risk_level'] = 'medium'
        
    except Exception as e:
        analysis['issues'].append(f'analysis_error: {str(e)}')
        analysis['risk_level'] = 'high'
        logger.error(f"Error analyzing example {idx}: {e}")
    
    return analysis

def generate_report(analyses: List[Dict[str, Any]], output_path: Path) -> None:
    """Generate a comprehensive analysis report."""
    
    # Overall statistics
    total_examples = len(analyses)
    risk_counts = defaultdict(int)
    issue_counts = defaultdict(int)
    
    high_risk_examples = []
    medium_risk_examples = []
    
    char_stats = []
    token_stats = []
    tokenization_times = []
    
    for analysis in analyses:
        risk_counts[analysis['risk_level']] += 1
        
        for issue in analysis['issues']:
            issue_counts[issue] += 1
        
        if analysis['risk_level'] == 'high':
            high_risk_examples.append(analysis)
        elif analysis['risk_level'] == 'medium':
            medium_risk_examples.append(analysis)
        
        char_stats.append(analysis['total_chars'])
        token_stats.append(analysis['total_tokens'])
        tokenization_times.append(analysis['tokenization_time'])
    
    # Generate report
    report_lines = [
        "# Dataset Analysis Report",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary Statistics",
        f"- Total examples: {total_examples:,}",
        f"- High risk examples: {risk_counts['high']:,} ({100*risk_counts['high']/total_examples:.1f}%)",
        f"- Medium risk examples: {risk_counts['medium']:,} ({100*risk_counts['medium']/total_examples:.1f}%)",
        f"- Low risk examples: {risk_counts['low']:,} ({100*risk_counts['low']/total_examples:.1f}%)",
        "",
        "## Content Size Statistics",
        f"- Character count - Mean: {np.mean(char_stats):,.0f}, Median: {np.median(char_stats):,.0f}, Max: {np.max(char_stats):,.0f}",
        f"- Token count - Mean: {np.mean(token_stats):,.0f}, Median: {np.median(token_stats):,.0f}, Max: {np.max(token_stats):,.0f}",
        f"- Tokenization time - Mean: {np.mean(tokenization_times):.3f}s, Max: {np.max(tokenization_times):.3f}s",
        "",
        "## Common Issues",
    ]
    
    # Sort issues by frequency
    sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
    for issue, count in sorted_issues:
        percentage = 100 * count / total_examples
        report_lines.append(f"- {issue}: {count:,} examples ({percentage:.1f}%)")
    
    # High risk examples details
    if high_risk_examples:
        report_lines.extend([
            "",
            "## High Risk Examples (First 20)",
            ""
        ])
        
        high_risk_examples.sort(key=lambda x: x['total_chars'], reverse=True)
        for i, analysis in enumerate(high_risk_examples[:20]):
            report_lines.extend([
                f"### Example {analysis['idx']} (Rank #{i+1})",
                f"- Characters: {analysis['total_chars']:,} (sys: {analysis['system_chars']:,}, user: {analysis['user_chars']:,}, asst: {analysis['assistant_chars']:,})",
                f"- Tokens: {analysis['total_tokens']:,} (sys: {analysis['system_tokens']:,}, user: {analysis['user_tokens']:,}, asst: {analysis['assistant_tokens']:,})",
                f"- Tokenization time: {analysis['tokenization_time']:.3f}s",
                f"- Issues: {', '.join(analysis['issues'])}",
                ""
            ])
    
    # Content size distribution
    report_lines.extend([
        "",
        "## Content Size Distribution",
        ""
    ])
    
    # Character distribution
    char_percentiles = np.percentile(char_stats, [50, 75, 90, 95, 99])
    report_lines.extend([
        "### Character Count Percentiles",
        f"- 50th: {char_percentiles[0]:,.0f}",
        f"- 75th: {char_percentiles[1]:,.0f}",
        f"- 90th: {char_percentiles[2]:,.0f}",
        f"- 95th: {char_percentiles[3]:,.0f}",
        f"- 99th: {char_percentiles[4]:,.0f}",
        ""
    ])
    
    # Token distribution
    token_percentiles = np.percentile(token_stats, [50, 75, 90, 95, 99])
    report_lines.extend([
        "### Token Count Percentiles",
        f"- 50th: {token_percentiles[0]:,.0f}",
        f"- 75th: {token_percentiles[1]:,.0f}",
        f"- 90th: {token_percentiles[2]:,.0f}",
        f"- 95th: {token_percentiles[3]:,.0f}",
        f"- 99th: {token_percentiles[4]:,.0f}",
        ""
    ])
    
    # Recommendations
    report_lines.extend([
        "## Recommendations",
        ""
    ])
    
    if risk_counts['high'] > 0:
        report_lines.append(f"- Consider filtering out {risk_counts['high']} high-risk examples to improve processing speed")
    
    if issue_counts.get('very_large_content', 0) > 0:
        report_lines.append(f"- {issue_counts['very_large_content']} examples have >100k characters - these will be very slow to process")
    
    if issue_counts.get('very_high_token_count', 0) > 0:
        report_lines.append(f"- {issue_counts['very_high_token_count']} examples have >20k tokens - consider reducing max_seq_length")
    
    if np.max(tokenization_times) > 5.0:
        report_lines.append(f"- Some examples take >5s just to tokenize - consider preprocessing with faster tokenizers")
    
    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Analysis report written to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze dataset for problematic examples")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to JSONL dataset file")
    parser.add_argument("--model_name", type=str, default="unsloth/llama-3.1-8b-bnb-4bit", help="Model name for tokenizer")
    parser.add_argument("--output_dir", type=str, default="./analysis_results", help="Output directory for analysis results")
    parser.add_argument("--sample_size", type=int, default=None, help="Analyze only first N examples (default: all)")
    parser.add_argument("--save_problematic", action="store_true", help="Save problematic examples to separate file")
    parser.add_argument("--timeout_simulation", action="store_true", help="Simulate timeout conditions")
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset_path}")
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    
    total_examples = len(dataset)
    if args.sample_size:
        total_examples = min(args.sample_size, total_examples)
        dataset = dataset.select(range(total_examples))
    
    logger.info(f"Analyzing {total_examples} examples...")
    
    # Analyze examples
    analyses = []
    problematic_examples = []
    
    start_time = time.time()
    
    for idx, example in enumerate(dataset):
        if idx % 1000 == 0:
            elapsed = time.time() - start_time
            if idx > 0:
                eta = (elapsed / idx) * (total_examples - idx)
                logger.info(f"Progress: {idx}/{total_examples} ({100*idx/total_examples:.1f}%) - ETA: {eta/60:.1f}m")
        
        analysis = analyze_example(example, tokenizer, idx)
        analyses.append(analysis)
        
        # Collect problematic examples
        if analysis['risk_level'] in ['high', 'medium']:
            problematic_examples.append({
                'idx': idx,
                'analysis': analysis,
                'example': example
            })
    
    total_time = time.time() - start_time
    logger.info(f"Analysis completed in {total_time:.1f}s ({total_examples/total_time:.1f} examples/s)")
    
    # Generate main report
    report_path = output_dir / "dataset_analysis_report.md"
    generate_report(analyses, report_path)
    
    # Save detailed analysis as JSON
    json_path = output_dir / "detailed_analysis.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(analyses, f, indent=2, ensure_ascii=False)
    logger.info(f"Detailed analysis saved to: {json_path}")
    
    # Save problematic examples if requested
    if args.save_problematic and problematic_examples:
        problematic_path = output_dir / "problematic_examples.json"
        with open(problematic_path, 'w', encoding='utf-8') as f:
            json.dump(problematic_examples, f, indent=2, ensure_ascii=False)
        logger.info(f"Problematic examples saved to: {problematic_path}")
    
    # Print summary
    risk_counts = defaultdict(int)
    for analysis in analyses:
        risk_counts[analysis['risk_level']] += 1
    
    print(f"\n📊 Analysis Summary:")
    print(f"   Total examples: {total_examples:,}")
    print(f"   High risk: {risk_counts['high']:,} ({100*risk_counts['high']/total_examples:.1f}%)")
    print(f"   Medium risk: {risk_counts['medium']:,} ({100*risk_counts['medium']/total_examples:.1f}%)")
    print(f"   Low risk: {risk_counts['low']:,} ({100*risk_counts['low']/total_examples:.1f}%)")
    print(f"\n📁 Results saved to: {output_dir}")
    
    if risk_counts['high'] > 0:
        print(f"\n⚠️  Consider filtering out {risk_counts['high']} high-risk examples for faster processing")

if __name__ == "__main__":
    main()