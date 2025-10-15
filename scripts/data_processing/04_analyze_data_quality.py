import json
import re
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm


BOILERPLATE_REGEX_LIST = [
    re.compile(r"exhibit \d+\.\d+", re.IGNORECASE),
    re.compile(r"confidential treatment requested", re.IGNORECASE),
    re.compile(r"has been separately filed with the commission", re.IGNORECASE),
    re.compile(r"this agreement is not intended to and shall not be construed to", re.IGNORECASE),
    re.compile(r"in witness whereof", re.IGNORECASE),
    re.compile(r"the parties have executed this agreement", re.IGNORECASE),
    re.compile(r"as of the date first written above", re.IGNORECASE),
    re.compile(r"all rights reserved", re.IGNORECASE),
    re.compile(r"table of contents", re.IGNORECASE),
]


def analyze_text_quality(text: str) -> dict:
    """
    Analyzes a single string of text and returns a dictionary of quality metrics.

    Args:
        text: The text content of a document.

    Returns:
        A dictionary containing various quality metrics.
    """
    if not isinstance(text, str) or not text:
        return {
            "char_count": 0,
            "redaction_char_count": 0,
            "redaction_percentage": 0,
            "whitespace_char_count": 0,
            "whitespace_percentage": 0,
            "alphanumeric_percentage": 0,
            "line_count": 0,
            "avg_line_length": 0,
            "boilerplate_count": 0,
            "quality_score": 0,
        }

    char_count = len(text)
    
    redaction_char_count = text.lower().count('[***]') * 5
    redaction_percentage = (redaction_char_count / char_count) * 100 if char_count > 0 else 0

    whitespace_char_count = text.count(' ') + text.count('\n') + text.count('\t')
    whitespace_percentage = (whitespace_char_count / char_count) * 100 if char_count > 0 else 0
    
    lines = text.splitlines()
    line_count = len(lines)
    avg_line_length = np.mean([len(line) for line in lines]) if lines else 0

    alphanumeric_chars = sum(1 for char in text if char.isalnum())
    alphanumeric_percentage = (alphanumeric_chars / char_count) * 100 if char_count > 0 else 0

    boilerplate_count = sum(1 for regex in BOILERPLATE_REGEX_LIST if regex.search(text))

    quality_score = (redaction_percentage * 0.5) + (whitespace_percentage * 0.3) + ((100 - alphanumeric_percentage) * 0.2)

    return {
        "char_count": char_count,
        "redaction_char_count": redaction_char_count,
        "redaction_percentage": redaction_percentage,
        "whitespace_char_count": whitespace_char_count,
        "whitespace_percentage": whitespace_percentage,
        "alphanumeric_percentage": alphanumeric_percentage,
        "line_count": line_count,
        "avg_line_length": avg_line_length,
        "boilerplate_count": boilerplate_count,
        "quality_score": quality_score,
    }

def process_dataset(file_path: Path) -> pd.DataFrame:
    """
    Loads a JSONL dataset and analyzes the quality of each document.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        A pandas DataFrame with quality metrics for each document.
    """
    records = []
    with file_path.open('r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading and analyzing documents"):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed JSON line: {line[:100]}")
                continue

    analysis_results = []
    for record in tqdm(records, desc="Analyzing text quality"):
        # The context is in the 'system' role message
        system_message = ""
        if 'messages' in record and isinstance(record['messages'], list):
            for msg in record['messages']:
                if msg.get('role') == 'system':
                    system_message = msg.get('content', '')
                    break
        
        metrics = analyze_text_quality(system_message)
        metrics['original_record'] = record  
        analysis_results.append(metrics)

    return pd.DataFrame(analysis_results)

def generate_summary_report(df: pd.DataFrame):
    """
    Prints a summary report of the data quality analysis to the console.
    """
    print("\n--- Data Quality Analysis Report ---")
    print(f"\nTotal Documents Analyzed: {len(df)}")

    if len(df) == 0:
        print("No data to analyze.")
        return

    # --- Summary Statistics ---
    print("\n[1] Overall Quality Metrics (Summary Statistics):")
    summary_stats = df[[
        "quality_score", "redaction_percentage", "whitespace_percentage", 
        "alphanumeric_percentage", "boilerplate_count"
    ]].describe(percentiles=[.25, .5, .75, .9, .99])
    print(summary_stats.to_string())

    # --- Problematic Documents ---
    print("\n[2] Identifying Problematic Documents:")
    
    high_redaction_threshold = 50.0
    high_whitespace_threshold = 60.0
    low_alphanum_threshold = 40.0

    print(f"\nDocuments with > {high_redaction_threshold}% redaction: {len(df[df['redaction_percentage'] > high_redaction_threshold])}")
    print(f"Documents with > {high_whitespace_threshold}% whitespace: {len(df[df['whitespace_percentage'] > high_whitespace_threshold])}")
    print(f"Documents with < {low_alphanum_threshold}% alphanumeric chars: {len(df[df['alphanumeric_percentage'] < low_alphanum_threshold])}")
    print(f"Documents with boilerplate phrases: {len(df[df['boilerplate_count'] > 0])}")

    # --- Top 5 Worst Offenders ---
    print("\n[3] Top 5 Worst Documents (by Quality Score):")
    worst_docs = df.nlargest(5, 'quality_score')
    for i, row in worst_docs.iterrows():
        print(f"\n--- Document Rank #{i+1} (Score: {row['quality_score']:.2f}) ---")
        print(f"  - Redaction: {row['redaction_percentage']:.2f}%")
        print(f"  - Whitespace: {row['whitespace_percentage']:.2f}%")
        print(f"  - Alphanumeric: {row['alphanumeric_percentage']:.2f}%")
        
        system_content = ""
        messages = row['original_record'].get('messages', [])
        for msg in messages:
            if msg.get('role') == 'system':
                system_content = msg.get('content', '')
                break
        snippet = system_content[:200].strip().replace('\n', ' ')
        print(f"  - Snippet: '{snippet}...'")

    print("\n--- End of Report ---")


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(description="Analyze the quality of a JSONL dataset for LLM training.")
    parser.add_argument("file_path", type=str, help="Path to the JSONL dataset file.")
    args = parser.parse_args()

    file_path = Path(args.file_path)
    if not file_path.is_file():
        print(f"Error: File not found at {file_path}")
        return

    df = process_dataset(file_path)
    generate_summary_report(df)


if __name__ == "__main__":
    main()
