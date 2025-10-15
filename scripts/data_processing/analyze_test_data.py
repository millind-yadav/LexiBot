import json
import argparse
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def analyze_jsonl(file_path: str):
    """
    Analyzes a JSONL file to understand its structure, content, and class balance.
    """
    total_lines = 0
    class_distribution = defaultdict(int)
    role_counts = defaultdict(int)
    malformed_lines = 0
    
    logger.info(f"Starting analysis of {file_path}...")

    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            total_lines += 1
            try:
                data = json.loads(line)
                
                # --- Check for 'messages' field ---
                if 'messages' not in data or not isinstance(data['messages'], list):
                    logger.warning(f"Line {i+1}: Missing or malformed 'messages' field.")
                    malformed_lines += 1
                    continue

                # --- Analyze class label ---
                # Find the assistant's message to determine the label
                assistant_message = next((msg['content'] for msg in data['messages'] if msg['role'] == 'assistant'), None)
                
                if assistant_message is None:
                    logger.warning(f"Line {i+1}: No assistant message found to determine label.")
                    label = "unknown"
                elif "true" in assistant_message.lower():
                    label = "true"
                elif "false" in assistant_message.lower():
                    label = "false"
                else:
                    # This case helps find if the model was trained on "yes"/"no" instead of "true"/"false"
                    if "yes" in assistant_message.lower():
                         label = "yes"
                    elif "no" in assistant_message.lower():
                         label = "no"
                    else:
                        label = "other_assistant_response"

                class_distribution[label] += 1

                # --- Analyze message roles ---
                for message in data['messages']:
                    if 'role' in message:
                        role_counts[message['role']] += 1
                    else:
                        logger.warning(f"Line {i+1}: Message without a 'role' found.")

            except json.JSONDecodeError:
                logger.error(f"Line {i+1}: Failed to decode JSON.")
                malformed_lines += 1

    # --- Print Summary ---
    logger.info("\n--- Analysis Complete ---")
    logger.info(f"Total examples processed: {total_lines}")
    if malformed_lines > 0:
        logger.warning(f"Malformed lines found: {malformed_lines}")
    
    logger.info("\n--- Class Distribution ---")
    for label, count in class_distribution.items():
        percentage = (count / total_lines) * 100
        logger.info(f"Label '{label}': {count} examples ({percentage:.2f}%)")

    logger.info("\n--- Message Role Distribution ---")
    for role, count in role_counts.items():
        logger.info(f"Role '{role}': {count} messages")
        
    logger.info("\n--- Recommendations ---")
    if class_distribution.get("true", 0) == 0 and class_distribution.get("yes", 0) == 0:
        logger.warning("No 'true' or 'yes' labels found. This is likely the primary reason for the model's behavior.")
    elif (class_distribution.get("true", 0) + class_distribution.get("yes", 0)) < (total_lines * 0.1):
         logger.warning("Significant class imbalance detected. The positive class ('true'/'yes') is rare, which can cause models to learn to always predict the majority negative class.")
    else:
        logger.info("Class balance seems reasonable.")

    if "other_assistant_response" in class_distribution:
        logger.warning("Found assistant responses that are not 'true'/'false' or 'yes'/'no'. This could confuse the label extraction logic during evaluation.")

    logger.info("Script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a JSONL dataset file.")
    parser.add_argument("file_path", type=str, help="The path to the JSONL file to analyze.")
    args = parser.parse_args()
    
    analyze_jsonl(args.file_path)
