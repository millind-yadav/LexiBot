# ===== CELL 1: Install Dependencies =====

# ===== CELL 2: Import Libraries =====
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

# ===== CELL 3: Load Model =====
# Configure your model path
MODEL_PATH = "/home/milindyadav98/mistral-tuner/lexibot_l4_runV2/final_model"  # Change this to your model path
# Or if you uploaded to working directory: MODEL_PATH = "./your-model-folder"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Set padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
# Use 8-bit for Kaggle's GPU memory (P100/T4)
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16
)

model.eval()
print("✓ Model loaded successfully!")
print(f"Device: {model.device}")

DEFAULT_MAX_NEW_TOKENS = 400

SYSTEM_PROMPT = (
    "You are a precise legal analyst. Base every answer strictly on the supplied "
    "contract excerpt. Quote or paraphrase only the relevant clauses. If the "
    "answer is not present, say so plainly without extra boilerplate."
)

# ===== CELL 4: Generation Function =====
def generate_text(prompt,
                  max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                  temperature=0.1,
                  top_p=0.9,
                  top_k=40,
                  do_sample=False,
                  repetition_penalty=1.05):
    """
    Generate text from a prompt
    """
    # Tokenize
    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt.strip()}"
    inputs = tokenizer(full_prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated continuation (exclude the prompt tokens)
    prompt_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][prompt_length:]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return text

# ===== CELL 5: CUAD Legal Contract Test Suite =====
print("="*70)
print("CUAD LEGAL CONTRACT MODEL - TEST SUITE")
print("="*70)

# TEST 1: Payment Terms Extraction
print("\n" + "="*70)
print("TEST 1: Payment Terms Extraction")
print("="*70)

contract_1 = """Contract: Software License Agreement

Section 5.2: The Licensee agrees to pay the Licensor a one-time fee of $50,000 
within 30 days of execution of this Agreement. Late payments will incur an 
interest rate of 5% per annum.

Question: What are the payment terms in this contract?
Answer:"""

print(contract_1)
response = generate_text(contract_1, temperature=0.1)
print(f"\n🤖 Model Response:\n{response}\n")

# TEST 2: Termination Clauses
print("\n" + "="*70)
print("TEST 2: Termination Clause Analysis")
print("="*70)

contract_2 = """Contract: Service Agreement

Section 8.1: Either party may terminate this Agreement with 60 days written 
notice. In the event of material breach, the non-breaching party may terminate 
immediately upon written notice. Upon termination, all outstanding fees become 
immediately due and payable.

Question: Under what conditions can this contract be terminated?
Answer:"""

print(contract_2)
response = generate_text(contract_2, temperature=0.1)
print(f"\n🤖 Model Response:\n{response}\n")

# TEST 3: Liability Limitations
print("\n" + "="*70)
print("TEST 3: Liability Clause Extraction")
print("="*70)

contract_3 = """Contract: Manufacturing Agreement

Section 12: LIMITATION OF LIABILITY
The Company's total liability under this Agreement shall not exceed the total 
amount paid by the Customer in the twelve months preceding the claim. The 
Company shall not be liable for any indirect, incidental, special, or 
consequential damages.

Question: What are the liability limitations in this contract?
Answer:"""

print(contract_3)
response = generate_text(contract_3, temperature=0.1)
print(f"\n🤖 Model Response:\n{response}\n")

# TEST 4: Confidentiality
print("\n" + "="*70)
print("TEST 4: Confidentiality Analysis")
print("="*70)

contract_4 = """Contract: Non-Disclosure Agreement

Section 3: The Receiving Party agrees to hold all Confidential Information in 
strict confidence for a period of 5 years from the date of disclosure. The 
Receiving Party shall not disclose such information to any third party without 
prior written consent of the Disclosing Party.

Question: What are the confidentiality obligations?
Answer:"""

print(contract_4)
response = generate_text(contract_4, temperature=0.1)
print(f"\n🤖 Model Response:\n{response}\n")

# TEST 5: Intellectual Property
print("\n" + "="*70)
print("TEST 5: IP Rights Identification")
print("="*70)

contract_5 = """Contract: Development Agreement

Section 7: INTELLECTUAL PROPERTY
All intellectual property rights, including patents, copyrights, and trade 
secrets, created under this Agreement shall be the sole property of the Client. 
The Developer hereby assigns all rights, title, and interest in such intellectual 
property to the Client.

Question: Who owns the intellectual property created under this agreement?
Answer:"""

print(contract_5)
response = generate_text(contract_5, temperature=0.1)
print(f"\n🤖 Model Response:\n{response}\n")

# ===== CELL 6: Quick Contract Analysis (Optional) =====
# Quick test for common contract scenarios
print("\n" + "="*70)
print("QUICK CONTRACT ANALYSIS TESTS")
print("="*70)

quick_tests = [
    """Contract clause: "This Agreement is governed by the laws of California."
Question: What is the governing law?
Answer:""",
    
    """Contract clause: "The warranty period is 24 months from date of delivery."
Question: What is the warranty duration?
Answer:""",
    
    """Contract clause: "The Contractor shall indemnify Client against all third-party claims."
Question: Who provides indemnification?
Answer:"""
]

for i, test in enumerate(quick_tests, 1):
    print(f"\nQuick Test {i}:")
    print("-"*70)
    print(test)
    response = generate_text(test, temperature=0.1)
    print(f"🤖 Response: {response}\n")

# ===== CELL 7: Multi-Clause Analysis (Optional) =====
# Complex contract with multiple clauses
print("\n" + "="*70)
print("MULTI-CLAUSE CONTRACT ANALYSIS")
print("="*70)

complex_contract = """Contract: Master Service Agreement

Section 4.1: Payment terms are Net 30 from invoice date.
Section 7.2: Either party may terminate with 90 days notice.
Section 9.1: Contractor shall indemnify Client against third-party claims.
Section 10.3: All work product is the exclusive property of Client.
Section 13.1: This Agreement is governed by California law.

Question: Summarize the key terms of this contract including payment, termination, 
indemnification, IP ownership, and governing law.
Answer:"""

print(complex_contract)
response = generate_text(complex_contract, temperature=0.1)
print(f"\n🤖 Model Response:\n{response}\n")

# ===== CELL 8: Risk Assessment (Optional) =====
# Identify risky clauses
print("\n" + "="*70)
print("RISK ASSESSMENT TEST")
print("="*70)

risky_contract = """Contract: Vendor Agreement

Section 8: The Vendor agrees to unlimited liability for any and all claims, 
whether direct or indirect, arising from this Agreement. The Client may modify 
the terms of this Agreement at any time without notice. The Vendor waives all 
rights to dispute resolution.

Question: Identify any potentially unfavorable or risky clauses for the Vendor.
Answer:"""

print(risky_contract)
response = generate_text(risky_contract, temperature=0.1)
print(f"\n🤖 Model Response:\n{response}\n")

print("="*70)
print("✅ CUAD TEST SUITE COMPLETED")
print("="*70)
print("""
Your CUAD-trained model (loss 0.55) should excel at:
✓ Extracting specific contract clauses
✓ Understanding legal terminology
✓ Identifying payment, termination, liability terms
✓ Analyzing IP, confidentiality, and indemnification
✓ Summarizing complex multi-clause contracts

NOTE: Always use temperature=0.3 for legal contract analysis
      (more factual and precise, less creative)
""")

# ===== CELL 9: Save Contract Analysis Results (Optional) =====
# Save analysis results to file
import json

analysis_results = {
    "model_info": {
        "model_path": MODEL_PATH,
        "training_loss": 0.55,
        "dataset": "CUAD",
        "model_size": "3B parameters"
    },
    "test_date": "2024",
    "notes": "CUAD legal contract analysis tests"
}

with open("cuad_analysis_results.json", "w") as f:
    json.dump(analysis_results, f, indent=2)

print("Analysis results saved to cuad_analysis_results.json")

# ===== CELL 9: Memory Management =====
# If you need to free up memory
import gc

# Clear CUDA cache
torch.cuda.empty_cache()
gc.collect()

print("Memory cleared!")

# ===== CELL 10: CUAD-Specific Generation Functions =====
# Specialized functions for legal contract analysis

def analyze_contract_clause(contract_text, question):
    """Analyze a specific contract clause with a question"""
    prompt = f"{contract_text}\n\nQuestion: {question}\nAnswer:"
    return generate_text(prompt, temperature=0.1)

def extract_key_terms(contract_text):
    """Extract key terms from a contract"""
    prompt = f"{contract_text}\n\nQuestion: What are the key terms and provisions in this contract?\nAnswer:"
    return generate_text(prompt, temperature=0.1)

def identify_risks(contract_text, party="Vendor"):
    """Identify potential risks for a specific party"""
    prompt = f"{contract_text}\n\nQuestion: Identify any potentially unfavorable or risky clauses for the {party}.\nAnswer:"
    return generate_text(prompt, temperature=0.1)

def compare_clauses(clause_a, clause_b):
    """Compare two contract clauses"""
    prompt = f"Clause A: {clause_a}\nClause B: {clause_b}\n\nQuestion: Compare these two clauses and explain the key differences.\nAnswer:"
    return generate_text(prompt, temperature=0.1)

# Example usage
print("="*70)
print("CUAD ANALYSIS FUNCTIONS - EXAMPLES")
print("="*70)

# Example 1: Analyze specific clause
example_contract = """Contract: Service Agreement
Section 5: The Client shall pay $5,000 monthly, due within 15 days of invoice."""

print("\nExample 1: Clause Analysis")
print("-"*70)
result = analyze_contract_clause(example_contract, "What are the payment terms?")
print(f"Result:\n{result}\n")

# Example 2: Extract key terms
print("\nExample 2: Key Terms Extraction")
print("-"*70)
full_contract = """Contract: Employment Agreement
Section 3: Annual salary of $100,000, paid bi-weekly.
Section 7: 2 weeks vacation per year.
Section 9: 90-day notice required for termination."""

result = extract_key_terms(full_contract)
print(f"Result:\n{result}\n")

print("="*70)
print("✅ Ready for CUAD Legal Contract Analysis!")
print("="*70)
