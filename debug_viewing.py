
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))
sys.path.append(os.path.join(os.getcwd(), 'backend/app'))

from app.core.rag import RAGService
from langchain_community.document_loaders import PyPDFLoader
from app.tools.clause_retrieval_tool import ClauseRetrievalTool

def test_viewing_rag():
    print("Loading PDF...")
    loader = PyPDFLoader('/Users/milindyadav/Downloads/contract.pdf')
    docs = loader.load()
    full_text = "\n".join([d.page_content for d in docs])
    
    print("Initializing RAG Service...")
    rag = RAGService()
    
    print("Creating Index...")
    documents = [{"text": full_text, "metadata": {"source": "contract.pdf"}}]
    vectorstore = rag.create_index(documents)
    
    # Simulate the tool's logic
    query = "Any clauses fir viewings ?"
    clause_types = []
    
    full_query = query or ""
    combined_text = (full_query + " " + " ".join(clause_types or [])).lower()
    
    print(f"Combined text for check: '{combined_text}'")
    
    if "viewing" in combined_text or "view" in combined_text:
        full_query += " show access inspect visit entry"
        print(">>> Added viewing synonyms")
    
    print(f"Final Query: {full_query}")
    
    results = rag.query(vectorstore, full_query, k=10)
    
    print(f"\nFound {len(results)} results:")
    for i, res in enumerate(results):
        print(f"\n--- Result {i+1} (Score: {res['score']}) ---")
        print(res['text'][:300] + "...")
        if "1.33" in res['text'] or "Permit Viewing" in res['text']:
            print(">>> FOUND CLAUSE 1.33! <<<")

if __name__ == "__main__":
    test_viewing_rag()
