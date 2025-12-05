
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))
sys.path.append(os.path.join(os.getcwd(), 'backend/app'))

from app.core.rag import RAGService
from langchain_community.document_loaders import PyPDFLoader

def test_rag():
    print("Loading PDF...")
    loader = PyPDFLoader('/Users/milindyadav/Downloads/contract.pdf')
    docs = loader.load()
    full_text = "\n".join([d.page_content for d in docs])
    print(f"Loaded {len(full_text)} characters.")

    print("Initializing RAG Service...")
    rag = RAGService()
    
    print("Creating Index...")
    # Simulate the context structure passed from frontend
    documents = [{"text": full_text, "metadata": {"source": "contract.pdf"}}]
    vectorstore = rag.create_index(documents)
    
    query = "Explain the termination conditions ending break clause notice period quit surrender"
    print(f"Querying: {query}")
    
    results = rag.query(vectorstore, query, k=5)
    
    print(f"\nFound {len(results)} results:")
    for i, res in enumerate(results):
        print(f"\n--- Result {i+1} (Score: {res['score']}) ---")
        print(res['text'][:200] + "...")
        if "8.1" in res['text'] or "Ending Tenancy" in res['text']:
            print(">>> FOUND TARGET CLAUSE! <<<")

if __name__ == "__main__":
    test_rag()
