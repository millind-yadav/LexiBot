import logging
from typing import List, Dict, Any, Optional

# LangChain Imports
# Ensure you have installed: langchain-huggingface, langchain-community, faiss-cpu
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
except ImportError as e:
    raise ImportError(
        "Missing RAG dependencies. Please install: "
        "pip install langchain-huggingface langchain-community faiss-cpu"
    ) from e

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGService:
    """
    Service for creating in-memory vector indices and performing semantic search.
    """
    def __init__(self):
        logger.info("Initializing RAG Service and loading Embedding Model...")
        try:
            # We use a lightweight local model (all-MiniLM-L6-v2) to avoid API costs/latency
            # This runs on CPU perfectly fine.
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,   # Good size for legal paragraphs
                chunk_overlap=200, # Overlap helps maintain context between chunks
                length_function=len,
            )
            logger.info("Embedding Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise e

    def create_index(self, documents: List[Dict[str, Any]]) -> Optional[FAISS]:
        """
        Creates a FAISS index from a list of raw document dictionaries.
        
        Args:
            documents: List of dicts, e.g. [{"text": "...", "metadata": {...}}]
            
        Returns:
            FAISS vectorstore or None if input is empty.
        """
        if not documents:
            logger.warning("create_index called with no documents.")
            return None

        # 1. Convert raw dicts into LangChain Documents
        langchain_docs = []
        for doc in documents:
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            
            if not text.strip():
                continue

            # Split huge texts into chunks so we don't exceed token limits
            chunks = self.text_splitter.split_text(text)
            for chunk in chunks:
                langchain_docs.append(Document(page_content=chunk, metadata=metadata))
        
        if not langchain_docs:
            logger.warning("No valid text found in documents to index.")
            return None

        # 2. Create Vector Store (This performs the embedding)
        try:
            logger.info(f"Embedding {len(langchain_docs)} document chunks...")
            vectorstore = FAISS.from_documents(langchain_docs, self.embeddings)
            return vectorstore
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            return None

    def query(self, vectorstore: FAISS, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches the provided vectorstore for the query.
        """
        if not vectorstore:
            logger.warning("Query attempted on None vectorstore.")
            return []
        
        try:
            # similarity_search_with_score returns L2 distance (lower is better) by default for FAISS
            results = vectorstore.similarity_search_with_score(query, k=k)
            
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score) 
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

# =========================================================
# SINGLETON INSTANCE
# This allows 'from ..core.rag import rag_service' to work
# without reloading the model every time.
# =========================================================
rag_service = RAGService()