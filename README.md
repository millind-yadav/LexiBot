# ⚖️ LexiBot - AI-Powered Legal Assistant

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![React](https://img.shields.io/badge/react-18%2B-cyan)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-green)
![Ollama](https://img.shields.io/badge/Ollama-Llama3-orange)

**LexiBot** is an advanced, local-first legal AI agent designed to assist with contract analysis, clause extraction, and risk assessment. Built with a privacy-centric architecture, it processes sensitive legal documents entirely within your local environment using **Retrieval-Augmented Generation (RAG)** and **Agentic Workflows**.

Unlike standard chatbots, LexiBot uses a **Planning Engine** to break down complex legal queries into executable steps, ensuring accurate and cited answers from your documents.

![LexiBot Interface](git assets/Screenshot 2025-12-05 at 15.23.47.png)

---

## 🚀 Key Features

- **📄 Document Analysis:** Upload PDF or DOCX contracts. LexiBot parses them client-side for maximum privacy.
- **🔍 Semantic Clause Retrieval:** Find specific clauses (e.g., "Termination", "Indemnity") even if the exact keywords aren't used, thanks to vector search and synonym expansion.
- **🧠 Stateful Conversations:** The agent remembers context from previous turns, allowing for natural follow-up questions (e.g., "Explain that clause further").
- **🤖 Agentic Reasoning:** The backend uses a planning engine to orchestrate tools:
    - *Clause Retrieval Tool* (Semantic Search)
    - *Contract QA Tool* (RAG-based answering)
    - *Structured Data Extractor* (Entity extraction)
- **🔒 Privacy-First:** Runs 100% locally using **Ollama** for the LLM and **FAISS** for vector storage. No data leaves your machine.
- **💬 Modern UI:** A responsive React interface with streaming responses, file management, and markdown support.

---

## 🛠️ Tech Stack

### Backend
- **Framework:** FastAPI
- **LLM Orchestration:** LangChain
- **Vector Store:** FAISS (Facebook AI Similarity Search)
- **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
- **LLM Runtime:** Ollama (Llama 3)
- **Agent Engine:** Custom `LegalPlanningEngine` & `AgentExecutor`

### Frontend
- **Framework:** React (Vite)
- **Styling:** Tailwind CSS
- **Icons:** Lucide React
- **Document Parsing:** `pdfjs-dist` (PDF), `mammoth` (DOCX)

---

## 🏗️ Architecture

```mermaid
graph TD
    User[User] -->|Uploads Doc| Frontend[React Frontend]
    Frontend -->|Parses Text| ClientParser[Client-Side Parser]
    ClientParser -->|Raw Text| Frontend
    Frontend -->|Query + History + Context| Backend[FastAPI Backend]
    
    subgraph "Backend Agent"
        Backend -->|Request| Agent[Agent Executor]
        Agent -->|Plan| Planner[Planning Engine]
        Planner -->|Steps| Agent
        
        Agent -->|Execute| Tools[Tool Registry]
        
        subgraph "RAG Pipeline"
            Tools -->|Clause Retrieval| RAG[RAG Service]
            RAG -->|Embed| HF[HuggingFace Embeddings]
            RAG -->|Search| FAISS[FAISS Vector Store]
        end
        
        Tools -->|Generate Answer| Ollama[Ollama (Llama 3)]
    end
    
    Agent -->|Final Response| Backend
    Backend -->|JSON| Frontend
```

---

## ⚡ Getting Started

### Prerequisites
- **Python 3.10+**
- **Node.js 18+** (and `pnpm`)
- **Ollama** installed and running (`ollama serve`)

### 1. Setup Ollama
Ensure Ollama is running and pull the Llama 3 model:
```bash
ollama pull llama3
```

### 2. Backend Setup
Navigate to the project root:
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
cd backend
pip install -r requirements.txt
# Install additional RAG dependencies
pip install langchain-community langchain-huggingface faiss-cpu

# Start the API server
python3 -m uvicorn app.main:app --reload --port 8002
```
*The backend will run on `http://localhost:8002`.*

### 3. Frontend Setup
Open a new terminal:
```bash
cd frontend

# Install dependencies
pnpm install

# Start the development server
pnpm dev
```
*The frontend will run on `http://localhost:5173`.*

### 4. Docker Setup (Optional)
If you prefer to run everything in containers:
```bash
# Build and start services
docker-compose up --build
```
*The app will be available at `http://localhost:3000`.*

---

## 📂 Project Structure

```
LexiBot/
├── agent/                  # Core agent logic and planning engine
│   └── core/
│       └── planning_engine.py
├── backend/                # FastAPI application
│   ├── app/
│   │   ├── agent/          # Agent executor and prompt templates
│   │   ├── core/           # RAG service and config
│   │   ├── llm/            # Ollama client wrapper
│   │   ├── tools/          # Tools (Clause Retrieval, Contract QA)
│   │   └── main.py         # API Entrypoint
│   └── requirements.txt
├── frontend/               # React application
│   ├── src/
│   │   ├── components/     # ChatInterface and UI components
│   │   └── ...
│   └── package.json
└── README.md
```

---

## 💡 Usage Guide

1.  **Open the App:** Go to `http://localhost:5173`.
2.  **Upload a Contract:** Click the paperclip icon and select a PDF or DOCX file.
3.  **Ask Questions:**
    -   *"Explain the termination conditions."*
    -   *"What are the tenant's liabilities?"*
    -   *"Are there any clauses regarding viewings?"*
4.  **Follow Up:**
    -   *"Explain that clause in simpler terms."*
    -   *"What are the risks associated with it?"*

---

## 🔧 Configuration

- **Backend Port:** Defaults to `8002`. Configurable in `backend/app/main.py`.
- **LLM Model:** Defaults to `llama3`. Configurable in `backend/app/core/config.py` or via `OLLAMA_MODEL` env var.
- **RAG Settings:** Chunk size and overlap can be tuned in `backend/app/core/rag.py`.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.
