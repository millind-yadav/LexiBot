**Repo Overview**
- Legal AI stack with a FastAPI agent backend (`backend/app`), a legacy-but-richer API surface in `api/main.py`, and a Next.js 15 front-end in `frontend`.
- Planning + execution pipeline lives in `agent/core/planning_engine.py` and `backend/app/agent/agent_executor.py`; tools in `backend/app/tools` operate on contract snippets provided via the request `context`.
- CUAD-derived corpora, processed splits, and vector store artifacts sit under `data/`; diagnostics in `analysis_results/` capture token length constraints to respect during training and inference.

**Backend Agent**
- `AgentExecutor.run` asks `LegalPlanningEngine` for a plan, then executes registered tools in order; the final answer is stitched via `build_summary_prompt` with no LLM call—additions to reasoning must modify that prompt.
- Plan templates wire in `contract_qa_tool`, `clause_retrieval_tool`, and `contract_comparison_tool`; register replacements or additions through `backend/app/tools/__init__.py`.
- Each tool expects `context` to expose `documents` or `contracts[].sections` dictionaries; missing inputs raise `ToolExecutionError`, so validate context in new flows.
- `api/main.py` still imports non-existent executor APIs (`set_model`, `execute_tool`); treat it as reference code until the executor grows parity.

**Frontend**
- `frontend/app` renders marketing + legal workflows; interactive analyzers live in `frontend/components/legal/*` and call server routes in `frontend/app/api/legal`.
- Those API routes proxy to the backend via `LEGAL_API_URL`/`LEGAL_API_TOKEN`; in dev, they return detailed mocks if the upstream 5xx’s.
- Tooling assumes `pnpm@9`: lint/format with `pnpm lint` or `pnpm format`, run e2e with `pnpm test` (Playwright + mocked backend) and start dev with `pnpm dev`.

**Data & Fine-Tuning**
- Prep CUAD data with `python scripts/data_processing/02_prepare_and_clean.py --input data/raw/CUADv1.json --output data/processed/train.jsonl --val_output data/processed/val.jsonl`; script downloads NLTK tokenizers and can spawn multiple processes.
- QLoRA training script (`scripts/fine_tuning/02_run_qlora_finetune.py`) loads datasets via Hugging Face, swaps in Unsloth adapters, and enforces aggressive chunking—verify GPU memory before bumping sequence length.
- Consult `analysis_results/dataset_analysis_report.md` for max token/char outliers prior to adjusting chunk sizes or batch params.

**Dev Workflows & Ops**
- Local backend: `uvicorn app.main:app --reload` from `backend/app`; Dockerfile targets CPU and mounts `data/` + `models/`—override `MODEL_PATH`/`VECTOR_STORE_PATH` envs per deployment.
- Frontend dev: `pnpm install && pnpm dev` (Next.js on port 3000); set `LEGAL_API_URL=http://localhost:8000/api/v1` in `.env.local` to reach the FastAPI agent.
 - Updated `docker-compose.yml` runs FastAPI + Next.js together (`docker-compose up --build`); share `LEGAL_API_TOKEN` across services and note `SKIP_DB_MIGRATE=1` skips Drizzle migrations inside containers.

**Conventions & Pitfalls**
- Keep tools deterministic and lightweight; executor loop is synchronous, so long operations will block the entire request.
- Extend `_KEYWORD_HINTS` in `clause_retrieval_tool.py` instead of hard-coding terms inside planners.
- Scripts assume write access for tokenizer downloads and cached chunking; prefer running outside read-only environments.
- `evaluation/` stubs exist; wire them up before promising automated regression checks.
