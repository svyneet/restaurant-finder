# Quickstart

## 1. Set up the environment

```bash
cd restaurant-finder
poetry install   # creates an in-project .venv (any Python 3.10+ works)
```

## 2. Choose an LLM provider

Create a `.env` file in the project root (not committed) with one of the following:

**Option A — Local, free, no key (default)**
```
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1
```
Requires [Ollama](https://ollama.com/download) installed and running, with a tool-calling model pulled:
```bash
ollama pull llama3.1
```

**Option B — GitHub Models (free/rate-limited GPT-4o access)**
```
LLM_PROVIDER=github
GITHUB_MODELS_TOKEN=github_pat_...
GITHUB_MODEL=openai/gpt-4o-mini
```
Get a token: [github.com/settings/tokens](https://github.com/settings/tokens) → generate a **fine-grained token** → no repo access needed → under "Account permissions" grant **"Models: Read-only"**.

**Option C — Anthropic Claude (paid)**
```
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-5
```

> If you hit connection/certificate errors on macOS (common on some sandboxed/managed Python builds), this project already installs and injects `truststore` in `src/config.py` to use the OS trust store. `truststore` is a dependency in `pyproject.toml`, so `poetry install` covers it.

## 3. Run it

**Interactive terminal chat:**
```bash
.venv/bin/python main.py
```

**Web UI (recommended):**

Start the FastAPI backend in one terminal:
```bash
.venv/bin/uvicorn backend.main:app --reload --port 8100
```

Start the SvelteKit frontend in another terminal (first time only: `npm install` in `frontend/`):
```bash
cd frontend
npm install   # first time only
npm run dev
```
Open http://localhost:5173 (Vite proxies `/api` requests to the backend on port 8100). Try:
- "What is the best place for sushi in Berlin?" — should refuse, no sushi in dataset
- "Where's good for Middle Eastern or halal food?" — should cite real reviews
- "How far is Byblos from Alexanderplatz?" — tests the free maps distance tool
- "Is AMRIT open now?" — tests the free maps opening-hours tool

**Run the eval harness:**
```bash
.venv/bin/python run_eval.py
```
Prints pass/fail for each adversarial + normal test question, checking both refusal behavior and citation grounding. If `ANTHROPIC_API_KEY` is set (see Option C above), it also prints LLM-judge faithfulness/answer-relevancy scores per question (via `src/eval/llm_judge.py`) — otherwise those scores are skipped.

## 4. View MLflow traces

Every run (terminal chat, web UI backend, and the eval harness) automatically logs LLM traces via MLflow — no extra setup needed. By default traces go to a local SQLite db at the project root (`mlflow.db`); set `MLFLOW_TRACKING_URI` / `MLFLOW_EXPERIMENT_NAME` in `.env` to point elsewhere.

Launch the MLflow UI to inspect traces:
```bash
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open http://localhost:5000 and select the `restaurant-finder` experiment to see spans for each chat request / eval question, including the underlying Anthropic/OpenAI-compatible LLM calls (see [src/observability.py](src/observability.py)).

## 5. If you change `.env` while the backend is running

The FastAPI backend creates its `Coordinator` once at startup (see `backend/main.py`'s lifespan handler). Env var changes only take effect after a restart:

```bash
# find and kill the old process
lsof -ti:8100 | xargs kill -9
# relaunch
.venv/bin/uvicorn backend.main:app --reload --port 8100
```

## 6. Add more restaurants

Drop additional scraper export `.json` files into `data/`. They're automatically merged and deduplicated by `reviewId` on next run — no code changes needed (see `src/data_loader.py`).