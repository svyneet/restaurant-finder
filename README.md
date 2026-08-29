# Berlin Restaurant Concierge

A multi-agent, tool-using LLM system that answers questions about restaurants in Berlin, grounded in real scraped Google Maps review data. Built to demonstrate **retrieval-augmented generation (RAG) + agentic tool use + citation verification**, not just "call an LLM and hope."

## Why this exists

Generic chatbots will confidently answer "what's the best sushi place in Berlin?" even when they have no real data to back it up. This project is built around a specific guarantee: **the agent must retrieve real evidence before answering, cite the exact review it used, and every citation is independently verified** before being shown as trustworthy. If the dataset has no relevant restaurant, the agent should say so instead of guessing.

## Architecture

```
User question
     │
     ▼
Researcher (LLM + tool calling)  ──▶  reviews tools ──▶  Vector search (llama_index + local embeddings) over review dataset
     │                
     ▼
Draft answer with per-claim citations
     │
     ▼
Verifier (deterministic, no LLM) ──▶  checks every citation is a real review that was actually retrieved
     │
     ▼
Revision pass (if any citation fails) ──▶ final answer + grounding report
```

The Researcher → Verifier → Revision control flow (including bounded retry/nudge
stages) is implemented as a [LangGraph](https://langchain-ai.github.io/langgraph/)
`StateGraph` in `orchestrator.py`; pydantic-ai still owns the actual LLM/tool-calling
loop, LangGraph only orchestrates which stage runs next.

- **Dataset**: scraped Google Maps reviews (`data/*.json`), merged and deduplicated automatically. Currently ~26 Berlin-area restaurants.
- **Retrieval**: local semantic vector search (`src/rag/index.py`) — reviews are embedded with a local HuggingFace sentence-transformer (`BAAI/bge-small-en-v1.5`) and indexed with [llama_index](https://docs.llamaindex.ai/)'s `VectorStoreIndex`; the index is persisted to `data/.index_cache` and only rebuilt when the corpus changes. No external embeddings API, no cost.
- **Tools** (`src/agents/tools.py`), registered in-process on the Researcher agent:
  - reviews: `list_places`, `search_reviews` (vector search via `src/rag/index.py`), `get_place_stats` (deterministic aspect/sentiment via keyword counting), `verify_quote` (deterministic NLI entailment check)
  - maps: `get_place_address`
- **Agents** (`src/agents/`), built on [pydantic-ai](https://ai.pydantic.dev/):
     - `researcher_agent.py` — the Researcher: a `pydantic_ai.Agent` wired directly to the tool functions above, with the LLM provider (Ollama/Anthropic/GitHub Models) selected via pydantic-ai providers
  - `orchestrator.py` — `Coordinator`, a LangGraph `StateGraph` implementing the Researcher → Verifier → revision loop described above; the deterministic verifier calls `verify_quote` directly (no LLM)
  - `cards.py` — an `AgentCard` (name/description/capabilities/skills, A2A-protocol-flavored) attached to each agent for self-description/introspection
  - `models.py` — Pydantic models (`Citation`, `RunResult`) shared across the pipeline instead of ad-hoc dicts

## LLM providers

Set `LLM_PROVIDER` in `.env`:

| Provider | Config | Notes |
|---|---|---|
| `ollama` (default) | `OLLAMA_MODEL=llama3.1` | Local, free, no key needed. Requires [Ollama](https://ollama.com) running with a tool-calling model pulled (e.g. `ollama pull llama3.1`). Tool-calling reliability varies by model — `llama3.1` performed noticeably better than `mistral` in testing. |
| `anthropic` | `ANTHROPIC_API_KEY=...`, `ANTHROPIC_MODEL=claude-sonnet-4-5` | Paid, requires billing on your Anthropic account. |
| `github` | `GITHUB_MODELS_TOKEN=...`, `GITHUB_MODEL=openai/gpt-4o-mini` | Free/rate-limited access to GPT-4o family via [GitHub Models](https://docs.github.com/en/github-models), authenticated with a GitHub personal access token (`Models: Read-only` permission) — not your Copilot subscription directly. |

## Interfaces

- **CLI**: `python main.py` — interactive terminal chat
- **Web UI**: FastAPI backend (`backend/main.py`, SSE `/api/chat` endpoint) + SvelteKit frontend (`frontend/`) — chat interface with a per-answer grounding report (citation pass/fail)
- **Eval harness**: `python run_eval.py` — runs a fixed set of adversarial + normal questions (`src/eval/adversarial_questions.py`) and reports whether refusal behavior and deterministic citation grounding match expectations. If `ANTHROPIC_API_KEY` is set, it also scores each answer with an LLM-judge (`src/eval/llm_judge.py`, faithfulness + answer relevancy) as a supplementary check — a lightweight custom equivalent to [Ragas](https://docs.ragas.io/) metrics, since the `ragas` package itself can't currently be installed alongside this project's `langgraph` version.

## Documentation

- [QUICKSTART.md](QUICKSTART.md) — environment setup and LLM provider configuration.
- [docs/data-loader.md](docs/data-loader.md) — how the dataset loader (`src/data_loader.py`) parses, dedupes, and caches the scraped reviews.

## Known limitations

- The dataset only covers ~26 restaurants — it will (correctly) refuse to answer about cuisines it has no data for (e.g. sushi, ramen).
- Local models (especially smaller ones) sometimes skip tool calls entirely and answer from parametric knowledge instead — the orchestrator forces a retry when this happens, but a model can still fail to comply. The verifier catches resulting hallucinated citations, but forcing retrieval isn't 100% guaranteed with weak tool-calling models.
- Maps tools depend on free public services (Nominatim/Overpass/OSRM) which have usage limits and can occasionally be slow or unavailable — the tools report this explicitly rather than fabricating results.

See [QUICKSTART.md](QUICKSTART.md) to get running.
