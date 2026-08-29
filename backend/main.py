"""FastAPI backend for the Berlin restaurant kiezscout, serving the Svelte
frontend over a Server-Sent Events (SSE) chat endpoint.

Run with:
    .venv/bin/uvicorn backend.main:app --reload --port 8100

The Coordinator is created once at startup and reused across requests since
FastAPI's event loop is already async (no thread/event-loop bridging needed,
unlike the old Streamlit app).
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.agents.orchestrator import Coordinator
from src.observability import configure_observability

# Human-readable labels for graph node names, shown as progress updates
# while the SSE stream is open (see Coordinator.stream()).
STAGE_LABELS = {
    "research": "Researching reviews...",
    "checkpoint": "Checking tool usage, search results, and confidence...",
    "nudge_missing_citations": "Answer was missing citations, requesting them...",
    "verify": "Verifying citations against the source reviews...",
    "strip_ungrounded": "Dropping any claims that failed verification...",
    "refuse": "No grounded evidence found...",
}

configure_observability()

coordinator: Coordinator | None = None


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global coordinator
    coordinator = Coordinator()
    yield
    coordinator = None


app = FastAPI(title="Berlin Restaurant Kiezscout API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/api/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    async def event_stream() -> AsyncIterator[str]:
        assert coordinator is not None
        with mlflow.start_span(name="chat_request") as span:
            span.set_inputs({"query": request.query})
            async for kind, payload in coordinator.stream(request.query):
                if kind == "status":
                    label = STAGE_LABELS.get(payload, payload)
                    yield _sse("status", {"stage": payload, "label": label})
                else:  # kind == "result"
                    yield _sse("result", payload.model_dump(mode="json"))

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok" if coordinator is not None else "unavailable"}
