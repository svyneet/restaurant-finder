"""Observability setup (MLflow Tracing) for the restaurant-finder pipeline.

Call configure_observability() once per process, before making any LLM
calls. With no MLFLOW_TRACKING_URI set, traces are written to a local
SQLite db at the project root (view with `mlflow ui --backend-store-uri
sqlite:///mlflow.db`) -- no server or account required. MLflow 3.x's plain
filesystem store (the old `./mlruns` default) is in maintenance mode and
raises on use, so a local run needs a database backend even for solo/offline
use.

mlflow.anthropic.autolog() / mlflow.openai.autolog() patch the underlying
anthropic/openai SDK clients, so they capture LLM calls made by pydantic-ai's
AnthropicModel and OpenAIChatModel (which is what the ollama/github providers
use under the hood) without pydantic-ai-specific instrumentation.
"""
from __future__ import annotations

import mlflow
import mlflow.anthropic
import mlflow.openai

from . import config

_configured = False


def configure_observability() -> None:
    global _configured
    if _configured:
        return
    _configured = True

    tracking_uri = config.MLFLOW_TRACKING_URI or f"sqlite:///{config.PROJECT_ROOT / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    mlflow.anthropic.autolog()
    mlflow.openai.autolog()
