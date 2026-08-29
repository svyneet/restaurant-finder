"""Central configuration for the restaurant-finder multi-agent system."""
from __future__ import annotations

import os
from pathlib import Path

import truststore

truststore.inject_into_ssl()  # use macOS/OS trust store, fixes SSL verify fails on some Python builds

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directory containing one or more scraped dataset JSON files. All *.json
# files here are loaded and merged (deduped by reviewId) - drop in more
# scraper exports to grow coverage without touching any code.
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data/districtwise"))

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

# LLM provider for the researcher agent: "ollama" (local, default), "anthropic",
# "github" (GitHub Models - free/rate-limited access to GPT-4o etc. via a
# GitHub personal access token with "models: read" permission. See
# https://docs.github.com/en/github-models), or "huggingface" (HF Inference
# Providers - free-tier, rate-limited access to hosted open models via a
# Hugging Face access token).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip() or None
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

GITHUB_MODELS_TOKEN = os.getenv("GITHUB_MODELS_TOKEN", "").strip() or None
GITHUB_MODELS_BASE_URL = os.getenv("GITHUB_MODELS_BASE_URL", "https://models.github.ai/inference")
GITHUB_MODEL = os.getenv("GITHUB_MODEL", "openai/gpt-4o-mini")

HF_TOKEN = os.getenv("HF_TOKEN", "").strip() or None
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct")

# Maps tools use free OpenStreetMap-based services (Nominatim/Overpass/OSRM) -
# no API key required. Set a descriptive contact per Nominatim's usage policy.
NOMINATIM_USER_AGENT = os.getenv(
    "NOMINATIM_USER_AGENT", "restaurant-finder-agent/1.0 (local hobby project)"
)

# Max researcher<->tool round trips before forcing a final answer.
MAX_AGENT_STEPS = 6

# Minimum cosine-similarity score for a search_reviews hit to be returned.
# Below this, a vector-embedding match is too weak to be useful evidence.
MIN_SEARCH_SIMILARITY = 0.35

# Minimum NLI entailment confidence for a cited review to be considered
# "grounded" in verify_quote.
MIN_ENTAILMENT_CONFIDENCE = 0.5

# Optional MLflow tracking server URI. When unset, traces are logged to a
# local SQLite db at ./mlflow.db (view with
# `mlflow ui --backend-store-uri sqlite:///mlflow.db`).
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "").strip() or None
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "restaurant-finder")
