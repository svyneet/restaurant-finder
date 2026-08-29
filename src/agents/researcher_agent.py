"""The Researcher agent: a pydantic-ai Agent wired to in-process review and
maps tool functions (see tools.py). Replaces the previous hand-rolled
mcp_client.MCPToolHub + agents/llm.py provider-switch code, and the MCP/stdio
tool servers that preceded these plain-function tools.
"""
from __future__ import annotations

import json
import logging
import re
from textwrap import dedent
from typing import Any

from pydantic import ValidationError
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.output import TextOutput
from pydantic_ai.providers.github import GitHubProvider
from pydantic_ai.providers.ollama import OllamaProvider

from .. import config
from .cards import RESEARCHER_CARD
from .models import ResearchAnswer
from .prompts import RESEARCHER_SYSTEM_PROMPT
from .tools import (
    ResearchDeps,
    get_place_address,
    get_place_stats,
    list_places,
    search_reviews,
)

logger = logging.getLogger(__name__)

# pydantic-ai's built-in PromptedOutput/NativeOutput both hand a *raw* JSON
# Schema document to the model (which itself has a top-level "properties"
# key) and validate the model's text directly against it. Against llama3.1
# that produces two separate failure modes we hit in practice: (1) the model
# echoes the schema's own shape back -- nesting the answer under a literal
# "properties" key, stringifying the recommendations array -- instead of
# producing an instance of it, and (2) it occasionally emits a stray space
# inside a JSON key (e.g. '"text "' instead of '"text"', a tokenizer
# artifact), which fails Pydantic's exact key matching even though the
# answer is otherwise a well-formed, correctly-shaped instance. Neither is
# something pydantic-ai's schema-in-prompt templating can route around, so
# instead: TextOutput hands us the model's raw text and we drive our own
# parse -> clean -> validate pipeline below, raising ModelRetry (which
# pydantic-ai's existing `retries=3` already re-prompts on) only for
# failures that are genuinely unrecoverable without the model's help.
_RESEARCH_ANSWER_JSON_INSTRUCTIONS_TEMPLATE = dedent(
    """\
    Always respond with a single JSON object that is an INSTANCE of the schema below -- never the schema
    itself. Do not include "properties", "type", "required", "$defs", or "title" keys in your response;
    those describe the shape of the answer, they are not fields of the answer.

    Schema:
    {schema}

    Example of a correctly-shaped response (this is illustrative only -- use the real place/claim data
    and reviewId aliases from your own tool calls, never these example values):
    {{"recommendations": [{{"place_name": "Example Place", "claims": [{{"text": "Example claim text.", \
"review_id": "R1"}}], "rating": 4.5, "address": "Example Str. 1, 10115 Berlin, Germany"}}], "refusal": null}}

    Don't include any text or Markdown fencing before or after the JSON object.
    """
)

_RESEARCH_ANSWER_JSON_INSTRUCTIONS = _RESEARCH_ANSWER_JSON_INSTRUCTIONS_TEMPLATE.format(
    schema=json.dumps(ResearchAnswer.model_json_schema())
)

_CODE_FENCE_RE = re.compile(r"^```\w*\n|\n```$")


def _strip_key_whitespace(value: Any) -> Any:
    """Recursively strip leading/trailing whitespace from dict keys. A
    stray space inside a key (see module docstring) is a tokenizer
    artifact, not a semantic difference from the intended key -- cleaning
    it before validation absorbs that without weakening any real
    validation, since every value is left untouched."""
    if isinstance(value, dict):
        return {k.strip(): _strip_key_whitespace(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_strip_key_whitespace(v) for v in value]
    return value


def _parse_research_answer(text: str) -> ResearchAnswer:
    cleaned = _CODE_FENCE_RE.sub("", text.strip()).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ModelRetry(
            f"Your response wasn't valid JSON ({e}). Respond with a single JSON object only -- "
            "no prose, no Markdown code fences."
        ) from e

    data = _strip_key_whitespace(data)

    try:
        return ResearchAnswer.model_validate(data)
    except ValidationError as e:
        raise ModelRetry(
            f"Your JSON didn't match the required shape:\n{e}\n\n{_RESEARCH_ANSWER_JSON_INSTRUCTIONS}"
        ) from e


def _ollama_model() -> OpenAIChatModel:
    return OpenAIChatModel(
        config.OLLAMA_MODEL,
        provider=OllamaProvider(base_url="http://localhost:11434/v1"),
    )


def _build_model() -> Any:
    """Translate config.LLM_PROVIDER into a pydantic-ai model, using the
    built-in providers for ollama/anthropic/github/huggingface so no
    hand-written request/response translation is needed per provider.
    """
    if config.LLM_PROVIDER == "huggingface":
        if not config.HF_TOKEN:
            logger.warning("LLM_PROVIDER=huggingface but HF_TOKEN is not set; falling back to ollama.")
            return _ollama_model()
        from pydantic_ai.models.huggingface import HuggingFaceModel
        from pydantic_ai.providers.huggingface import HuggingFaceProvider

        return HuggingFaceModel(
            config.HF_MODEL,
            provider=HuggingFaceProvider(api_key=config.HF_TOKEN, provider_name="auto"),
        )

    if config.LLM_PROVIDER == "anthropic":
        if not config.ANTHROPIC_API_KEY:
            logger.warning("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set; falling back to ollama.")
            return _ollama_model()
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        return AnthropicModel(
            config.ANTHROPIC_MODEL,
            provider=AnthropicProvider(api_key=config.ANTHROPIC_API_KEY),
        )

    if config.LLM_PROVIDER == "github":
        if not config.GITHUB_MODELS_TOKEN:
            logger.warning(
                "LLM_PROVIDER=github but GITHUB_MODELS_TOKEN is not set; falling back to ollama."
            )
            return _ollama_model()
        return OpenAIChatModel(config.GITHUB_MODEL, provider=GitHubProvider(api_key=config.GITHUB_MODELS_TOKEN))

    return _ollama_model()


researcher_agent: Agent[ResearchDeps, ResearchAnswer] = Agent(
    _build_model(),
    deps_type=ResearchDeps,
    system_prompt=RESEARCHER_SYSTEM_PROMPT,
    instructions=_RESEARCH_ANSWER_JSON_INSTRUCTIONS,
    # TextOutput + our own parse/clean/validate pipeline above, not
    # NativeOutput/PromptedOutput -- see the module docstring for why:
    # NativeOutput's response_format is sent on every turn (not just the
    # final one) and Ollama's grammar-constrained decoding then competes
    # with tool_choice, so llama3.1 reliably takes the trivially-schema-
    # valid empty answer instead of ever calling a tool. PromptedOutput
    # avoids that but hands the model a raw JSON Schema to imitate, which
    # it sometimes echoes structurally (nesting under "properties") instead
    # of instancing, and pydantic-ai validates its text directly with no
    # opportunity to absorb tokenizer-level quirks (stray whitespace in
    # keys) before that validation runs.
    output_type=TextOutput(_parse_research_answer),
    retries=3,
    tools=[
        list_places,
        search_reviews,
        get_place_stats,
        get_place_address,
    ],
)
researcher_agent.card = RESEARCHER_CARD  # type: ignore[attr-defined]
