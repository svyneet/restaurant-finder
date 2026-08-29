"""LLM-judge eval metrics (faithfulness + answer relevancy), in the spirit of
Ragas but implemented directly with pydantic-ai/Anthropic instead of the
`ragas` package.

`ragas` 0.4.3 cannot currently be installed alongside this project's
`langgraph`/`langchain-anthropic` versions: it eagerly imports
`langchain_community.chat_models.vertexai` (removed upstream) and only
resolves by downgrading `langchain-core` to 0.3.x, which breaks `langgraph`
(1.2.10, requires langchain-core>=1.4). See /memories/repo/restaurant-finder.md.

These metrics are supplementary to (not a replacement for) the deterministic
citation grounding already done in `orchestrator._verify_citations` -- they
catch things per-citation word-overlap can't, like a compound claim ("Korean
BBQ") stitched together from two individually-true citations.
"""
from __future__ import annotations

import logging

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from .. import config

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """\
You are a strict evaluator of a restaurant-recommendation agent's answer.

You will be given the user's question, the review snippets the agent actually \
retrieved (its only allowed evidence), and the agent's final answer.

Score two things, each from 0.0 to 1.0:

- faithfulness: what fraction of the answer's factual claims about specific \
  restaurants are actually supported by the retrieved review context? A claim \
  that combines two pieces of context into something neither one supports on \
  its own (e.g. citing a "Korean" review and a "BBQ" review to claim a \
  restaurant is "Korean BBQ") is NOT faithful, even if each citation is \
  individually accurate. If the answer contains no restaurant-specific claims \
  (e.g. it's a clean refusal), faithfulness is 1.0.
- answer_relevancy: does the answer actually address what the user asked, \
  without unnecessary padding or off-topic recommendations?
"""


class JudgeScore(BaseModel):
    faithfulness: float = Field(ge=0, le=1)
    answer_relevancy: float = Field(ge=0, le=1)
    rationale: str


_judge_agent: Agent[None, JudgeScore] | None = None


def _build_judge_model():
    """Prefer Anthropic (stronger judge model); fall back to local Ollama so
    the eval harness still works with no API key configured."""
    if config.ANTHROPIC_API_KEY:
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        return AnthropicModel(config.ANTHROPIC_MODEL, provider=AnthropicProvider(api_key=config.ANTHROPIC_API_KEY))

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.ollama import OllamaProvider

    return OpenAIChatModel(
        config.OLLAMA_MODEL,
        provider=OllamaProvider(base_url="http://localhost:11434/v1"),
    )


def _get_judge_agent() -> Agent[None, JudgeScore] | None:
    global _judge_agent
    if _judge_agent is None:
        model = _build_judge_model()
        if model is None:
            return None
        _judge_agent = Agent(model, system_prompt=JUDGE_SYSTEM_PROMPT, output_type=JudgeScore)
    return _judge_agent


async def judge_answer(query: str, answer: str, contexts: list[str]) -> JudgeScore | None:
    """Score an answer's faithfulness/relevancy. Returns None if the judge
    call itself fails (e.g. Ollama not running, billing/rate-limit errors),
    so the eval harness can degrade gracefully instead of crashing the whole
    run."""
    agent = _get_judge_agent()
    if agent is None:
        return None

    context_block = "\n\n".join(f"- {c}" for c in contexts) if contexts else "(no review context retrieved)"
    prompt = f"User question: {query}\n\nRetrieved review context:\n{context_block}\n\nAgent's answer:\n{answer}"
    try:
        result = await agent.run(prompt)
    except Exception as exc:  # noqa: BLE001 - judge is best-effort, never fail the eval run over it
        logger.error("LLM judge call failed, skipping score: %s", exc)
        return None
    return result.output
