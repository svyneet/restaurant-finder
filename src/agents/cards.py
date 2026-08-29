"""Agent cards: lightweight, A2A-protocol-flavored metadata describing what
each agent in the pipeline is and what it can do (capabilities + skills).

These aren't wired into a network transport -- they're plain Pydantic models
attached to each agent (`agent.card`) so the rest of the system (CLI, web UI,
eval harness, logs) can introspect an agent's identity and advertised skills
instead of that information living only in prose comments/docstrings.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Skill(BaseModel):
    """A single named capability an agent exposes, analogous to an A2A
    AgentCard skill entry."""

    id: str
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)


class Capability(BaseModel):
    """Coarse-grained capability flags for an agent."""

    tool_use: bool = True
    structured_output: bool = True
    streaming: bool = False


class AgentCard(BaseModel):
    """Self-describing metadata for one agent in the multi-agent system."""

    name: str
    description: str
    capabilities: Capability = Field(default_factory=Capability)
    skills: list[Skill] = Field(default_factory=list)


RESEARCHER_CARD = AgentCard(
    name="berlin-restaurant-researcher",
    description=(
        "Retrieves grounded evidence about Berlin restaurants from review"
    ),
    capabilities=Capability(tool_use=True, structured_output=True, streaming=False),
    skills=[
        Skill(
            id="search_reviews",
            name="Review search",
            description="Document search over scraped Google Maps review text.",
            tags=["retrieval", "reviews"],
        ),
        Skill(
            id="place_stats",
            name="Aspect sentiment stats",
            description="Deterministic keyword-based aspect/sentiment aggregation per place.",
            tags=["analytics", "reviews"],
        ),
        Skill(
            id="list_places",
            name="Place directory",
            description="Lists all restaurants covered by the dataset.",
            tags=["retrieval"],
        ),
    ],
)

VERIFIER_CARD = AgentCard(
    name="citation-verifier",
    description=(
        "Deterministic (non-LLM) grounding check: confirms every cited claim "
        "in a structured draft answer references a review that was actually "
        "retrieved and is supported by the underlying review text."
    ),
    capabilities=Capability(tool_use=True, structured_output=True, streaming=False),
    skills=[
        Skill(
            id="verify_quote",
            name="Quote verification",
            description="Word-overlap check between a claim and the source review text.",
            tags=["verification", "reviews"],
        ),
    ],
)
