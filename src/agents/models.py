"""Pydantic data models shared across the agent pipeline, replacing the
previous dict-shaped messages / plain dataclasses."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Citation(BaseModel):
    """One claim's citation, plus the result of running it through the
    deterministic verifier."""

    review_id: str
    claim: str
    grounded: bool
    confidence: float | None = None
    reason: str | None = None


class Claim(BaseModel):
    """One factual claim about a recommendation, backed by a reviewId that
    must have actually been returned by search_reviews in this run."""

    text: str
    review_id: str


class Recommendation(BaseModel):
    """One restaurant recommendation with its supporting claims."""

    place_name: str
    claims: list[Claim] = Field(default_factory=list)
    rating: float | None = None
    address: str | None = None

    @field_validator("claims")
    @classmethod
    def _drop_empty_claims(cls, claims: list[Claim]) -> list[Claim]:
        """Weaker models (esp. local ones) sometimes pad this list with
        blank placeholder entries to hit a perceived length; those carry no
        evidence and would break citation rendering, so drop them here
        rather than trusting every model to never emit them."""
        return [c for c in claims if c.text.strip() and c.review_id.strip()]


class ResearchAnswer(BaseModel):
    """Structured output type for researcher_agent. Exactly one of
    `recommendations` or `refusal` should be meaningfully populated."""

    recommendations: list[Recommendation] = Field(default_factory=list)
    refusal: str | None = None


class RunResult(BaseModel):
    """Result of one end-to-end Coordinator.run() call."""

    answer: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    refusal: str | None = None
    seen_review_ids: set[str] = Field(default_factory=set)
    citations: list[Citation] = Field(default_factory=list)
    revised: bool = False
    refused: bool = False
    tool_calls_made: list[str] = Field(default_factory=list)
