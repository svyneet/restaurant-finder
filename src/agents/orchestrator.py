"""Multi-agent coordinator: Researcher (pydantic-ai Agent, tool-using ReAct
loop handled internally by pydantic-ai) -> Verifier (deterministic citation
grounding check, see cards.VERIFIER_CARD) -> deterministic strip of any
claim that fails grounding.

The staged retry/nudge logic that used to live in nested `while` loops is
expressed as a LangGraph `StateGraph`. The three simplest checks (tool-call,
empty-search, hedging) are structurally identical -- test a predicate, nudge
with a fixed message if it fails and a counter allows it, otherwise move on
(or refuse) -- so they're data-driven via the `CHECKS` rule table and a single
`_checkpoint` node instead of one node pair each. `_checkpoint` batches every
failing check in one pass into a single combined nudge, instead of nudging
one check at a time across multiple full researcher re-runs. Citation
verification stays as its own node because its condition depends on
citations computed at runtime, not a static predicate over existing state.
pydantic-ai still owns the actual LLM/tool-calling loop -- LangGraph only
orchestrates *which* nudge/verification step runs next.

The researcher agent returns a structured `ResearchAnswer` (see models.py)
instead of free prose with inline `[REVIEW:id]` tags, so citation
verification and stripping walk the `recommendations`/`claims` structure
directly instead of regex-scraping text.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TypedDict

import mlflow
from langgraph.graph import END, StateGraph
from mlflow.entities import SpanEvent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage, ToolCallPart, ToolReturnPart

from .cards import VERIFIER_CARD
from .models import Citation, Claim, Recommendation, ResearchAnswer, RunResult
from .researcher_agent import researcher_agent
from .tools import ResearchDeps, verify_quote

# Upper bound on how many *additional* full researcher invocations one query
# can trigger across every nudge path combined (checkpoint nudges +
# missing-citations nudge), regardless of how much budget any individual
# counter still has left. Bounds worst-case latency/cost per query.
TOTAL_RERUN_BUDGET = 4

NO_TOOL_CALL_NUDGE = (
    "You did not call any tools. You are not allowed to answer from general "
    "knowledge. Call search_reviews or list_places now to check what is "
    "actually in the dataset before answering."
)


# Deterministic fallback used when the researcher never retrieves any
# evidence at all, even after being nudged, or when every claim in its
# answer fails grounding. An answer with zero grounded claims cannot be
# trusted no matter how it reads, so we refuse outright instead of shipping
# untooled or unverifiable text.
UNGROUNDED_REFUSAL = (
    "I wasn't able to retrieve any grounded evidence from the review dataset "
    "for this question, so I can't answer reliably. This usually means the "
    "dataset doesn't cover what you're asking about."
)

# Used when the researcher model exhausts pydantic-ai's structured-output
# retries (raises UnexpectedModelBehavior) instead of ever producing a valid
# ResearchAnswer -- typically a weaker/local model spiraling into unstructured
# prose after repeated validation-error nudges. Without this, the exception
# propagates uncaught out of Coordinator.run/stream and crashes the request
# instead of degrading to a refusal like every other "can't answer reliably"
# path in this pipeline.
MODEL_FAILURE_REFUSAL = (
    "I ran into trouble producing a structured, well-formed answer for this "
    "question and can't reliably answer it right now. Please try rephrasing "
    "or asking again."
)

# Nudge used when the researcher did retrieve evidence (called search_reviews
# and got results) but produced zero claims. An answer with no claims is
# invisible to the citation verifier -- it would trivially "pass" with zero
# failed citations despite making unverifiable claims -- so we force the
# model to add citations (or admit it can't) before accepting the answer.
MISSING_CITATIONS_NUDGE = (
    "Your answer contains no claims with citations, but you did retrieve "
    "review evidence. Every restaurant-specific claim must be a `claims` "
    "entry with a reviewId you actually retrieved. Rewrite your answer with "
    "those citations now, or set `refusal` stating you don't have enough "
    "grounded evidence to answer."
)

# Nudge used when search_reviews was called but returned zero results. The
# model may try to answer anyway from general knowledge. Force it to explicitly
# acknowledge that the cuisine/restaurant type doesn't exist in the dataset.
EMPTY_SEARCH_RESULTS_NUDGE = (
    "Your search_reviews call returned zero results. This means the dataset "
    "does not contain any restaurants or reviews matching that request. You must "
    "explicitly state that you don't have data for this in `refusal`, rather than "
    "attempting to answer from general knowledge."
)

# Nudge used when the model's answer contains hedging language ("I couldn't
# determine", "might", "possibly", "unsure") that suggests it doesn't have a
# good match for what the user asked. Force it to be explicit about what's
# NOT in the dataset rather than providing uncertain recommendations.
UNCERTAIN_MATCH_NUDGE = (
    "Your answer contains uncertain language like 'couldn't determine', 'might', "
    "or similar hedging. This suggests the search results don't actually match "
    "what the user asked for. If you have zero genuine matches, set `refusal` to "
    "one or two sentences stating what's not covered in the dataset -- don't list "
    "or explain every near-miss restaurant you considered. If you DO have "
    "genuine matches, drop the hedging and state them plainly with citations; "
    "do not mention restaurants that don't fit the request at all."
)


def _draft_text(draft: ResearchAnswer) -> str:
    """Flatten a structured draft's claim text (+ refusal) into one string
    for the hedging-phrase heuristic to scan."""
    parts = [claim.text for rec in draft.recommendations for claim in rec.claims]
    if draft.refusal:
        parts.append(draft.refusal)
    return " ".join(parts)


def _flatten_claims(draft: ResearchAnswer) -> list[Claim]:
    return [claim for rec in draft.recommendations for claim in rec.claims]


def _render_prose(draft: ResearchAnswer) -> str:
    """Render a structured draft back into prose, for RunResult.answer --
    kept only so src/eval (which treats the answer as free text for the LLM
    judge and a refusal-phrase heuristic) keeps working unchanged."""
    if draft.refusal:
        return draft.refusal
    if not draft.recommendations:
        return UNGROUNDED_REFUSAL
    lines = []
    for rec in draft.recommendations:
        claim_text = " ".join(f"{c.text} [REVIEW:{c.review_id}]." for c in rec.claims)
        rating_text = f" Rated {rec.rating}/5." if rec.rating is not None else ""
        address_text = f" Address: {rec.address}." if rec.address else ""
        lines.append(f"{rec.place_name}\n{claim_text}{rating_text}{address_text}")
    return "\n\n".join(lines)


def _extract_tool_calls(messages: list[ModelMessage]) -> list[str]:
    """Walk pydantic-ai message history for tool names called, so the
    no-tool-call/empty-search checks can see what actually happened this
    run. (Which reviewIds were retrieved is tracked live on ResearchDeps
    instead -- see tools.ResearchDeps -- since search_reviews now hands the
    model a short local alias rather than the real reviewId.)"""
    tool_calls_made: list[str] = []
    for message in messages:
        for part in getattr(message, "parts", []):
            if isinstance(part, ToolCallPart):
                tool_calls_made.append(part.tool_name)
    return tool_calls_made


def _search_reviews_returned_empty(messages: list[ModelMessage]) -> bool:
    """Check if search_reviews was called but returned zero results."""
    for message in messages:
        for part in getattr(message, "parts", []):
            if isinstance(part, ToolReturnPart) and part.tool_name == "search_reviews":
                content = part.content
                # If search_reviews was called and returned an empty list, return True
                if isinstance(content, list) and len(content) == 0:
                    return True
    return False


def _answer_contains_uncertain_hedging(answer: str) -> bool:
    """Check if answer contains hedging language suggesting incomplete match."""
    uncertain_phrases = (
        "couldn't determine",
        "couldn't find",
        "couldn't verify",
        "can't determine",
        "can't verify",
        "might",
        "possibly",
        "unsure",
        "uncertain",
        "I'm not sure",
        "not sure if",
        "but I couldn't",
        "but i couldn't",
    )
    lower_answer = answer.lower()
    return any(phrase in lower_answer for phrase in uncertain_phrases)


class GraphState(TypedDict):
    """State threaded through the LangGraph nodes for one Coordinator.run().

    `stage` tracks which node should run immediately after "research"
    returns: "checkpoint" (run the CHECKS rule table) or "verify" (resume
    citation verification after a missing-citations nudge).
    """

    pending_input: str
    stage: str
    checkpoint_outcome: str
    history: list[ModelMessage] | None
    draft: ResearchAnswer
    deps: ResearchDeps
    tool_calls_made: list[str]
    citations: list[Citation]
    revised: bool
    refused: bool
    model_failure: bool
    no_tool_attempts: int
    empty_search_attempts: int
    hedging_attempts: int
    missing_citation_attempts: int
    total_reruns: int


@dataclass(frozen=True)
class CheckRule:
    """One entry in the CHECKS table: a predicate over GraphState, the nudge
    to send if it fails, which counter guards its retries, and what to do
    once that counter is exhausted."""

    name: str
    predicate: Callable[[GraphState], bool]
    nudge: Callable[[GraphState], str]
    counter_key: str
    limit: int
    on_exhausted: str  # "refuse" or "continue"


CHECKS: list[CheckRule] = [
    CheckRule(
        name="no_tool_call",
        predicate=lambda s: bool(s["tool_calls_made"]),
        nudge=lambda s: NO_TOOL_CALL_NUDGE,
        counter_key="no_tool_attempts",
        limit=2,
        on_exhausted="refuse",  # no grounded evidence was ever retrieved -- can't trust the answer
    ),
    CheckRule(
        name="empty_search",
        predicate=lambda s: not _search_reviews_returned_empty(s["history"]),
        nudge=lambda s: EMPTY_SEARCH_RESULTS_NUDGE,
        counter_key="empty_search_attempts",
        limit=2,
        on_exhausted="continue",  # falls through to the next check, doesn't refuse
    ),
    CheckRule(
        name="hedging",
        predicate=lambda s: not (
            s["deps"].review_aliases and _answer_contains_uncertain_hedging(_draft_text(s["draft"]))
        ),
        nudge=lambda s: UNCERTAIN_MATCH_NUDGE,
        counter_key="hedging_attempts",
        limit=2,
        on_exhausted="continue",
    ),
]


async def _run_researcher(state: GraphState) -> dict[str, Any]:
    try:
        result = await researcher_agent.run(
            state["pending_input"], message_history=state["history"], deps=state["deps"]
        )
    except UnexpectedModelBehavior:
        # The model exhausted its structured-output retries without ever
        # producing a valid ResearchAnswer (see MODEL_FAILURE_REFUSAL). Fall
        # back to a refusal instead of letting this propagate and crash the
        # run -- there's no valid `result.output` to recover here.
        return {
            "draft": ResearchAnswer(recommendations=[], refusal=MODEL_FAILURE_REFUSAL),
            "model_failure": True,
        }
    history = result.all_messages()
    return {
        "history": history,
        "draft": result.output,
        "deps": state["deps"],
        "tool_calls_made": _extract_tool_calls(history),
    }


def _route_after_research(state: GraphState) -> str:
    if state["model_failure"]:
        return "refuse"
    return "checkpoint" if state["stage"] == "checkpoint" else "verify"


async def _checkpoint(state: GraphState) -> dict[str, Any]:
    """Walk the full CHECKS table in one pass, collecting a nudge from
    *every* failing rule whose counter still allows a retry (instead of
    stopping at the first failure), so a query that fails several checks at
    once is fixed with a single combined nudge and one researcher re-run
    instead of one re-run per check. A rule whose counter is exhausted
    either refuses immediately (if configured to) or is skipped. Reaching
    the end with no nudges collected means every check passed (or was
    exhausted without refusing), so the run proceeds to verification."""
    nudges: list[str] = []
    counter_updates: dict[str, int] = {}
    for rule in CHECKS:
        if rule.predicate(state):
            continue
        attempts = state[rule.counter_key]
        if attempts < rule.limit:
            nudges.append(rule.nudge(state))
            counter_updates[rule.counter_key] = attempts + 1
            continue
        if rule.on_exhausted == "refuse":
            return {"checkpoint_outcome": "refuse"}
        # on_exhausted == "continue": this rule is done nudging, move on.

    if not nudges:
        return {"stage": "verify", "checkpoint_outcome": "pass"}

    if state["total_reruns"] >= TOTAL_RERUN_BUDGET:
        return {"checkpoint_outcome": "refuse"}

    result: dict[str, Any] = {
        "pending_input": "\n\n".join(nudges),
        "stage": "checkpoint",
        "checkpoint_outcome": "nudge",
        "total_reruns": state["total_reruns"] + 1,
    }
    result.update(counter_updates)
    return result


def _route_checkpoint(state: GraphState) -> str:
    outcome = state["checkpoint_outcome"]
    if outcome == "nudge":
        return "research"
    if outcome == "refuse":
        return "refuse"
    return "verify"


async def _refuse(state: GraphState) -> dict[str, Any]:
    if state["model_failure"]:
        # Keep the MODEL_FAILURE_REFUSAL draft _run_researcher already built;
        # it's a more accurate refusal reason than the generic ungrounded one.
        return {"refused": True}
    return {"refused": True, "draft": ResearchAnswer(recommendations=[], refusal=UNGROUNDED_REFUSAL)}


async def _verify_citations(draft: ResearchAnswer, deps: ResearchDeps) -> list[Citation]:
    """Resolve each claim's alias (see tools.ResearchDeps) back to the real
    reviewId before grounding it -- the model only ever sees/cites the short
    alias, so this is the one place alias and real ID meet. An alias that
    doesn't resolve (never issued this run, or garbled beyond recognition)
    is exposed as-is on the Citation for debuggability and fails grounding,
    same as an unrecognized reviewId did before aliasing existed."""
    citations: list[Citation] = []
    for claim in _flatten_claims(draft):
        real_id = deps.resolve(claim.review_id)
        if real_id is None:
            citations.append(
                Citation(
                    review_id=claim.review_id,
                    claim=claim.text,
                    grounded=False,
                    reason="reviewId was never returned by search_reviews in this session",
                )
            )
            continue

        tool_result = verify_quote(real_id, claim.text)
        grounded = bool(tool_result.get("grounded")) if isinstance(tool_result, dict) else False
        citations.append(
            Citation(
                review_id=real_id,
                claim=claim.text,
                grounded=grounded,
                confidence=tool_result.get("confidence") if isinstance(tool_result, dict) else None,
            )
        )
    return citations


async def _verify(state: GraphState) -> dict[str, Any]:
    citations = await _verify_citations(state["draft"], state["deps"])
    return {"citations": citations}


def _route_verify(state: GraphState) -> str:
    if not state["citations"] and state["deps"].review_aliases:
        if state["missing_citation_attempts"] < 1 and state["total_reruns"] < TOTAL_RERUN_BUDGET:
            return "nudge_missing_citations"
        # Nudge already spent (or the global budget is exhausted) and the
        # model still produced zero citations -- an empty citations list is
        # vacuously "no failed citations" to _route_failed_citations, so
        # without this branch an entirely unverified, possibly off-topic
        # answer would sail through to END.
        return "refuse"
    return "check_failed_citations"


async def _nudge_missing_citations(state: GraphState) -> dict[str, Any]:
    return {
        "pending_input": MISSING_CITATIONS_NUDGE,
        "stage": "verify",
        "missing_citation_attempts": state["missing_citation_attempts"] + 1,
        "total_reruns": state["total_reruns"] + 1,
    }


async def _noop(_: GraphState) -> dict[str, Any]:
    return {}


def _route_failed_citations(state: GraphState) -> str:
    if [c for c in state["citations"] if not c.grounded]:
        return "strip_ungrounded"
    return END


async def _strip_ungrounded(state: GraphState) -> dict[str, Any]:
    """Deterministically drop only the claims that failed grounding instead
    of spending another LLM call to rewrite the answer -- stripping can only
    remove citations, never add ungrounded ones back, so no re-verification
    is needed afterward."""
    draft = state["draft"]
    citations = state["citations"]
    flat_claims = _flatten_claims(draft)
    # Claims still carry the model-facing alias (see tools.ResearchDeps); the
    # citation built during _verify already resolved it to the real
    # reviewId, so use that -- not the original claim -- as the source of
    # truth for what gets kept.
    resolved_claims = {
        id(claim): Claim(text=citation.claim, review_id=citation.review_id)
        for claim, citation in zip(flat_claims, citations)
        if citation.grounded
    }

    kept_recommendations: list[Recommendation] = []
    for rec in draft.recommendations:
        kept_claims = [resolved_claims[id(c)] for c in rec.claims if id(c) in resolved_claims]
        if kept_claims:
            kept_recommendations.append(rec.model_copy(update={"claims": kept_claims}))

    kept_citations = [c for c in citations if c.grounded]

    if not kept_recommendations:
        return {
            "draft": ResearchAnswer(recommendations=[], refusal=UNGROUNDED_REFUSAL),
            "citations": kept_citations,
            "refused": True,
        }

    return {
        "draft": ResearchAnswer(recommendations=kept_recommendations, refusal=None),
        "citations": kept_citations,
        "revised": True,
    }


def _build_graph() -> Any:
    graph = StateGraph(GraphState)

    graph.add_node("research", _run_researcher)
    graph.add_node("checkpoint", _checkpoint)
    graph.add_node("refuse", _refuse)
    graph.add_node("verify", _verify)
    graph.add_node("nudge_missing_citations", _nudge_missing_citations)
    graph.add_node("check_failed_citations", _noop)
    graph.add_node("strip_ungrounded", _strip_ungrounded)

    graph.set_entry_point("research")

    graph.add_conditional_edges(
        "research",
        _route_after_research,
        {"checkpoint": "checkpoint", "verify": "verify", "refuse": "refuse"},
    )
    graph.add_conditional_edges(
        "checkpoint",
        _route_checkpoint,
        {"research": "research", "refuse": "refuse", "verify": "verify"},
    )
    graph.add_edge("refuse", END)
    graph.add_conditional_edges(
        "verify",
        _route_verify,
        {
            "nudge_missing_citations": "nudge_missing_citations",
            "check_failed_citations": "check_failed_citations",
            "refuse": "refuse",
        },
    )
    graph.add_edge("nudge_missing_citations", "research")
    graph.add_conditional_edges(
        "check_failed_citations", _route_failed_citations, {"strip_ungrounded": "strip_ungrounded", END: END}
    )
    graph.add_edge("strip_ungrounded", END)

    return graph.compile()


def _run_summary_attributes(state: GraphState, result: RunResult) -> dict[str, Any]:
    """Quality signals for this run -- refusal/revision/grounding rates and
    nudge-retry counts -- so they can be tracked over time in MLflow without
    touching pipeline logic."""
    grounded = sum(1 for c in result.citations if c.grounded)
    ungrounded = len(result.citations) - grounded
    return {
        "refused": result.refused,
        "revised": result.revised,
        "tool_calls_made": result.tool_calls_made,
        "citations_grounded": grounded,
        "citations_ungrounded": ungrounded,
        "no_tool_attempts": state["no_tool_attempts"],
        "empty_search_attempts": state["empty_search_attempts"],
        "hedging_attempts": state["hedging_attempts"],
        "missing_citation_attempts": state["missing_citation_attempts"],
        "total_reruns": state["total_reruns"],
    }


def _initial_state(user_query: str) -> GraphState:
    return {
        "pending_input": user_query,
        "stage": "checkpoint",
        "checkpoint_outcome": "pass",
        "history": None,
        "draft": ResearchAnswer(),
        "deps": ResearchDeps(),
        "tool_calls_made": [],
        "citations": [],
        "revised": False,
        "refused": False,
        "model_failure": False,
        "no_tool_attempts": 0,
        "empty_search_attempts": 0,
        "hedging_attempts": 0,
        "missing_citation_attempts": 0,
        "total_reruns": 0,
    }


def _build_run_result(state: GraphState) -> RunResult:
    draft = state["draft"]
    return RunResult(
        answer=_render_prose(draft),
        recommendations=[] if state["refused"] else draft.recommendations,
        refusal=draft.refusal if state["refused"] else None,
        seen_review_ids=set(state["deps"].review_aliases.values()),
        citations=[] if state["refused"] else state["citations"],
        revised=state["revised"],
        refused=state["refused"],
        tool_calls_made=state["tool_calls_made"],
    )


class Coordinator:
    """Runs the Researcher -> Verifier -> (deterministic strip) pipeline via
    a LangGraph state machine (see module docstring)."""

    verifier_card = VERIFIER_CARD

    def __init__(self) -> None:
        self._graph = _build_graph()

    async def run(self, user_query: str) -> RunResult:
        with mlflow.start_span(name="coordinator_run") as span:
            span.set_inputs({"query": user_query})
            async with researcher_agent:
                final_state: GraphState = await self._graph.ainvoke(_initial_state(user_query))

            result = _build_run_result(final_state)
            span.set_outputs(_run_summary_attributes(final_state, result))
            return result

    async def stream(self, user_query: str):
        """Async-generator variant of run() for UIs that want progress
        updates. Yields ("status", node_name) as each graph node executes,
        then a final ("result", RunResult) once the graph reaches END.

        This is *not* token-level LLM streaming: the answer isn't safe to
        show until citation verification (and any stripping) has finished,
        so the best honest "streaming" signal here is which pipeline stage
        is currently running.
        """
        with mlflow.start_span(name="coordinator_run") as span:
            span.set_inputs({"query": user_query})
            async with researcher_agent:
                state: GraphState = _initial_state(user_query)
                async for update in self._graph.astream(state, stream_mode="updates"):
                    for node_name, partial in update.items():
                        if partial:
                            state.update(partial)  # type: ignore[typeddict-item]
                        span.add_event(
                            SpanEvent("graph_transition", attributes={"node": node_name, "stage": state["stage"]})
                        )
                        yield ("status", node_name)

            result = _build_run_result(state)
            span.set_outputs(_run_summary_attributes(state, result))

        yield ("result", result)
