"""Runs the eval question set through the multi-agent system and reports
whether refusal behavior + citation grounding match expectations.

Usage:
    python run_eval.py
"""
from __future__ import annotations

import asyncio

import mlflow

from src.agents.orchestrator import Coordinator
from src.data_loader import get_review_by_id
from src.eval.adversarial_questions import QUESTIONS
from src.eval.llm_judge import judge_answer
from src.observability import configure_observability

REFUSAL_HINTS = (
    "don't have",
    "do not have",
    "no data",
    "no restaurant",
    "not in the dataset",
    "no information",
    "doesn't include",
    "does not include",
    "not available in",
    "couldn't find",
    "could not find",
    "no sushi",
    "no ramen",
    "no korean",
    "sorry",
)


def looks_like_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(hint in lowered for hint in REFUSAL_HINTS)


async def main() -> None:
    configure_observability()
    passed = 0
    faithfulness_scores: list[float] = []
    relevancy_scores: list[float] = []
    coordinator = Coordinator()
    for q in QUESTIONS:
        with mlflow.start_span(name="eval_question") as span:
            span.set_inputs({"query": q.query, "expected_refusal": q.should_refuse})
            result = await coordinator.run(q.query)
            # The graph's own `refused` flag only fires on the no-tool-call path;
            # most refusals happen in prose after a real (empty) search, so OR it
            # with the text heuristic instead of replacing it.
            refused = result.refused or looks_like_refusal(result.answer)
            failed_citations = [c for c in result.citations if not c.grounded]

            ok = (refused == q.should_refuse) and not failed_citations
            passed += ok

            contexts = [
                review.text
                for review_id in result.seen_review_ids
                if (review := get_review_by_id(review_id)) is not None
            ]
            score = await judge_answer(q.query, result.answer, contexts)
            if score is not None:
                faithfulness_scores.append(score.faithfulness)
                relevancy_scores.append(score.answer_relevancy)

            span.set_outputs(
                {
                    "passed": bool(ok),
                    "detected_refusal": refused,
                    "unverified_citation_count": len(failed_citations),
                    "faithfulness": score.faithfulness if score is not None else None,
                    "answer_relevancy": score.answer_relevancy if score is not None else None,
                }
            )

        print("=" * 70)
        print(f"Q: {q.query}")
        print(f"Expected refusal: {q.should_refuse} | Detected refusal: {refused}")
        print(f"Unverified citations remaining: {len(failed_citations)}")
        print(f"Result: {'PASS' if ok else 'FAIL'}  ({q.note})")
        print(f"Answer: {result.answer[:300]}")
        if score is not None:
            print(f"LLM judge: faithfulness={score.faithfulness:.2f} answer_relevancy={score.answer_relevancy:.2f}")
            print(f"  rationale: {score.rationale}")

    print("\n" + "=" * 70)
    print(f"{passed}/{len(QUESTIONS)} eval questions passed")
    if faithfulness_scores:
        print(
            f"Avg LLM-judge faithfulness: {sum(faithfulness_scores) / len(faithfulness_scores):.2f}  "
            f"| Avg answer_relevancy: {sum(relevancy_scores) / len(relevancy_scores):.2f}"
        )
    else:
        print("LLM-judge scores skipped (set ANTHROPIC_API_KEY to enable).")


if __name__ == "__main__":
    asyncio.run(main())

