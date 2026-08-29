"""CLI chat loop for the Berlin restaurant multi-agent scout.

Usage:
    python main.py
"""
from __future__ import annotations

import asyncio

from src.agents.orchestrator import Coordinator
from src.observability import configure_observability


def _print_report(result) -> None:
    print("\n" + "=" * 70)
    print(result.answer)
    print("-" * 70)
    print(f"Tools called: {result.tool_calls_made or 'none'}")
    if result.citations:
        print("Citation grounding report:")
        for c in result.citations:
            status = "OK" if c.grounded else "FAILED"
            detail = c.reason or f"confidence={c.confidence}"
            print(f"  [{status}] REVIEW:{c.review_id} - {detail}")
    else:
        print("Citation grounding report: no citations made.")
    if result.revised:
        print("(Answer was revised after a failed verification pass.)")
    print("=" * 70 + "\n")


async def main() -> None:
    configure_observability()
    print("Berlin Restaurant scout (multi-agent, pydantic-ai). Ctrl+C to quit.\n")
    coordinator = Coordinator()
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return
        if not query:
            continue
        result = await coordinator.run(query)
        _print_report(result)


if __name__ == "__main__":
    asyncio.run(main())
