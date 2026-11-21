"""Command-line entry point.

    rf-agent "Did any firmware build make ACLR worse?"
    rf-agent --trace "What is the pass rate for 25.3.0 at 60 C?"
"""
from __future__ import annotations

import argparse
import asyncio

from .agent import RFAgent
from .config import load_settings
from .data import generate_dataset
from .grounding import check_grounding, extract_numbers
from .llm import OpenAICompatibleLLM, RoutedLLM
from .mcp_server import build_server


def build_agent() -> RFAgent:
    s = load_settings()
    llm = RoutedLLM([OpenAICompatibleLLM(p, s.agent.temperature) for p in s.providers])
    return RFAgent(llm, build_server(generate_dataset(s.data_seed)), s.agent)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ask the RF data agent a question.")
    ap.add_argument("question")
    ap.add_argument("--trace", action="store_true", help="print every tool call")
    args = ap.parse_args()

    res = asyncio.run(build_agent().run(args.question))
    if args.trace:
        for s in res.trace:
            print(f"[step {s.step}] {s.tool}({s.arguments}) -> {'ok' if s.ok else 'ERROR'}")
    g = check_grounding(res.answer, res.tool_outputs, allowed=tuple(extract_numbers(args.question)))
    print("\nANSWER:", res.answer)
    print(f"\n(steps: {res.steps_used}, grounded numbers: {g.grounded}"
          + (f", ungrounded: {g.ungrounded}" if g.ungrounded else "") + ")")


if __name__ == "__main__":
    main()
