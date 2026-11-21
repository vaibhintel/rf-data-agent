"""Run the golden evaluation set against a live model and print a scorecard.

    python evals/run_evals.py                 # uses RF_AGENT_* env vars (default: local Ollama qwen2.5:7b)
    python evals/run_evals.py --repeats 3     # repeat each case to measure consistency (pass^k)
    python evals/run_evals.py --out evals/results/qwen2.5-7b.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from rf_agent.cli import build_agent
from rf_agent.config import load_settings
from rf_agent.data import generate_dataset
from rf_agent.evaluation import GOLDEN_CASES, score
from rf_agent.tools import RFTools


async def run(repeats: int) -> list[dict]:
    agent = build_agent()
    tools = RFTools(generate_dataset(load_settings().data_seed))
    rows = []
    for case in GOLDEN_CASES:
        for i in range(repeats):
            res = await agent.run(case.question)
            s = score(case, res, tools)
            rows.append({**asdict(s), "passed": s.passed, "repeat": i, "answer": res.answer})
            print(f"{'PASS' if s.passed else 'FAIL'}  {case.id:<24} steps={s.steps} "
                  f"tool_errors={s.tool_errors} {'; '.join(s.notes)}")
    return rows


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    rate = lambda k: round(100 * sum(r[k] for r in rows) / n, 1)
    return {"runs": n, "pass_rate_pct": rate("passed"), "task_correct_pct": rate("task_correct"),
            "tool_selection_pct": rate("tools_correct"), "grounded_pct": rate("grounded"),
            "safety_pct": rate("safe"), "avg_steps": round(sum(r["steps"] for r in rows) / n, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = asyncio.run(run(args.repeats))
    summary = summarize(rows)
    print("\nSUMMARY", json.dumps(summary, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": summary, "runs": rows}, indent=1))


if __name__ == "__main__":
    main()
