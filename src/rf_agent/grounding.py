"""Trace-level grounding: every number in the answer must come from a tool output.

On engineering data, a fluent but wrong dB value is the worst failure mode, so numeric
claims are verified deterministically against the execution trace, not by an LLM judge.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_BUILD = re.compile(r"\b\d+\.\d+\.\d+\b")                 # version strings like 25.3.0
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")


def extract_numbers(text: str) -> list[float]:
    """Numeric claims in free text, ignoring build/version identifiers."""
    return [float(x) for x in _NUMBER.findall(_BUILD.sub(" ", text))]


@dataclass
class GroundingReport:
    claimed: list[float]
    ungrounded: list[float]

    @property
    def grounded(self) -> bool:
        return not self.ungrounded


def check_grounding(answer: str, sources: list[str], tolerance: float = 0.05,
                    allowed: tuple[float, ...] = ()) -> GroundingReport:
    """A number is grounded if it appears (within tolerance, sign-insensitive) in any source.

    Sign-insensitivity allows "worse by 5.1 dB" when the tool reported delta_db=5.1 or -5.1.
    `allowed` covers values legitimately taken from the question itself (e.g. a temperature).
    """
    source_vals = [abs(v) for s in sources for v in extract_numbers(s)] + [abs(a) for a in allowed]
    claimed = extract_numbers(answer)
    ungrounded = [c for c in claimed
                  if not any(abs(abs(c) - v) <= tolerance for v in source_vals)]
    return GroundingReport(claimed=claimed, ungrounded=ungrounded)
