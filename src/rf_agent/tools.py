"""Domain tools the agent can call.

Design rules:
  * Plain, typed Python functions -> unit-testable with no LLM or MCP in the loop.
  * Read-only: no tool can modify or delete data.
  * Strict input validation; errors are returned to the agent with the valid options so it can self-correct.
  * Results are compact summaries, never raw rows, to keep context small and numbers checkable.
"""
from __future__ import annotations

import pandas as pd

from .data import RFDataset

REGRESSION_THRESHOLD_DB = 1.0   # a build is "regressed" if mean ACLR is >1 dB worse than its baseline


class ToolInputError(ValueError):
    """Raised for invalid tool arguments; the message is shown to the agent."""


class RFTools:
    def __init__(self, dataset: RFDataset):
        self.ds = dataset

    # ---- validation ----------------------------------------------------------------
    def _validate(self, build: str | None = None, temperature_c: int | None = None) -> None:
        if build is not None and build not in self.ds.builds:
            raise ToolInputError(f"Unknown build '{build}'. Valid builds: {self.ds.builds}")
        if temperature_c is not None and temperature_c not in self.ds.temps_c:
            raise ToolInputError(
                f"Unknown temperature '{temperature_c}'. Valid temperatures (C): {self.ds.temps_c}")

    # ---- tools ---------------------------------------------------------------------
    def list_options(self) -> dict:
        """Start here. Lists valid builds and temperatures, and how to interpret ACLR."""
        return {
            "builds_oldest_to_newest": self.ds.builds,
            "temperatures_c": self.ds.temps_c,
            "metric": "aclr_dbc (dBc). Higher (less negative) is WORSE.",
            "spec_limit": f"pass if aclr_dbc <= {self.ds.spec_limit_dbc}",
        }

    def summarize(self, build: str | None = None, temperature_c: int | None = None) -> dict:
        """Mean, worst value and pass rate (%) of ACLR per build and temperature. Filters are optional."""
        self._validate(build, temperature_c)
        d = self.ds.df
        if build is not None:
            d = d[d.build == build]
        if temperature_c is not None:
            d = d[d.temperature_c == temperature_c]
        limit = self.ds.spec_limit_dbc
        out = (d.groupby(["build", "temperature_c"]).aclr_dbc
                 .agg(mean_dbc="mean", worst_dbc="max", n="count",
                      pass_rate_pct=lambda s: (s <= limit).mean() * 100)
                 .round(2).reset_index())
        return {"rows": out.to_dict("records")}

    def compare_builds(self, old_build: str, new_build: str) -> dict:
        """Change in mean ACLR from old_build to new_build at each temperature.
        Positive delta_db means the new build is worse."""
        self._validate(old_build)
        self._validate(new_build)
        m = self.ds.df.groupby(["temperature_c", "build"]).aclr_dbc.mean().unstack()
        out = pd.DataFrame({
            "old_mean_dbc": m[old_build],
            "new_mean_dbc": m[new_build],
            "delta_db": m[new_build] - m[old_build],
        }).round(2)
        out["regression"] = out.delta_db > REGRESSION_THRESHOLD_DB
        return {"old_build": old_build, "new_build": new_build,
                "regression_threshold_db": REGRESSION_THRESHOLD_DB,
                "by_temperature": out.reset_index().to_dict("records")}

    def all(self) -> list:
        """Tools exposed to the agent. Note: deliberately no write/delete tools."""
        return [self.list_options, self.summarize, self.compare_builds]
