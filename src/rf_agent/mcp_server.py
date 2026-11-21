"""Expose the RF tools over the Model Context Protocol (MCP).

The agent never imports the tools directly; it discovers them through MCP. That keeps the
tool layer swappable (in-process for tests, stdio/HTTP for a real deployment) and lets
any MCP-capable client (Claude Desktop, IDEs, other agents) reuse the same tools.

Run standalone as a stdio server:
    python -m rf_agent.mcp_server
"""
from __future__ import annotations

import functools
import logging

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from .data import RFDataset, generate_dataset
from .tools import RFTools, ToolInputError

logging.getLogger("fastmcp").setLevel(logging.CRITICAL)   # tool errors go to the agent, not stderr


def _as_mcp_tool(fn):
    """Translate domain validation errors into MCP tool errors the model can read."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolInputError as e:
            raise ToolError(str(e)) from e
    return wrapper


def build_server(dataset: RFDataset | None = None) -> FastMCP:
    tools = RFTools(dataset or generate_dataset())
    mcp = FastMCP("rf-data")
    for fn in tools.all():
        mcp.tool(_as_mcp_tool(fn))
    return mcp


if __name__ == "__main__":
    build_server().run()   # stdio transport by default
