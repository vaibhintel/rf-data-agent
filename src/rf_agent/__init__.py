"""RF test-data agent: an LLM agent that answers engineering questions over RF measurement data via MCP tools."""
from .agent import AgentResult, RFAgent
from .data import generate_dataset
from .mcp_server import build_server

__all__ = ["AgentResult", "RFAgent", "build_server", "generate_dataset"]
__version__ = "0.1.0"
