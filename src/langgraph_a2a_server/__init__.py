"""LangGraph A2A Server - A2A Protocol implementation for LangGraph agents.

This package provides a bridge between LangGraph agents and the A2A (Agent2Agent) protocol,
allowing LangGraph-based agents to be easily served and integrated into A2A-compatible ecosystems.
"""

from .executor import LangGraphA2AExecutor
from .server import A2AServer

__version__ = "0.1.6"
__all__ = ["A2AServer", "LangGraphA2AExecutor"]
