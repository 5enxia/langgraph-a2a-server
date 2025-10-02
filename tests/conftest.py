"""Common fixtures for A2A module tests."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from langgraph.graph import StateGraph
from typing_extensions import TypedDict


class SimpleState(TypedDict):
    """Simple test state."""

    messages: list


def simple_node(state: SimpleState):
    """Simple test node that returns a response."""
    return {"messages": [{"role": "assistant", "content": "Test response"}]}


@pytest.fixture
def mock_langgraph():
    """Create a mock LangGraph for testing."""
    graph = StateGraph(SimpleState)
    graph.add_node("test", simple_node)
    graph.set_entry_point("test")
    graph.set_finish_point("test")
    compiled = graph.compile()
    return compiled


@pytest.fixture
def mock_request_context():
    """Create a mock RequestContext for testing."""
    context = MagicMock(spec=RequestContext)
    context.get_user_input.return_value = "Test input"
    return context


@pytest.fixture
def mock_event_queue():
    """Create a mock EventQueue for testing."""
    queue = MagicMock(spec=EventQueue)
    queue.enqueue_event = AsyncMock()
    return queue
