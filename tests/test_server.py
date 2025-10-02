"""Tests for LangGraph A2A Server."""

from unittest.mock import patch

import pytest
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from fastapi import FastAPI
from starlette.applications import Starlette

from langgraph_a2a_server.server import A2AServer


def test_server_initialization(mock_langgraph):
    """Test that A2AServer initializes correctly with default values."""
    server = A2AServer(
        graph=mock_langgraph,
        name="Test Agent",
        description="A test agent",
    )

    assert server.name == "Test Agent"
    assert server.description == "A test agent"
    assert server.host == "127.0.0.1"
    assert server.port == 9000
    assert server.http_url == "http://127.0.0.1:9000/"
    assert server.version == "0.0.1"
    assert isinstance(server.capabilities, AgentCapabilities)
    assert server.capabilities.streaming is True


def test_server_initialization_with_custom_values(mock_langgraph):
    """Test that A2AServer initializes correctly with custom values."""
    server = A2AServer(
        graph=mock_langgraph,
        name="Custom Agent",
        description="A custom test agent",
        host="0.0.0.0",
        port=8080,
        version="1.0.0",
    )

    assert server.host == "0.0.0.0"
    assert server.port == 8080
    assert server.http_url == "http://0.0.0.0:8080/"
    assert server.version == "1.0.0"
    assert server.capabilities.streaming is True


def test_server_with_http_url(mock_langgraph):
    """Test server with custom HTTP URL."""
    server = A2AServer(
        graph=mock_langgraph,
        name="Test Agent",
        description="A test agent",
        http_url="http://example.com/agent",
    )

    assert server.http_url == "http://example.com/agent/"
    assert server.public_base_url == "http://example.com"
    assert server.mount_path == "/agent"


def test_server_with_http_url_no_path(mock_langgraph):
    """Test server with HTTP URL containing no path."""
    server = A2AServer(
        graph=mock_langgraph,
        name="Test Agent",
        description="A test agent",
        http_url="http://my-alb.amazonaws.com",
    )

    assert server.http_url == "http://my-alb.amazonaws.com/"
    assert server.public_base_url == "http://my-alb.amazonaws.com"
    assert server.mount_path == ""


def test_server_with_serve_at_root(mock_langgraph):
    """Test server with serve_at_root flag."""
    server = A2AServer(
        graph=mock_langgraph,
        name="Test Agent",
        description="A test agent",
        http_url="http://example.com/agent",
        serve_at_root=True,
    )

    assert server.http_url == "http://example.com/agent/"
    assert server.mount_path == ""  # Should be empty despite path in URL


def test_server_with_skills(mock_langgraph):
    """Test server with custom skills."""
    skills = [
        AgentSkill(name="skill1", id="skill1", description="First skill", tags=[]),
        AgentSkill(name="skill2", id="skill2", description="Second skill", tags=[]),
    ]

    server = A2AServer(
        graph=mock_langgraph,
        name="Test Agent",
        description="A test agent",
        skills=skills,
    )

    assert len(server.agent_skills) == 2
    assert server.agent_skills[0].name == "skill1"
    assert server.agent_skills[1].name == "skill2"


def test_server_with_empty_skills_list(mock_langgraph):
    """Test that passing an empty skills list works correctly."""
    server = A2AServer(
        graph=mock_langgraph,
        name="Test Agent",
        description="A test agent",
        skills=[],
    )

    # Should have empty skills list
    skills = server.agent_skills
    assert isinstance(skills, list)
    assert len(skills) == 0


def test_server_with_none_skills(mock_langgraph):
    """Test server with None skills defaults to empty list."""
    server = A2AServer(
        graph=mock_langgraph,
        name="Test Agent",
        description="A test agent",
        skills=None,
    )

    assert isinstance(server.agent_skills, list)
    assert len(server.agent_skills) == 0


def test_parse_public_url(mock_langgraph):
    """Test URL parsing."""
    server = A2AServer(
        graph=mock_langgraph,
        name="Test Agent",
        description="A test agent",
    )

    base_url, mount_path = server._parse_public_url("http://example.com/agent")
    assert base_url == "http://example.com"
    assert mount_path == "/agent"

    base_url, mount_path = server._parse_public_url("http://example.com")
    assert base_url == "http://example.com"
    assert mount_path == ""

    base_url, mount_path = server._parse_public_url("http://example.com/")
    assert base_url == "http://example.com"
    assert mount_path == ""


def test_public_agent_card(mock_langgraph):
    """Test agent card generation."""
    server = A2AServer(
        graph=mock_langgraph,
        name="Test Agent",
        description="A test agent",
        version="1.0.0",
    )

    card = server.public_agent_card
    assert isinstance(card, AgentCard)
    assert card.name == "Test Agent"
    assert card.description == "A test agent"
    assert card.version == "1.0.0"
    assert card.url == "http://127.0.0.1:9000/"
    assert card.capabilities.streaming is True
    assert card.default_input_modes == ["text"]
    assert card.default_output_modes == ["text"]


def test_public_agent_card_with_custom_skills(mock_langgraph):
    """Test that public_agent_card includes custom skills."""
    custom_skills = [
        AgentSkill(name="custom_skill", id="custom_skill", description="A custom skill", tags=["test"]),
    ]

    server = A2AServer(
        graph=mock_langgraph,
        name="Test Agent",
        description="A test agent",
        skills=custom_skills,
    )
    card = server.public_agent_card

    assert card.skills == custom_skills
    assert len(card.skills) == 1
    assert card.skills[0].name == "custom_skill"


def test_server_name_validation(mock_langgraph):
    """Test that server validates name."""
    server = A2AServer(
        graph=mock_langgraph,
        name="",
        description="A test agent",
    )

    with pytest.raises(ValueError, match="name cannot be None or empty"):
        _ = server.public_agent_card


def test_server_description_validation(mock_langgraph):
    """Test that server validates description."""
    server = A2AServer(
        graph=mock_langgraph,
        name="Test Agent",
        description="",
    )

    with pytest.raises(ValueError, match="description cannot be None or empty"):
        _ = server.public_agent_card


def test_agent_skills_setter(mock_langgraph):
    """Test that agent_skills setter works correctly."""
    server = A2AServer(
        graph=mock_langgraph,
        name="Test Agent",
        description="A test agent",
    )

    # Set new skills using setter
    new_skills = [
        AgentSkill(name="new_skill", id="new_skill", description="A new skill", tags=["new"]),
        AgentSkill(name="another_new_skill", id="another_new_skill", description="Another new skill", tags=[]),
    ]

    server.agent_skills = new_skills

    # Verify skills were updated
    assert server.agent_skills == new_skills
    assert len(server.agent_skills) == 2
    assert server.agent_skills[0].name == "new_skill"
    assert server.agent_skills[1].name == "another_new_skill"


def test_to_starlette_app(mock_langgraph):
    """Test that to_starlette_app returns a Starlette application."""
    server = A2AServer(
        graph=mock_langgraph,
        name="Test Agent",
        description="A test agent",
        skills=[],
    )

    app = server.to_starlette_app()

    assert isinstance(app, Starlette)


def test_to_starlette_app_with_mounting(mock_langgraph):
    """Test that to_starlette_app creates mounted app when mount_path exists."""
    server = A2AServer(
        graph=mock_langgraph,
        name="Test Agent",
        description="A test agent",
        http_url="http://example.com/agent1",
        skills=[],
    )

    app = server.to_starlette_app()

    assert isinstance(app, Starlette)


def test_to_fastapi_app(mock_langgraph):
    """Test that to_fastapi_app returns a FastAPI application."""
    server = A2AServer(
        graph=mock_langgraph,
        name="Test Agent",
        description="A test agent",
        skills=[],
    )

    app = server.to_fastapi_app()

    assert isinstance(app, FastAPI)


def test_to_fastapi_app_with_mounting(mock_langgraph):
    """Test that to_fastapi_app creates mounted app when mount_path exists."""
    server = A2AServer(
        graph=mock_langgraph,
        name="Test Agent",
        description="A test agent",
        http_url="http://example.com/agent1",
        skills=[],
    )

    app = server.to_fastapi_app()

    assert isinstance(app, FastAPI)


@patch("uvicorn.run")
def test_serve_with_starlette(mock_run, mock_langgraph):
    """Test that serve starts a Starlette server by default."""
    server = A2AServer(
        graph=mock_langgraph,
        name="Test Agent",
        description="A test agent",
        skills=[],
    )

    server.serve()

    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    assert isinstance(args[0], Starlette)
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 9000


@patch("uvicorn.run")
def test_serve_with_fastapi(mock_run, mock_langgraph):
    """Test that serve starts a FastAPI server when specified."""
    server = A2AServer(
        graph=mock_langgraph,
        name="Test Agent",
        description="A test agent",
        skills=[],
    )

    server.serve(app_type="fastapi")

    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    assert isinstance(args[0], FastAPI)
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 9000


@patch("uvicorn.run")
def test_serve_with_custom_host_port(mock_run, mock_langgraph):
    """Test that serve can override host and port."""
    server = A2AServer(
        graph=mock_langgraph,
        name="Test Agent",
        description="A test agent",
        host="0.0.0.0",
        port=8080,
    )

    server.serve(host="192.168.1.1", port=9999)

    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs["host"] == "192.168.1.1"
    assert kwargs["port"] == 9999


@patch("uvicorn.run")
def test_serve_with_custom_kwargs(mock_run, mock_langgraph):
    """Test that serve passes additional kwargs to uvicorn.run."""
    server = A2AServer(
        graph=mock_langgraph,
        name="Test Agent",
        description="A test agent",
        skills=[],
    )

    server.serve(log_level="debug", reload=True)

    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs["log_level"] == "debug"
    assert kwargs["reload"] is True


@patch("uvicorn.run", side_effect=KeyboardInterrupt)
def test_serve_handles_keyboard_interrupt(mock_run, mock_langgraph, caplog):
    """Test that serve handles KeyboardInterrupt gracefully."""
    import logging
    
    server = A2AServer(
        graph=mock_langgraph,
        name="Test Agent",
        description="A test agent",
        skills=[],
    )

    with caplog.at_level(logging.INFO):
        server.serve()

    assert "LangGraph A2A server shutdown requested (KeyboardInterrupt)" in caplog.text
    assert "LangGraph A2A server has shutdown" in caplog.text


@patch("uvicorn.run", side_effect=Exception("Test exception"))
def test_serve_handles_general_exception(mock_run, mock_langgraph, caplog):
    """Test that serve handles general exceptions gracefully."""
    import logging
    
    server = A2AServer(
        graph=mock_langgraph,
        name="Test Agent",
        description="A test agent",
        skills=[],
    )

    with caplog.at_level(logging.INFO):
        server.serve()

    assert "LangGraph A2A server encountered exception" in caplog.text
    assert "LangGraph A2A server has shutdown" in caplog.text


def test_executor_created_correctly(mock_langgraph):
    """Test that the executor is created correctly."""
    from langgraph_a2a_server.executor import LangGraphA2AExecutor

    server = A2AServer(
        graph=mock_langgraph,
        name="Test Agent",
        description="A test agent",
    )

    assert isinstance(server.request_handler.agent_executor, LangGraphA2AExecutor)
    assert server.request_handler.agent_executor.graph == mock_langgraph


def test_backwards_compatibility_without_http_url(mock_langgraph):
    """Test that the old behavior is preserved when http_url is not provided."""
    server = A2AServer(
        graph=mock_langgraph,
        name="Test Agent",
        description="A test agent",
        host="localhost",
        port=9000,
        skills=[],
    )

    # Should behave exactly like before
    assert server.host == "localhost"
    assert server.port == 9000
    assert server.http_url == "http://localhost:9000/"
    assert server.public_base_url == "http://localhost:9000"
    assert server.mount_path == ""

    # Agent card should use the traditional URL
    card = server.public_agent_card
    assert card.url == "http://localhost:9000/"
