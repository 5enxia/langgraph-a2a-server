"""A2A-compatible wrapper for LangGraph agents.

This module provides the A2AServer class, which adapts a LangGraph agent to the A2A protocol,
allowing it to be used in A2A-compatible systems.
"""

import logging
from typing import Any, Literal
from urllib.parse import urlparse

import uvicorn
from a2a.server.apps import A2AFastAPIApplication, A2AStarletteApplication
from a2a.server.events import QueueManager
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import (
    InMemoryTaskStore,
    PushNotificationConfigStore,
    PushNotificationSender,
    TaskStore,
)
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from fastapi import FastAPI
from langgraph.graph.state import CompiledStateGraph
from starlette.applications import Starlette

from .executor import LangGraphA2AExecutor

logger = logging.getLogger(__name__)


class A2AServer:
    """A2A-compatible wrapper for LangGraph agents.

    This class provides a high-level interface to serve a LangGraph agent as an
    A2A-compliant service. It handles the creation of web applications (FastAPI or Starlette)
    and configures the necessary request handlers and executors.

    Example:
        ```python
        from langgraph_a2a_server import A2AServer
        from a2a.types import AgentCard

        # Define your LangGraph 'graph'
        agent_card = AgentCard(name="My Agent", description="A helpful assistant")
        server = A2AServer(graph, agent_card=agent_card)
        server.serve(port=9000)
        ```
    """

    def __init__(
        self,
        graph: CompiledStateGraph,
        *,
        agent_card: AgentCard,
        # Server configuration
        host: str = "127.0.0.1",
        port: int = 9000,
        http_url: str | None = None,
        serve_at_root: bool = False,
        # Graph configuration
        input_key: str = "messages",
        output_key: str = "messages",
        # RequestHandler
        task_store: TaskStore | None = None,
        queue_manager: QueueManager | None = None,
        push_config_store: PushNotificationConfigStore | None = None,
        push_sender: PushNotificationSender | None = None,
    ):
        """Initialize an A2A-compatible server from a LangGraph agent.

        Args:
            graph: The compiled LangGraph instance to wrap with A2A compatibility.
            agent_card: The AgentCard containing metadata about the agent.
            host: The hostname or IP address to bind the A2A server to. Defaults to "127.0.0.1".
            port: The port to bind the A2A server to. Defaults to 9000.
            http_url: The public HTTP URL where this agent will be accessible.
            serve_at_root: If True, forces the server to serve at root path.
            input_key: The key in the graph state to send input messages to. Defaults to "messages".
            output_key: The key in the graph state to read output from. Defaults to "messages".
            task_store: Custom task store implementation. Defaults to InMemoryTaskStore.
            queue_manager: Custom queue manager for handling message queues.
            push_config_store: Custom store for push notification configurations.
            push_sender: Custom push notification sender implementation.
        """
        # Validate required fields
        if not agent_card.name:
            raise ValueError("A2A agent name cannot be None or empty")
        if not agent_card.description:
            raise ValueError("A2A agent description cannot be None or empty")

        self._agent_card = agent_card

        # Server configuration
        self.host = host
        self.port = port

        if http_url:
            # Parse the provided URL to extract components for mounting
            self.public_base_url, self.mount_path = self._parse_public_url(http_url)
            self.http_url = http_url.rstrip("/") + "/"

            # Override mount path if serve_at_root is requested
            if serve_at_root:
                self.mount_path = ""
        else:
            # Fall back to constructing the URL from host and port
            self.public_base_url = f"http://{host}:{port}"
            self.http_url = f"{self.public_base_url}/"
            self.mount_path = ""

        self.graph = graph
        self.capabilities = agent_card.capabilities or AgentCapabilities(streaming=True)

        self.request_handler = DefaultRequestHandler(
            agent_executor=LangGraphA2AExecutor(graph, input_key=input_key, output_key=output_key),
            task_store=task_store or InMemoryTaskStore(),
            queue_manager=queue_manager,
            push_config_store=push_config_store,
            push_sender=push_sender,
        )
        logger.info("LangGraph integration with A2A is ready for use.")

    @property
    def name(self) -> str:
        """Get the agent's name."""
        return self._agent_card.name

    @property
    def description(self) -> str:
        """Get the agent's description."""
        return self._agent_card.description

    @property
    def version(self) -> str:
        """Get the agent's version."""
        return self._agent_card.version or "0.0.1"

    def _parse_public_url(self, url: str) -> tuple[str, str]:
        """Parse the public URL into base URL and mount path components."""
        parsed = urlparse(url.rstrip("/"))
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        mount_path = parsed.path if parsed.path != "/" else ""
        return base_url, mount_path

    @property
    def public_agent_card(self) -> AgentCard:
        """Get the public AgentCard for this agent.

        Returns:
            AgentCard: The public agent card containing metadata about this agent.
        """
        # Return a copy of the agent card with updated URL and skills
        return self._agent_card.model_copy(update={"url": self.http_url, "skills": self.agent_skills})

    @property
    def agent_skills(self) -> list[AgentSkill]:
        """Get the list of skills this agent provides."""
        return self._agent_card.skills or []

    @agent_skills.setter
    def agent_skills(self, skills: list[AgentSkill]) -> None:
        """Set the list of skills this agent provides."""
        self._agent_card.skills = skills

    def _create_app(self, app_class: type[FastAPI] | type[Starlette], application_class: Any) -> Any:
        """Helper to create a Starlette or FastAPI application."""
        a2a_app = application_class(agent_card=self.public_agent_card, http_handler=self.request_handler).build()

        if self.mount_path:
            # Create parent app and mount the A2A app at the specified path
            parent_app = app_class()
            parent_app.mount(self.mount_path, a2a_app)
            logger.info("Mounting A2A server at path: %s", self.mount_path)
            return parent_app

        return a2a_app

    def to_starlette_app(self) -> Starlette:
        """Create a Starlette application for serving this agent via HTTP."""
        return self._create_app(Starlette, A2AStarletteApplication)

    def to_fastapi_app(self) -> FastAPI:
        """Create a FastAPI application for serving this agent via HTTP."""
        return self._create_app(FastAPI, A2AFastAPIApplication)

    def serve(
        self,
        app_type: Literal["fastapi", "starlette"] = "starlette",
        *,
        host: str | None = None,
        port: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Start the A2A server with the specified application type."""
        try:
            logger.info("Starting LangGraph A2A server...")
            app = self.to_fastapi_app() if app_type == "fastapi" else self.to_starlette_app()
            uvicorn.run(app, host=host or self.host, port=port or self.port, **kwargs)
        except KeyboardInterrupt:
            logger.warning("LangGraph A2A server shutdown requested (KeyboardInterrupt).")
        except Exception:
            logger.exception("LangGraph A2A server encountered exception.")
        finally:
            logger.info("LangGraph A2A server has shutdown.")
