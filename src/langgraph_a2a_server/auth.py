"""Authentication utilities for LangGraph A2A Server.

This module provides helper classes for implementing authentication in A2A servers.
"""

from abc import ABC, abstractmethod
from typing import Callable

from a2a.server.apps.jsonrpc.jsonrpc_app import CallContextBuilder
from a2a.server.context import ServerCallContext, User
from starlette.requests import Request


class AuthenticatedUser(User):
    """A simple implementation of an authenticated user."""

    def __init__(self, user_name: str):
        """Initialize an authenticated user.

        Args:
            user_name: The name/identifier of the authenticated user.
        """
        self._user_name = user_name

    @property
    def is_authenticated(self) -> bool:
        """Returns whether the current user is authenticated."""
        return True

    @property
    def user_name(self) -> str:
        """Returns the user name of the current user."""
        return self._user_name


class BearerTokenAuthContextBuilder(CallContextBuilder):
    """A CallContextBuilder that extracts Bearer tokens from the Authorization header.

    This builder extracts Bearer tokens from the HTTP Authorization header and validates
    them using a provided validation function. If validation succeeds, it creates an
    authenticated user context.

    Example:
        ```python
        def validate_token(token: str) -> str | None:
            # Return username if valid, None otherwise
            if token == "secret-token-123":
                return "user@example.com"
            return None

        context_builder = BearerTokenAuthContextBuilder(validate_token)

        server = A2AServer(
            graph=graph,
            agent_card=agent_card,
            context_builder=context_builder,
        )
        ```
    """

    def __init__(self, validate_token: Callable[[str], str | None]):
        """Initialize the Bearer token authentication context builder.

        Args:
            validate_token: A function that takes a token string and returns the
                username if the token is valid, or None if invalid.
        """
        self.validate_token = validate_token

    def build(self, request: Request) -> ServerCallContext:
        """Builds a ServerCallContext from a Starlette Request.

        Extracts the Bearer token from the Authorization header and validates it.
        If valid, creates an authenticated user context.

        Args:
            request: The incoming HTTP request.

        Returns:
            ServerCallContext with authenticated user if token is valid.
        """
        context = ServerCallContext()

        # Extract Authorization header
        auth_header = request.headers.get("Authorization", "")

        # Check for Bearer token
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]  # Remove "Bearer " prefix
            username = self.validate_token(token)
            if username:
                context.user = AuthenticatedUser(username)

        return context


class HeaderAuthContextBuilder(CallContextBuilder):
    """A CallContextBuilder that extracts authentication from custom HTTP headers.

    This builder extracts authentication information from specified HTTP headers
    and validates them using a provided validation function.

    Example:
        ```python
        def validate_credentials(headers: dict[str, str]) -> str | None:
            # Return username if valid, None otherwise
            api_key = headers.get("X-API-Key", "")
            if api_key == "secret-key-123":
                return "user@example.com"
            return None

        context_builder = HeaderAuthContextBuilder(
            validate_credentials,
            header_names=["X-API-Key", "X-User-ID"]
        )

        server = A2AServer(
            graph=graph,
            agent_card=agent_card,
            context_builder=context_builder,
        )
        ```
    """

    def __init__(
        self,
        validate_credentials: Callable[[dict[str, str]], str | None],
        header_names: list[str] | None = None,
    ):
        """Initialize the header authentication context builder.

        Args:
            validate_credentials: A function that takes a dictionary of header
                key-value pairs and returns the username if valid, or None if invalid.
            header_names: List of header names to extract. If None, all headers
                are passed to the validation function.
        """
        self.validate_credentials = validate_credentials
        self.header_names = header_names

    def build(self, request: Request) -> ServerCallContext:
        """Builds a ServerCallContext from a Starlette Request.

        Extracts specified headers and validates them.

        Args:
            request: The incoming HTTP request.

        Returns:
            ServerCallContext with authenticated user if credentials are valid.
        """
        context = ServerCallContext()

        # Extract specified headers or all headers
        if self.header_names:
            headers = {name: request.headers.get(name, "") for name in self.header_names}
        else:
            headers = dict(request.headers)

        # Validate credentials
        username = self.validate_credentials(headers)
        if username:
            context.user = AuthenticatedUser(username)

        return context
