"""Tests for authentication functionality in LangGraph A2A Server."""

from unittest.mock import MagicMock

import pytest
from a2a.server.context import UnauthenticatedUser
from starlette.requests import Request

from langgraph_a2a_server.auth import (
    AuthenticatedUser,
    BearerTokenAuthContextBuilder,
    HeaderAuthContextBuilder,
)


class TestAuthenticatedUser:
    """Tests for AuthenticatedUser class."""

    def test_authenticated_user_is_authenticated(self):
        """Test that authenticated user returns True for is_authenticated."""
        user = AuthenticatedUser("test@example.com")
        assert user.is_authenticated is True

    def test_authenticated_user_username(self):
        """Test that authenticated user returns correct username."""
        user = AuthenticatedUser("test@example.com")
        assert user.user_name == "test@example.com"


class TestBearerTokenAuthContextBuilder:
    """Tests for BearerTokenAuthContextBuilder class."""

    def test_valid_bearer_token(self):
        """Test that valid bearer token creates authenticated user."""
        def validate_token(token: str) -> str | None:
            if token == "valid-token":
                return "user@example.com"
            return None

        builder = BearerTokenAuthContextBuilder(validate_token)

        # Create mock request with valid Bearer token
        mock_request = MagicMock(spec=Request)
        mock_request.headers.get.return_value = "Bearer valid-token"

        context = builder.build(mock_request)

        assert context.user.is_authenticated is True
        assert context.user.user_name == "user@example.com"

    def test_invalid_bearer_token(self):
        """Test that invalid bearer token creates unauthenticated context."""
        def validate_token(token: str) -> str | None:
            if token == "valid-token":
                return "user@example.com"
            return None

        builder = BearerTokenAuthContextBuilder(validate_token)

        # Create mock request with invalid Bearer token
        mock_request = MagicMock(spec=Request)
        mock_request.headers.get.return_value = "Bearer invalid-token"

        context = builder.build(mock_request)

        assert context.user.is_authenticated is False
        assert isinstance(context.user, UnauthenticatedUser)

    def test_no_authorization_header(self):
        """Test that missing authorization header creates unauthenticated context."""
        def validate_token(token: str) -> str | None:
            return "user@example.com"

        builder = BearerTokenAuthContextBuilder(validate_token)

        # Create mock request without Authorization header
        mock_request = MagicMock(spec=Request)
        mock_request.headers.get.return_value = ""

        context = builder.build(mock_request)

        assert context.user.is_authenticated is False
        assert isinstance(context.user, UnauthenticatedUser)

    def test_non_bearer_authorization(self):
        """Test that non-Bearer authorization creates unauthenticated context."""
        def validate_token(token: str) -> str | None:
            return "user@example.com"

        builder = BearerTokenAuthContextBuilder(validate_token)

        # Create mock request with Basic auth
        mock_request = MagicMock(spec=Request)
        mock_request.headers.get.return_value = "Basic dXNlcjpwYXNz"

        context = builder.build(mock_request)

        assert context.user.is_authenticated is False
        assert isinstance(context.user, UnauthenticatedUser)


class TestHeaderAuthContextBuilder:
    """Tests for HeaderAuthContextBuilder class."""

    def test_valid_api_key_header(self):
        """Test that valid API key header creates authenticated user."""
        def validate_credentials(headers: dict[str, str]) -> str | None:
            if headers.get("X-API-Key") == "secret-key":
                return "api-user@example.com"
            return None

        builder = HeaderAuthContextBuilder(
            validate_credentials,
            header_names=["X-API-Key"]
        )

        # Create mock request with valid API key
        mock_request = MagicMock(spec=Request)
        mock_request.headers.get.side_effect = lambda key, default="": (
            "secret-key" if key == "X-API-Key" else default
        )

        context = builder.build(mock_request)

        assert context.user.is_authenticated is True
        assert context.user.user_name == "api-user@example.com"

    def test_invalid_api_key_header(self):
        """Test that invalid API key header creates unauthenticated context."""
        def validate_credentials(headers: dict[str, str]) -> str | None:
            if headers.get("X-API-Key") == "secret-key":
                return "api-user@example.com"
            return None

        builder = HeaderAuthContextBuilder(
            validate_credentials,
            header_names=["X-API-Key"]
        )

        # Create mock request with invalid API key
        mock_request = MagicMock(spec=Request)
        mock_request.headers.get.side_effect = lambda key, default="": (
            "wrong-key" if key == "X-API-Key" else default
        )

        context = builder.build(mock_request)

        assert context.user.is_authenticated is False
        assert isinstance(context.user, UnauthenticatedUser)

    def test_multiple_headers(self):
        """Test authentication with multiple headers."""
        def validate_credentials(headers: dict[str, str]) -> str | None:
            if (headers.get("X-API-Key") == "secret-key" and
                    headers.get("X-User-ID") == "user123"):
                return "user123@example.com"
            return None

        builder = HeaderAuthContextBuilder(
            validate_credentials,
            header_names=["X-API-Key", "X-User-ID"]
        )

        # Create mock request with valid headers
        mock_request = MagicMock(spec=Request)
        def get_header(key, default=""):
            if key == "X-API-Key":
                return "secret-key"
            elif key == "X-User-ID":
                return "user123"
            return default
        mock_request.headers.get.side_effect = get_header

        context = builder.build(mock_request)

        assert context.user.is_authenticated is True
        assert context.user.user_name == "user123@example.com"

    def test_all_headers_when_none_specified(self):
        """Test that all headers are passed when header_names is None."""
        def validate_credentials(headers: dict[str, str]) -> str | None:
            # Check if specific header exists in all headers
            if "X-Custom-Auth" in headers:
                return "custom-user@example.com"
            return None

        builder = HeaderAuthContextBuilder(
            validate_credentials,
            header_names=None
        )

        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"X-Custom-Auth": "value", "Other-Header": "other"}

        context = builder.build(mock_request)

        assert context.user.is_authenticated is True
        assert context.user.user_name == "custom-user@example.com"
