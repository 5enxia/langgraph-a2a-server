# LangGraph A2A Server

A2A (Agent-to-Agent) Protocol server implementation for LangGraph agents.

## Overview

This library provides a wrapper for LangGraph agents to be exposed via the A2A protocol, enabling seamless agent-to-agent communication and integration.

## Installation

```bash
pip install langgraph-a2a-server
```

Or with uv:

```bash
uv add langgraph-a2a-server
```

## Usage

```python
from langgraph_a2a_server import A2AServer
from your_langgraph_app import your_graph

# Create an A2A server with your LangGraph agent
server = A2AServer(
    graph=your_graph,
    name="My LangGraph Agent",
    description="An intelligent agent built with LangGraph",
    host="127.0.0.1",
    port=9000,
)

# Start the server
server.serve()
```

## Features

- Easy integration with existing LangGraph applications
- HTTP header-based authentication support (Bearer tokens, API keys, custom headers)
- Flexible authentication with custom validation functions
- Support for extended agent cards and dynamic card modification

## Authentication

The A2A server supports authentication via HTTP headers. You can use built-in authentication helpers or create custom authentication logic.

### Bearer Token Authentication

Use Bearer token authentication with the `Authorization` header:

```python
from langgraph_a2a_server import A2AServer, BearerTokenAuthContextBuilder

def validate_token(token: str) -> str | None:
    """Return username if token is valid, None otherwise."""
    if token == "secret-token-123":
        return "user@example.com"
    return None

context_builder = BearerTokenAuthContextBuilder(validate_token)

server = A2AServer(
    graph=your_graph,
    agent_card=agent_card,
    context_builder=context_builder,
)
```

Clients can then authenticate with:
```bash
curl -H "Authorization: Bearer secret-token-123" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"execute","params":{"input":"Hello"},"id":1}' \
  http://localhost:9000/jsonrpc
```

### Custom Header Authentication

Use custom headers for authentication (e.g., API keys):

```python
from langgraph_a2a_server import A2AServer, HeaderAuthContextBuilder

def validate_api_key(headers: dict[str, str]) -> str | None:
    """Return username if API key is valid, None otherwise."""
    api_key = headers.get("X-API-Key", "")
    if api_key == "sk-my-secret-key":
        return "api-user@example.com"
    return None

context_builder = HeaderAuthContextBuilder(
    validate_api_key,
    header_names=["X-API-Key"]
)

server = A2AServer(
    graph=your_graph,
    agent_card=agent_card,
    context_builder=context_builder,
)
```

Clients can then authenticate with:
```bash
curl -H "X-API-Key: sk-my-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"execute","params":{"input":"Hello"},"id":1}' \
  http://localhost:9000/jsonrpc
```

### Custom Authentication

For advanced use cases, implement your own `CallContextBuilder`:

```python
from a2a.server.apps.jsonrpc.jsonrpc_app import CallContextBuilder
from a2a.server.context import ServerCallContext
from starlette.requests import Request
from langgraph_a2a_server.auth import AuthenticatedUser

class CustomAuthContextBuilder(CallContextBuilder):
    def build(self, request: Request) -> ServerCallContext:
        context = ServerCallContext()
        # Your custom authentication logic here
        # Extract credentials from request, validate them, etc.
        user_id = extract_and_validate_credentials(request)
        if user_id:
            context.user = AuthenticatedUser(user_id)
        return context

server = A2AServer(
    graph=your_graph,
    agent_card=agent_card,
    context_builder=CustomAuthContextBuilder(),
)
```

## Examples

### simple_agent.py (no llm)

```sh
uv run --extra examples examples/simple_agent.py
```

### langchain_agent.py (with llm)

```sh
uv run --extra examples examples/langchain_agent.py
```

### tools_agent.py (with tools, no llm)

```sh
uv run --extra examples examples/tools_agent.py
```

### authenticated_agent.py (Bearer token authentication)

```sh
uv run --extra examples examples/authenticated_agent.py
```

### api_key_agent.py (API key authentication)

```sh
uv run --extra examples examples/api_key_agent.py
```

## License

MIT
