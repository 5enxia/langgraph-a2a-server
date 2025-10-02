# Development Guide

## Setup Development Environment

### Prerequisites

- Python 3.10 or higher
- uv

### Installation

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies (including dev dependencies)
uv sync
```

This will automatically:
- Create a virtual environment (.venv)
- Install all dependencies including dev dependency group
- Install the package in editable mode

## Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=langgraph_a2a_server --cov-report=html

# Run specific test file
uv run pytest tests/test_executor.py
```

## Code Quality

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Fix linting issues automatically
uv run ruff check --fix .
```

## Running Examples

```bash
# First, install example dependencies
uv sync --extra examples

# Run the simple agent example
uv run python examples/simple_agent.py
```

Then visit:
- Agent Card: http://127.0.0.1:9000/.well-known/agent.json
- API Documentation: http://127.0.0.1:9000/docs (FastAPI only)

## Building the Package

```bash
# Build distribution packages
uv build

# This will create:
# - dist/langgraph_a2a_server-*.tar.gz (source distribution)
# - dist/langgraph_a2a_server-*.whl (wheel distribution)
```

## Publishing to PyPI

```bash
# Publish to PyPI (requires PyPI credentials)
uv publish

# Publish to TestPyPI first (recommended)
uv publish --publish-url https://test.pypi.org/legacy/
```

## Project Structure

```
langgraph-a2a-server/
├── src/
│   └── langgraph_a2a_server/
│       ├── __init__.py      # Package initialization
│       ├── executor.py       # A2A executor implementation
│       └── server.py         # A2A server implementation
├── tests/
│   ├── __init__.py
│   ├── test_executor.py     # Executor tests
│   └── test_server.py       # Server tests
├── examples/
│   └── simple_agent.py      # Simple example
├── pyproject.toml           # Project configuration
├── README.md                # User documentation
├── DEVELOPMENT.md           # This file
├── LICENSE                  # MIT License
└── .gitignore               # Git ignore rules
```

## Adding New Features

1. Create a new branch for your feature
2. Implement the feature with tests
3. Ensure all tests pass and code is formatted
4. Update documentation if needed
5. Submit a pull request

## Debugging

To enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Common Issues

### Import errors

Make sure you've installed the package with dependencies:
```bash
uv sync
```

### Tests not found

Make sure all dependencies are installed:
```bash
uv sync
```

### Example dependencies missing

If running examples fails, install the example dependencies:
```bash
uv sync --extra examples
```

## Resources

- [A2A Protocol Specification](https://github.com/google/a2a)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
