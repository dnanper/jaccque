# Memory API

A memory system for AI agents with temporal + semantic + entity memory architecture.

## Features

- PostgreSQL with pgvector for vector storage
- Multi-strategy retrieval (TEMPR)
- LLM-powered fact extraction and reasoning
- FastAPI HTTP endpoints
- MCP (Model Context Protocol) support

## Quick Start

```bash
# Start database
docker-compose -f docker-compose.dev.yml up -d

# Install package
pip install -e .

# Run tests
python test_direct.py
```

## Development

See the parent directory's README for full documentation.
