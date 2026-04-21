# Memory MCP (memorIA)

Persistent semantic memory for agents (Claude Code, Cursor, and custom agents)
using MCP + Qdrant.

## Goal

Avoid starting every session from zero:

- Before working: the agent retrieves context with `memory_search`.
- After finishing: the agent persists learnings with `memory_save`.

## Current Architecture

```text
Agent (Claude/Cursor/custom)
        <- MCP stdio ->
memory-mcp-server/server.py
        <- qdrant-client ->
Qdrant Docker (localhost:6333)

Scopes: project / session / global / cache
```

## Requirements

- Docker Desktop running
- Python 3.10+
- A shell environment:
  - Windows: PowerShell
  - macOS: Terminal (zsh/bash)

## Quick Setup

From the repository root:

### Windows (PowerShell)

```powershell
docker compose up -d
cd memory-mcp-server
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe server.py
```

### macOS (zsh/bash)

```bash
docker compose up -d
cd memory-mcp-server
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
./.venv/bin/python server.py
```

## Environment Variables

Set these in `memory-mcp-server/.env`:

- `QDRANT_URL` (default: `http://localhost:6333`)
- `QDRANT_API_KEY` (optional)
- `MCP_SERVER_NAME` (default: `memorIA`)
- `EMBEDDING_MODEL` (default: multilingual model)

## MCP Client Configuration

Use these references:

- `memory-mcp-server/cursor-mcp.example.json`
- `memory-mcp-server/claude-desktop-mcp.example.json`

Make sure `command` and `args` point to your `.venv` Python interpreter and to
`memory-mcp-server/server.py`. If you use a generic `python` command, your MCP
client may pick a different interpreter and fail with
`ModuleNotFoundError: fastembed`.

## Available Tools (Actual Contract)

- `memory_search(query, project_id="default", scope="project", limit=5, tags=None, session_id=None)`
- `memory_save(text, project_id="default", scope="project", tags=None, session_id=None, source="agent", metadata=None)`
- `memory_list(project_id="default", scope="project", limit=20, cursor=None, tags=None, session_id=None)`

Notes:

- `memory_list` returns `next_cursor` for pagination.
- `memory_delete` is not part of this MVP.

## Recommended Smoke Test

1. Save 2 memories with `memory_save`.
2. Search by intent with `memory_search` and validate relevance.
3. List results with pagination using `memory_list` + `next_cursor`.

If all 3 steps work, the memory loop is operational.

## Dashboard

Qdrant UI: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

## Agent Prompt Guide

Use `AGENT_INSTRUCTIONS.md` as your base system prompt to standardize memory
usage across agents.
