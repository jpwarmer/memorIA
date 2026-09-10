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
- `EMBED_MAX_TOKENS` (default: `128` — MiniLM trained window; do not inherit tokenizer.json)
- `EMBED_CHUNK_OVERLAP` (default: `16`)

## MCP Client Configuration

Use these references:

- `memory-mcp-server/cursor-mcp.example.json`
- `memory-mcp-server/claude-desktop-mcp.example.json`

Make sure `command` and `args` point to your `.venv` Python interpreter and to
`memory-mcp-server/server.py`. If you use a generic `python` command, your MCP
client may pick a different interpreter and fail with
`ModuleNotFoundError: fastembed`.

## Available Tools (Actual Contract)

- `memory_search(..., min_score=None)`
- `memory_save(...)`
- `memory_list(..., include_text=False, include_superseded=False)`
- `memory_update(memory_id, text=None, tags=None, metadata=None)`
- `memory_delete(memory_id)`
- `memory_supersede(memory_id, text, ...)`
- `memory_stats(project_id="default", scope=None)`
- `memory_related(memory_id, limit=5, min_score=None)`
- `memory_reindex(scope=None, dry_run=True)`
- `memory_dedupe(project_id="default", scope="project", threshold=0.95)`

Notes:

- Search runs over chunks and returns de-duplicated parents. Superseded memories are excluded.
- `memory_list` returns `next_cursor`. Default omits `text`.
- `memory_dedupe` reports clusters and never deletes.

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
