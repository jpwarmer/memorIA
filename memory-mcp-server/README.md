# Memory MCP Server (MVP)

Local MCP server for persistent semantic memory, using Qdrant as backend.

## 1) Start Qdrant

From the repository root:

```bash
docker compose up -d
```

Qdrant will be available at `http://localhost:6333`.

## 2) Install server dependencies

### Windows (PowerShell)

```powershell
cd memory-mcp-server
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### macOS (zsh/bash)

```bash
cd memory-mcp-server
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

## 3) Configure environment

### Windows (PowerShell)

```powershell
Copy-Item .env.example .env
```

### macOS (zsh/bash)

```bash
cp .env.example .env
```

Variables:
- `QDRANT_URL` (default `http://localhost:6333`)
- `QDRANT_API_KEY` (optional)
- `MCP_SERVER_NAME` (default `memorIA`)
- `EMBEDDING_MODEL` (default multilingual ES/EN model)
- `EMBED_MAX_TOKENS` (default `128`)
- `EMBED_CHUNK_OVERLAP` (default `16`)

## 4) Run the MCP server

### Windows (PowerShell)

```powershell
.\.venv\Scripts\python.exe server.py
```

### macOS (zsh/bash)

```bash
./.venv/bin/python server.py
```

Exposes tools:
- `memory_search(..., min_score=None)`
- `memory_save(...)`
- `memory_list(..., include_text=False)`
- `memory_update` / `memory_delete` / `memory_supersede`
- `memory_stats` / `memory_related` / `memory_reindex` / `memory_dedupe`

Supported scopes:
- `project`
- `session`
- `global`
- `cache`

## 5) Recommended agent workflow

1. Before task: `memory_search`
2. During task: use results as context
3. After task: `memory_save` with learnings and tags

## Notes

This skeleton uses local embeddings with `fastembed` to provide real semantic
search from the MVP stage. The first startup may take longer due to model
download.

`memory_list` returns `next_cursor`; call it again with that value in `cursor`
to paginate.
