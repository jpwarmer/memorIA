# Agent Memory Instructions

This project uses persistent semantic memory through MCP (`memorIA`).
Your goal is to maintain continuity across sessions without reloading full files.

## Required Workflow

### 1) Before starting any task
Run a semantic search first:

```text
memory_search(
  query="<resumen corto de la tarea>",
  project_id="<id del proyecto>",
  scope="project",
  limit=5,
  min_score=0.45
)
```

If the result includes relevant context, use it for planning and execution.
Pass `min_score` so weak cosine hits are not treated as facts. If nothing passes the floor, drop it or retry with a sharper query — do not use a 0.33 "best of garbage" hit.

### 2) During the task
Save milestones when they add meaningful continuity:

- Reusable bug and solution -> `scope="global"` or `scope="project"`.
- Architecture decision -> `scope="project"`.
- Current progress status -> `scope="session"`.
- Costly/repeatable result -> `scope="cache"`.
- Business decisions -> `scope="global"` or `scope="project"`.

When a new finding **corrects** an existing memory, call `memory_supersede(old_id, text)`.
Do not `memory_save` a parallel "this replaces X" note — the old vector stays searchable and competes with the correction.

Use `memory_update` only for tags/metadata or non-semantic edits of the same claim.
Use `memory_delete` when a memory is wrong and should not be kept for audit.

### 3) At the end of the task
Save a useful and actionable summary:

```text
memory_save(
  text="[TAREA COMPLETADA] <qué se hizo, resultado, próximos pasos>",
  project_id="<id del proyecto>",
  scope="session",
  tags=["<tecnologia>", "<tipo_tarea>"],
  session_id="<id_sesion>",
  source="agent"
)
```

Survey the store with `memory_stats`, not `memory_list(limit=100)`. List returns metadata only unless `include_text=True`.

## Current Tools Contract

- `memory_search(query, project_id="default", scope="project", limit=5, tags=None, session_id=None, min_score=None)`
- `memory_save(text, project_id="default", scope="project", tags=None, session_id=None, source="agent", metadata=None)`
- `memory_list(project_id="default", scope="project", limit=20, cursor=None, tags=None, session_id=None, include_text=False, include_superseded=False)`
- `memory_update(memory_id, text=None, tags=None, metadata=None)`
- `memory_delete(memory_id)`
- `memory_supersede(memory_id, text, tags=None, metadata=None, source="agent")`
- `memory_stats(project_id="default", scope=None)`
- `memory_related(memory_id, limit=5, min_score=None)`
- `memory_reindex(scope=None, dry_run=True)`
- `memory_dedupe(project_id="default", scope="project", threshold=0.95)`

Notes:

- `memory_list` returns `next_cursor` for pagination. Default omits `text`.
- Search runs over chunks and returns de-duplicated parent memories. Superseded parents are excluded.
- `memory_dedupe` reports clusters; it never deletes.
- `memory_reindex` defaults to `dry_run=True`.

## What to Save

Save information that is brief, specific, and reusable:

- Architecture decisions with rationale.
- Confirmed bugs and fixes.
- Project conventions (style, stack, constraints).
- Performance/debugging findings that are hard to reproduce.
- Any operation that may be of interest for future decisions

## What Not to Save

- Secrets or credentials in plain text.
- Long code blocks that already exist in the repository.
- Trivial information that does not improve future decisions.

## Recommended Scopes

- `project`: context and decisions for the current repository.
- `session`: temporary state of the active session.
- `global`: cross-project learnings.
- `cache`: expensive outputs worth reusing.

## Tagging Best Practices

Use between 2 and 5 tags per memory:

- Technology: `python`, `typescript`, `sql`.
- Domain: `architecture`, `bugfix`, `performance`.
- Component: `mcp`, `qdrant`, `api`.
