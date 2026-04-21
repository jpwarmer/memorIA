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
  limit=5
)
```

If the result includes relevant context, use it for planning and execution.

### 2) During the task
Save milestones when they add meaningful continuity:

- Reusable bug and solution -> `scope="global"` or `scope="project"`.
- Architecture decision -> `scope="project"`.
- Current progress status -> `scope="session"`.
- Costly/repeatable result -> `scope="cache"`.
- Business decisions -> `scope="global"` or `scope="project"`.

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

## Current Tools Contract

- `memory_search(query, project_id="default", scope="project", limit=5, tags=None, session_id=None)`
- `memory_save(text, project_id="default", scope="project", tags=None, session_id=None, source="agent", metadata=None)`
- `memory_list(project_id="default", scope="project", limit=20, cursor=None, tags=None, session_id=None)`

Note: `memory_list` returns `next_cursor`; pass that value in `cursor` to paginate.

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
