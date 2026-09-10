import os
from typing import Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from store import MemoryStore, _safe_scope

load_dotenv()


memory_store = MemoryStore()
mcp = FastMCP(os.getenv("MCP_SERVER_NAME", "memorIA"))


@mcp.tool(description="Guarda conocimiento nuevo en memoria persistente. Textos largos se parten en chunks.")
def memory_save(
    text: str,
    project_id: str = "default",
    scope: str = "project",
    tags: list[str] | None = None,
    session_id: str | None = None,
    source: str = "agent",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scope = _safe_scope(scope)
    if not text or not text.strip():
        raise ValueError("text no puede estar vacio")
    return memory_store.save(
        text=text.strip(),
        project_id=project_id,
        scope=scope,
        tags=tags,
        session_id=session_id,
        source=source,
        metadata=metadata,
    )


@mcp.tool(description="Busca contexto semantico. Busca sobre chunks y devuelve padres deduplicados.")
def memory_search(
    query: str,
    project_id: str = "default",
    scope: str = "project",
    limit: int = 5,
    tags: list[str] | None = None,
    session_id: str | None = None,
    min_score: float | None = None,
) -> dict[str, Any]:
    scope = _safe_scope(scope)
    if not query or not query.strip():
        raise ValueError("query no puede estar vacio")
    limit = max(1, min(limit, 50))
    return memory_store.search(
        query=query.strip(),
        project_id=project_id,
        scope=scope,
        limit=limit,
        session_id=session_id,
        tags=tags,
        min_score=min_score,
    )


@mcp.tool(description="Lista memorias. Por defecto sin texto completo para no reventar el contexto.")
def memory_list(
    project_id: str = "default",
    scope: str = "project",
    limit: int = 20,
    cursor: str | int | None = None,
    tags: list[str] | None = None,
    session_id: str | None = None,
    include_text: bool = False,
    include_superseded: bool = False,
) -> dict[str, Any]:
    scope = _safe_scope(scope)
    limit = max(1, min(limit, 100))
    if isinstance(cursor, str):
        cursor = cursor.strip() or None
    return memory_store.list_memories(
        project_id=project_id,
        scope=scope,
        limit=limit,
        cursor=cursor,
        tags=tags,
        session_id=session_id,
        include_text=include_text,
        include_superseded=include_superseded,
    )


@mcp.tool(description="Actualiza texto, tags o metadata de una memoria. Re-embedde si cambia el texto.")
def memory_update(
    memory_id: str,
    text: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not memory_id or not memory_id.strip():
        raise ValueError("memory_id no puede estar vacio")
    return memory_store.update(
        memory_id=memory_id.strip(),
        text=text.strip() if isinstance(text, str) else text,
        tags=tags,
        metadata=metadata,
    )


@mcp.tool(description="Borra una memoria y todos sus chunks.")
def memory_delete(memory_id: str) -> dict[str, Any]:
    if not memory_id or not memory_id.strip():
        raise ValueError("memory_id no puede estar vacio")
    return memory_store.delete(memory_id=memory_id.strip())


@mcp.tool(description="Escribe una memoria nueva y marca la anterior como superseded (excluida de search, queda para audit).")
def memory_supersede(
    memory_id: str,
    text: str,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    source: str = "agent",
) -> dict[str, Any]:
    if not memory_id or not memory_id.strip():
        raise ValueError("memory_id no puede estar vacio")
    if not text or not text.strip():
        raise ValueError("text no puede estar vacio")
    return memory_store.supersede(
        memory_id=memory_id.strip(),
        text=text.strip(),
        tags=tags,
        metadata=metadata,
        source=source,
    )


@mcp.tool(description="Resumen del store: counts, histogramas, truncation_exposure, candidatos a duplicado. No dumpa textos.")
def memory_stats(project_id: str = "default", scope: str | None = None) -> dict[str, Any]:
    if scope is not None:
        scope = _safe_scope(scope)
    return memory_store.stats(project_id=project_id, scope=scope)


@mcp.tool(description="Vecinos mas cercanos de una memoria existente.")
def memory_related(
    memory_id: str,
    limit: int = 5,
    min_score: float | None = None,
) -> dict[str, Any]:
    if not memory_id or not memory_id.strip():
        raise ValueError("memory_id no puede estar vacio")
    limit = max(1, min(limit, 20))
    return memory_store.related(memory_id=memory_id.strip(), limit=limit, min_score=min_score)


@mcp.tool(description="Re-chunk y re-embed del store. dry_run=True por defecto.")
def memory_reindex(scope: str | None = None, dry_run: bool = True) -> dict[str, Any]:
    if scope is not None:
        scope = _safe_scope(scope)
    return memory_store.reindex(scope=scope, dry_run=dry_run)


@mcp.tool(description="Reporta clusters de duplicados. Nunca borra.")
def memory_dedupe(
    project_id: str = "default",
    scope: str = "project",
    threshold: float = 0.95,
) -> dict[str, Any]:
    scope = _safe_scope(scope)
    return memory_store.dedupe(project_id=project_id, scope=scope, threshold=threshold)


if __name__ == "__main__":
    mcp.run()
