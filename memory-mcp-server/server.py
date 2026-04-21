import os
import uuid
from datetime import datetime, timezone
from typing import Any

from fastembed import TextEmbedding
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchAny, MatchValue, PointStruct, VectorParams


VALID_SCOPES = {"project", "session", "global", "cache"}
COLLECTION_BY_SCOPE = {
    "project": "memory_project",
    "session": "memory_session",
    "global": "memory_global",
    "cache": "memory_cache",
}


load_dotenv()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_scope(scope: str) -> str:
    if scope not in VALID_SCOPES:
        raise ValueError(f"scope invalido: {scope}. Usa uno de: {sorted(VALID_SCOPES)}")
    return scope


def _build_filter(project_id: str, scope: str, session_id: str | None, tags: list[str] | None) -> Filter:
    must = [
        FieldCondition(key="project_id", match=MatchValue(value=project_id)),
        FieldCondition(key="scope", match=MatchValue(value=scope)),
    ]

    if session_id:
        must.append(FieldCondition(key="session_id", match=MatchValue(value=session_id)))
    if tags:
        must.append(FieldCondition(key="tags", match=MatchAny(any=tags)))

    return Filter(must=must)


class MemoryStore:
    def __init__(self) -> None:
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY") or None
        embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.embedder = TextEmbedding(model_name=embedding_model)
        self._vector_size = len(self._embed_one("dimension_probe"))

    def _embed_one(self, text: str) -> list[float]:
        embedding = next(self.embedder.embed([text]))
        return embedding.tolist()

    def _ensure_collection(self, collection_name: str, vector_size: int) -> None:
        if self.client.collection_exists(collection_name=collection_name):
            return
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def save(
        self,
        text: str,
        project_id: str,
        scope: str,
        tags: list[str] | None,
        session_id: str | None,
        source: str,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        vector = self._embed_one(text)
        collection_name = COLLECTION_BY_SCOPE[scope]
        self._ensure_collection(collection_name=collection_name, vector_size=len(vector))

        memory_id = str(uuid.uuid4())
        payload: dict[str, Any] = {
            "memory_id": memory_id,
            "text": text,
            "project_id": project_id,
            "scope": scope,
            "session_id": session_id,
            "tags": tags or [],
            "source": source,
            "created_at": _utc_now_iso(),
            "metadata": metadata or {},
        }
        self.client.upsert(
            collection_name=collection_name,
            points=[PointStruct(id=memory_id, vector=vector, payload=payload)],
        )

        return {
            "ok": True,
            "memory_id": memory_id,
            "collection": collection_name,
            "project_id": project_id,
            "scope": scope,
        }

    def search(
        self,
        query: str,
        project_id: str,
        scope: str,
        limit: int,
        session_id: str | None,
        tags: list[str] | None,
    ) -> dict[str, Any]:
        vector = self._embed_one(query)
        collection_name = COLLECTION_BY_SCOPE[scope]
        self._ensure_collection(collection_name=collection_name, vector_size=len(vector))

        query_filter = _build_filter(project_id=project_id, scope=scope, session_id=session_id, tags=tags)
        if hasattr(self.client, "query_points"):
            points_result = self.client.query_points(
                collection_name=collection_name,
                query=vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            hits = points_result.points
        else:
            # Backward compatibility for older qdrant-client releases.
            hits = self.client.search(
                collection_name=collection_name,
                query_vector=vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

        results = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                {
                    "memory_id": payload.get("memory_id"),
                    "score": hit.score,
                    "text": payload.get("text"),
                    "tags": payload.get("tags", []),
                    "project_id": payload.get("project_id"),
                    "scope": payload.get("scope"),
                    "session_id": payload.get("session_id"),
                    "source": payload.get("source"),
                    "created_at": payload.get("created_at"),
                    "metadata": payload.get("metadata", {}),
                }
            )

        return {
            "ok": True,
            "collection": collection_name,
            "query": query,
            "count": len(results),
            "results": results,
        }

    def list_memories(
        self,
        project_id: str,
        scope: str,
        limit: int,
        cursor: str | int | None,
        session_id: str | None,
        tags: list[str] | None,
    ) -> dict[str, Any]:
        collection_name = COLLECTION_BY_SCOPE[scope]
        self._ensure_collection(collection_name=collection_name, vector_size=self._vector_size)

        query_filter = _build_filter(project_id=project_id, scope=scope, session_id=session_id, tags=tags)
        points, next_offset = self.client.scroll(
            collection_name=collection_name,
            scroll_filter=query_filter,
            limit=limit,
            offset=cursor,
            with_payload=True,
            with_vectors=False,
        )

        items = []
        for point in points:
            payload = point.payload or {}
            items.append(
                {
                    "memory_id": payload.get("memory_id"),
                    "text": payload.get("text"),
                    "tags": payload.get("tags", []),
                    "project_id": payload.get("project_id"),
                    "scope": payload.get("scope"),
                    "session_id": payload.get("session_id"),
                    "source": payload.get("source"),
                    "created_at": payload.get("created_at"),
                    "metadata": payload.get("metadata", {}),
                }
            )

        return {
            "ok": True,
            "collection": collection_name,
            "count": len(items),
            "items": items,
            "next_cursor": str(next_offset) if next_offset is not None else None,
        }


memory_store = MemoryStore()
mcp = FastMCP(os.getenv("MCP_SERVER_NAME", "memorIA"))


@mcp.tool(description="Guarda conocimiento nuevo en memoria persistente.")
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


@mcp.tool(description="Busca contexto semantico relevante en memoria.")
def memory_search(
    query: str,
    project_id: str = "default",
    scope: str = "project",
    limit: int = 5,
    tags: list[str] | None = None,
    session_id: str | None = None,
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
    )


@mcp.tool(description="Lista memorias almacenadas por scope/proyecto.")
def memory_list(
    project_id: str = "default",
    scope: str = "project",
    limit: int = 20,
    cursor: str | int | None = None,
    tags: list[str] | None = None,
    session_id: str | None = None,
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
    )


if __name__ == "__main__":
    mcp.run()
