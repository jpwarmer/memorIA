import math
import os
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    IsEmptyCondition,
    MatchAny,
    MatchValue,
    PayloadField,
    PointStruct,
    VectorParams,
)

from embeddings import (
    chunk_overlap_from_env,
    chunk_text,
    embed_max_tokens_from_env,
    make_text_embedder,
    token_count,
    tokenizer_of,
)


VALID_SCOPES = {"project", "session", "global", "cache"}
COLLECTION_BY_SCOPE = {
    "cache": "memory_cache",
    "global": "memory_global",
    "project": "memory_project",
    "session": "memory_session",
}
_POINT_NS = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")
_STATUS_ACTIVE = "active"
_STATUS_SUPERSEDED = "superseded"
_TOKEN_BUCKETS = (64, 128, 256, 512, 1024, 2048)
_STATS_DUPLICATE_THRESHOLD = 0.95
_SEARCH_OVERFETCH = 8


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_scope(scope: str) -> str:
    if scope not in VALID_SCOPES:
        raise ValueError(f"scope invalido: {scope}. Usa uno de: {sorted(VALID_SCOPES)}")
    return scope


def _chunk_point_id(memory_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_POINT_NS, f"{memory_id}:{chunk_index}"))


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)


def _preview(text: str, chars: int = 80) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= chars:
        return compact
    return compact[: chars - 1] + "…"


def _histogram(values: list[int], buckets: tuple[int, ...]) -> dict[str, int]:
    counts = {f"<= {edge}": 0 for edge in buckets}
    counts[f"> {buckets[-1]}"] = 0
    for value in values:
        placed = False
        for edge in buckets:
            if value <= edge:
                counts[f"<= {edge}"] += 1
                placed = True
                break
        if not placed:
            counts[f"> {buckets[-1]}"] += 1
    return counts


def _build_filter(
    project_id: str | None,
    scope: str | None,
    session_id: str | None,
    tags: list[str] | None,
    include_superseded: bool,
    parents_only: bool,
) -> Filter:
    must: list[Any] = []
    if project_id:
        must.append(FieldCondition(key="project_id", match=MatchValue(value=project_id)))
    if scope:
        must.append(FieldCondition(key="scope", match=MatchValue(value=scope)))
    if session_id:
        must.append(FieldCondition(key="session_id", match=MatchValue(value=session_id)))
    if tags:
        must.append(FieldCondition(key="tags", match=MatchAny(any=tags)))
    must_not: list[Any] = []
    if not include_superseded:
        must_not.append(FieldCondition(key="status", match=MatchValue(value=_STATUS_SUPERSEDED)))
    should: list[Any] | None = None
    if parents_only:
        should = [
            FieldCondition(key="chunk_index", match=MatchValue(value=0)),
            IsEmptyCondition(is_empty=PayloadField(key="chunk_index")),
        ]
    return Filter(
        must=must or None,
        must_not=must_not or None,
        should=should,
    )


def _public_record(payload: dict[str, Any], score: float | None = None) -> dict[str, Any]:
    record = {
        "memory_id": payload.get("memory_id"),
        "text": payload.get("text"),
        "tags": payload.get("tags", []),
        "project_id": payload.get("project_id"),
        "scope": payload.get("scope"),
        "session_id": payload.get("session_id"),
        "source": payload.get("source"),
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
        "metadata": payload.get("metadata", {}),
        "status": payload.get("status", _STATUS_ACTIVE),
        "chunk_count": payload.get("chunk_count", 1),
        "token_count": payload.get("token_count"),
        "superseded_by": payload.get("superseded_by"),
        "supersedes": (payload.get("metadata") or {}).get("supersedes"),
    }
    if score is not None:
        record["score"] = score
        record["matched_chunk_index"] = payload.get("chunk_index", 0)
    return record


class MemoryStore:
    def __init__(
        self,
        client: QdrantClient | None = None,
        embedder: Any | None = None,
        tokenizer: Any | None = None,
        max_tokens: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY") or None
        embedding_model = os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        self.max_tokens = max_tokens if max_tokens is not None else embed_max_tokens_from_env()
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else chunk_overlap_from_env()
        self.client = client or QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.embedder = embedder or make_text_embedder(embedding_model, self.max_tokens)
        self.tokenizer = tokenizer or tokenizer_of(self.embedder)
        self._vector_size = len(self._embed_one("dimension_probe"))

    def _embed_one(self, text: str) -> list[float]:
        embedding = next(self.embedder.embed([text]))
        return embedding.tolist()

    def _ensure_collection(self, collection_name: str) -> None:
        if self.client.collection_exists(collection_name=collection_name):
            return
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
        )

    def _iter_collections(self, scope: str | None) -> list[tuple[str, str]]:
        scopes = [_safe_scope(scope)] if scope else list(COLLECTION_BY_SCOPE)
        found: list[tuple[str, str]] = []
        for item in scopes:
            name = COLLECTION_BY_SCOPE[item]
            if self.client.collection_exists(collection_name=name):
                found.append((item, name))
        return found

    def _scroll(
        self,
        collection_name: str,
        query_filter: Filter | None,
        with_vectors: bool = False,
        limit: int = 64,
        offset: Any = None,
    ) -> tuple[list[Any], Any]:
        return self.client.scroll(
            collection_name=collection_name,
            scroll_filter=query_filter,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=with_vectors,
        )

    def _scroll_all(
        self,
        collection_name: str,
        query_filter: Filter | None,
        with_vectors: bool = False,
    ) -> list[Any]:
        offset = None
        points: list[Any] = []
        while True:
            batch, offset = self._scroll(
                collection_name=collection_name,
                query_filter=query_filter,
                with_vectors=with_vectors,
                offset=offset,
            )
            points.extend(batch)
            if offset is None:
                break
        return points

    def _points_for_memory(self, collection_name: str, memory_id: str) -> list[Any]:
        query_filter = Filter(must=[FieldCondition(key="memory_id", match=MatchValue(value=memory_id))])
        return self._scroll_all(collection_name, query_filter, with_vectors=True)

    def _locate(self, memory_id: str) -> tuple[str, str, list[Any]]:
        for scope, collection_name in self._iter_collections(None):
            points = self._points_for_memory(collection_name, memory_id)
            if points:
                return scope, collection_name, points
        raise ValueError(f"memory_id no encontrado: {memory_id}")

    def _parent_payload(self, points: list[Any]) -> dict[str, Any]:
        ordered = sorted(points, key=lambda point: (point.payload or {}).get("chunk_index", 0))
        return dict(ordered[0].payload or {})

    def _delete_points(self, collection_name: str, points: list[Any]) -> None:
        ids = [point.id for point in points]
        if ids:
            self.client.delete(collection_name=collection_name, points_selector=ids)

    def _write_chunks(
        self,
        collection_name: str,
        memory_id: str,
        text: str,
        payload_base: dict[str, Any],
        replace_points: list[Any] | None = None,
    ) -> int:
        chunks = chunk_text(self.tokenizer, text, self.max_tokens, self.chunk_overlap)
        tokens = token_count(self.tokenizer, text, self.max_tokens)
        points = []
        new_ids = set()
        for index, chunk in enumerate(chunks):
            point_id = _chunk_point_id(memory_id, index)
            new_ids.add(point_id)
            payload = {
                **payload_base,
                "memory_id": memory_id,
                "text": text,
                "chunk_text": chunk,
                "chunk_index": index,
                "chunk_count": len(chunks),
                "token_count": tokens,
            }
            points.append(
                PointStruct(
                    id=point_id,
                    vector=self._embed_one(chunk),
                    payload=payload,
                )
            )
        self.client.upsert(collection_name=collection_name, points=points)
        if replace_points:
            stale = [point.id for point in replace_points if point.id not in new_ids]
            if stale:
                self.client.delete(collection_name=collection_name, points_selector=stale)
        return len(chunks)

    def _query(
        self,
        collection_name: str,
        vector: list[float],
        query_filter: Filter,
        limit: int,
        score_threshold: float | None,
    ) -> list[Any]:
        if hasattr(self.client, "query_points"):
            points_result = self.client.query_points(
                collection_name=collection_name,
                query=vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False,
            )
            return list(points_result.points)
        return list(
            self.client.search(
                collection_name=collection_name,
                query_vector=vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False,
            )
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
        memory_id: str | None = None,
        created_at: str | None = None,
        status: str = _STATUS_ACTIVE,
        superseded_by: str | None = None,
    ) -> dict[str, Any]:
        collection_name = COLLECTION_BY_SCOPE[scope]
        self._ensure_collection(collection_name)
        memory_id = memory_id or str(uuid.uuid4())
        now = _utc_now_iso()
        payload_base: dict[str, Any] = {
            "project_id": project_id,
            "scope": scope,
            "session_id": session_id,
            "tags": tags or [],
            "source": source,
            "created_at": created_at or now,
            "updated_at": now,
            "metadata": metadata or {},
            "status": status,
            "superseded_by": superseded_by,
        }
        chunk_count = self._write_chunks(collection_name, memory_id, text, payload_base)
        return {
            "ok": True,
            "memory_id": memory_id,
            "collection": collection_name,
            "project_id": project_id,
            "scope": scope,
            "chunk_count": chunk_count,
            "token_count": token_count(self.tokenizer, text, self.max_tokens),
        }

    def search(
        self,
        query: str,
        project_id: str,
        scope: str,
        limit: int,
        session_id: str | None,
        tags: list[str] | None,
        min_score: float | None = None,
        include_superseded: bool = False,
    ) -> dict[str, Any]:
        vector = self._embed_one(query)
        collection_name = COLLECTION_BY_SCOPE[scope]
        self._ensure_collection(collection_name)
        query_filter = _build_filter(
            project_id=project_id,
            scope=scope,
            session_id=session_id,
            tags=tags,
            include_superseded=include_superseded,
            parents_only=False,
        )
        raw_limit = min(max(limit * _SEARCH_OVERFETCH, limit), 100)
        hits = self._query(
            collection_name=collection_name,
            vector=vector,
            query_filter=query_filter,
            limit=raw_limit,
            score_threshold=min_score,
        )
        best: dict[str, dict[str, Any]] = {}
        for hit in hits:
            payload = hit.payload or {}
            memory_id = payload.get("memory_id")
            if not memory_id:
                continue
            if min_score is not None and hit.score < min_score:
                continue
            current = best.get(memory_id)
            if current is None or hit.score > current["score"]:
                best[memory_id] = _public_record(payload, score=hit.score)
        results = sorted(best.values(), key=lambda item: item["score"], reverse=True)[:limit]
        return {
            "ok": True,
            "collection": collection_name,
            "query": query,
            "min_score": min_score,
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
        include_text: bool = False,
        include_superseded: bool = False,
    ) -> dict[str, Any]:
        collection_name = COLLECTION_BY_SCOPE[scope]
        self._ensure_collection(collection_name)
        query_filter = _build_filter(
            project_id=project_id,
            scope=scope,
            session_id=session_id,
            tags=tags,
            include_superseded=include_superseded,
            parents_only=True,
        )
        points, next_offset = self._scroll(
            collection_name=collection_name,
            query_filter=query_filter,
            limit=limit,
            offset=cursor,
        )
        items = []
        for point in points:
            record = _public_record(point.payload or {})
            if not include_text:
                record.pop("text", None)
            items.append(record)
        return {
            "ok": True,
            "collection": collection_name,
            "count": len(items),
            "items": items,
            "next_cursor": str(next_offset) if next_offset is not None else None,
        }

    def update(
        self,
        memory_id: str,
        text: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if text is None and tags is None and metadata is None:
            raise ValueError("nada para actualizar")
        _scope, collection_name, points = self._locate(memory_id)
        payload = self._parent_payload(points)
        if payload.get("status") == _STATUS_SUPERSEDED:
            raise ValueError(
                f"memory {memory_id} fue reemplazada por {payload.get('superseded_by')}; usa esa"
            )
        new_text = text.strip() if text is not None else payload.get("text") or ""
        if not new_text:
            raise ValueError("text no puede estar vacio")
        new_tags = tags if tags is not None else payload.get("tags", [])
        new_metadata = metadata if metadata is not None else payload.get("metadata", {})
        reembedded = text is not None and new_text != (payload.get("text") or "")
        now = _utc_now_iso()
        if reembedded:
            self._write_chunks(
                collection_name,
                memory_id,
                new_text,
                {
                    "project_id": payload.get("project_id"),
                    "scope": payload.get("scope"),
                    "session_id": payload.get("session_id"),
                    "tags": new_tags,
                    "source": payload.get("source"),
                    "created_at": payload.get("created_at"),
                    "updated_at": now,
                    "metadata": new_metadata,
                    "status": _STATUS_ACTIVE,
                    "superseded_by": None,
                },
                replace_points=points,
            )
        else:
            patch = {
                "tags": new_tags,
                "metadata": new_metadata,
                "updated_at": now,
            }
            self.client.set_payload(
                collection_name=collection_name,
                payload=patch,
                points=[point.id for point in points],
            )
        return {"ok": True, "memory_id": memory_id, "reembedded": reembedded}

    def delete(self, memory_id: str) -> dict[str, Any]:
        try:
            _scope, collection_name, points = self._locate(memory_id)
        except ValueError:
            return {"ok": True, "deleted": False, "memory_id": memory_id}
        self._delete_points(collection_name, points)
        return {"ok": True, "deleted": True, "memory_id": memory_id}

    def supersede(
        self,
        memory_id: str,
        text: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        source: str = "agent",
    ) -> dict[str, Any]:
        _scope, collection_name, points = self._locate(memory_id)
        payload = self._parent_payload(points)
        if payload.get("status") == _STATUS_SUPERSEDED:
            raise ValueError(
                f"memory {memory_id} ya fue reemplazada por {payload.get('superseded_by')}"
            )
        merged_metadata = dict(payload.get("metadata") or {})
        if metadata:
            merged_metadata.update(metadata)
        merged_metadata["supersedes"] = memory_id
        created = self.save(
            text=text,
            project_id=payload.get("project_id") or "default",
            scope=payload.get("scope") or "project",
            tags=tags if tags is not None else payload.get("tags") or [],
            session_id=payload.get("session_id"),
            source=source,
            metadata=merged_metadata,
        )
        self.client.set_payload(
            collection_name=collection_name,
            payload={
                "status": _STATUS_SUPERSEDED,
                "superseded_by": created["memory_id"],
                "updated_at": _utc_now_iso(),
            },
            points=[point.id for point in points],
        )
        return {
            "ok": True,
            "new_memory_id": created["memory_id"],
            "superseded": memory_id,
            "chunk_count": created["chunk_count"],
        }

    def related(
        self,
        memory_id: str,
        limit: int = 5,
        min_score: float | None = None,
    ) -> dict[str, Any]:
        _scope, collection_name, points = self._locate(memory_id)
        payload = self._parent_payload(points)
        query_filter = _build_filter(
            project_id=payload.get("project_id"),
            scope=payload.get("scope"),
            session_id=None,
            tags=None,
            include_superseded=False,
            parents_only=False,
        )
        best: dict[str, dict[str, Any]] = {}
        raw_limit = min(max(limit * _SEARCH_OVERFETCH, limit), 100)
        for point in points:
            vector = point.vector
            if vector is None:
                continue
            if isinstance(vector, dict):
                continue
            hits = self._query(
                collection_name=collection_name,
                vector=list(vector),
                query_filter=query_filter,
                limit=raw_limit,
                score_threshold=min_score,
            )
            for hit in hits:
                hit_payload = hit.payload or {}
                other_id = hit_payload.get("memory_id")
                if not other_id or other_id == memory_id:
                    continue
                current = best.get(other_id)
                if current is None or hit.score > current["score"]:
                    best[other_id] = _public_record(hit_payload, score=hit.score)
        results = sorted(best.values(), key=lambda item: item["score"], reverse=True)[:limit]
        return {"ok": True, "memory_id": memory_id, "count": len(results), "results": results}

    def _collect_parents(
        self,
        project_id: str | None,
        scope: str | None,
        include_superseded: bool,
        with_vectors: bool,
    ) -> list[tuple[str, dict[str, Any], list[float] | None]]:
        parents: list[tuple[str, dict[str, Any], list[float] | None]] = []
        seen: set[str] = set()
        query_filter = _build_filter(
            project_id=project_id,
            scope=None,
            session_id=None,
            tags=None,
            include_superseded=include_superseded,
            parents_only=True,
        )
        for item_scope, collection_name in self._iter_collections(scope):
            if scope is None:
                scoped = _build_filter(
                    project_id=project_id,
                    scope=item_scope,
                    session_id=None,
                    tags=None,
                    include_superseded=include_superseded,
                    parents_only=True,
                )
            else:
                scoped = query_filter
            for point in self._scroll_all(collection_name, scoped, with_vectors=with_vectors):
                payload = point.payload or {}
                memory_id = payload.get("memory_id")
                if not memory_id or memory_id in seen:
                    continue
                seen.add(memory_id)
                vector = point.vector if with_vectors else None
                if isinstance(vector, dict):
                    vector = None
                parents.append((collection_name, payload, list(vector) if vector is not None else None))
        return parents

    def stats(self, project_id: str | None = "default", scope: str | None = None) -> dict[str, Any]:
        if scope is not None:
            _safe_scope(scope)
        live = self._collect_parents(project_id, scope, include_superseded=False, with_vectors=True)
        archived = self._collect_parents(project_id, scope, include_superseded=True, with_vectors=False)
        superseded_count = sum(
            1 for _c, payload, _v in archived if payload.get("status") == _STATUS_SUPERSEDED
        )
        counts_by_scope: dict[str, int] = Counter()
        tag_counts: dict[str, int] = Counter()
        token_lengths: list[int] = []
        chunk_counts: list[int] = []
        created_at_values: list[str] = []
        over_window = 0
        over_window_unchunked = 0
        for _collection, payload, _vector in live:
            counts_by_scope[payload.get("scope") or "unknown"] += 1
            for tag in payload.get("tags") or []:
                tag_counts[str(tag)] += 1
            tokens = payload.get("token_count")
            if tokens is None:
                tokens = token_count(self.tokenizer, payload.get("text") or "", self.max_tokens)
            token_lengths.append(int(tokens))
            chunks = int(payload.get("chunk_count") or 1)
            chunk_counts.append(chunks)
            if payload.get("created_at"):
                created_at_values.append(payload["created_at"])
            if int(tokens) > self.max_tokens:
                over_window += 1
                if chunks <= 1:
                    over_window_unchunked += 1
        live_count = len(live)
        duplicate_candidates = self._duplicate_pairs(live, _STATS_DUPLICATE_THRESHOLD)[:20]
        return {
            "ok": True,
            "project_id": project_id,
            "scope": scope,
            "embed_max_tokens": self.max_tokens,
            "chunk_overlap": self.chunk_overlap,
            "counts_by_scope": dict(counts_by_scope),
            "live_count": live_count,
            "superseded_count": superseded_count,
            "token_length_histogram": _histogram(token_lengths, _TOKEN_BUCKETS),
            "chunks_per_memory_histogram": _histogram(chunk_counts, (1, 2, 4, 8, 16)),
            "truncation_exposure": {
                "parent_count": live_count,
                "over_window": over_window,
                "over_window_unchunked": over_window_unchunked,
                "ratio": round(over_window_unchunked / live_count, 4) if live_count else 0.0,
            },
            "tag_histogram": dict(tag_counts.most_common()),
            "oldest": min(created_at_values) if created_at_values else None,
            "newest": max(created_at_values) if created_at_values else None,
            "duplicate_candidates": duplicate_candidates,
        }

    def _duplicate_pairs(
        self,
        parents: list[tuple[str, dict[str, Any], list[float] | None]],
        threshold: float,
    ) -> list[dict[str, Any]]:
        pairs: list[dict[str, Any]] = []
        for i, (_c1, left, left_vec) in enumerate(parents):
            if not left_vec:
                continue
            for _c2, right, right_vec in parents[i + 1 :]:
                if not right_vec:
                    continue
                score = _cosine(left_vec, right_vec)
                if score < threshold:
                    continue
                pairs.append(
                    {
                        "memory_ids": [left.get("memory_id"), right.get("memory_id")],
                        "score": round(score, 4),
                        "previews": [
                            _preview(left.get("text") or ""),
                            _preview(right.get("text") or ""),
                        ],
                    }
                )
        pairs.sort(key=lambda item: item["score"], reverse=True)
        return pairs

    def reindex(self, scope: str | None = None, dry_run: bool = True) -> dict[str, Any]:
        if scope is not None:
            _safe_scope(scope)
        before = self.stats(project_id=None, scope=scope)
        reembedded = 0
        chunks_created = 0
        for collection_name, payload, _vector in self._collect_parents(
            project_id=None,
            scope=scope,
            include_superseded=True,
            with_vectors=False,
        ):
            text = payload.get("text") or ""
            if not text.strip():
                continue
            planned = len(chunk_text(self.tokenizer, text, self.max_tokens, self.chunk_overlap))
            chunks_created += planned
            reembedded += 1
            if dry_run:
                continue
            memory_id = payload["memory_id"]
            points = self._points_for_memory(collection_name, memory_id)
            self._write_chunks(
                collection_name,
                memory_id,
                text,
                {
                    "project_id": payload.get("project_id"),
                    "scope": payload.get("scope"),
                    "session_id": payload.get("session_id"),
                    "tags": payload.get("tags") or [],
                    "source": payload.get("source"),
                    "created_at": payload.get("created_at"),
                    "updated_at": _utc_now_iso(),
                    "metadata": payload.get("metadata") or {},
                    "status": payload.get("status") or _STATUS_ACTIVE,
                    "superseded_by": payload.get("superseded_by"),
                },
                replace_points=points,
            )
        after = before if dry_run else self.stats(project_id=None, scope=scope)
        return {
            "ok": True,
            "dry_run": dry_run,
            "reembedded": reembedded,
            "chunks_created": chunks_created,
            "before": {
                "live_count": before["live_count"],
                "truncation_exposure": before["truncation_exposure"],
                "token_length_histogram": before["token_length_histogram"],
            },
            "after": {
                "live_count": after["live_count"],
                "truncation_exposure": after["truncation_exposure"],
                "token_length_histogram": after["token_length_histogram"],
            },
        }

    def dedupe(
        self,
        project_id: str = "default",
        scope: str = "project",
        threshold: float = 0.95,
    ) -> dict[str, Any]:
        _safe_scope(scope)
        threshold = min(0.999, max(0.5, threshold))
        parents = self._collect_parents(
            project_id=project_id,
            scope=scope,
            include_superseded=False,
            with_vectors=True,
        )
        parent_by_id = {payload["memory_id"]: payload for _c, payload, _v in parents if payload.get("memory_id")}
        ids = [payload["memory_id"] for _c, payload, _v in parents if payload.get("memory_id")]
        parent = {memory_id: memory_id for memory_id in ids}

        def find(item: str) -> str:
            while parent[item] != item:
                parent[item] = parent[parent[item]]
                item = parent[item]
            return item

        def union(left: str, right: str) -> None:
            root_left, root_right = find(left), find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        for i, (_c1, left, left_vec) in enumerate(parents):
            left_id = left.get("memory_id")
            if not left_id or not left_vec:
                continue
            for _c2, right, right_vec in parents[i + 1 :]:
                right_id = right.get("memory_id")
                if not right_id or not right_vec:
                    continue
                if _cosine(left_vec, right_vec) >= threshold:
                    union(left_id, right_id)
        grouped: dict[str, list[str]] = defaultdict(list)
        for memory_id in ids:
            grouped[find(memory_id)].append(memory_id)
        clusters = []
        for members in grouped.values():
            if len(members) < 2:
                continue
            clusters.append(
                {
                    "size": len(members),
                    "memory_ids": members,
                    "previews": [_preview(parent_by_id[item].get("text") or "") for item in members],
                }
            )
        clusters.sort(key=lambda item: item["size"], reverse=True)
        return {
            "ok": True,
            "project_id": project_id,
            "scope": scope,
            "threshold": threshold,
            "cluster_count": len(clusters),
            "clusters": clusters,
        }
