import pytest

from tests.conftest import make_store


def _words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_save_chunks_long_text_and_search_returns_one_parent():
    # Arrange
    store = make_store(max_tokens=8, chunk_overlap=2)
    tail = "unique_tail_marker"
    text = f"{_words(20)} {tail}"

    # Act
    saved = store.save(text, "default", "project", ["t"], None, "agent", None)
    found = store.search(tail, "default", "project", 5, None, None)

    # Assert
    assert saved["chunk_count"] > 1
    assert found["count"] == 1
    assert found["results"][0]["memory_id"] == saved["memory_id"]
    assert tail in found["results"][0]["text"]


def test_min_score_drops_weak_hits():
    # Arrange
    store = make_store()
    store.save("alpha beta gamma", "default", "project", None, None, "agent", None)

    # Act
    unfiltered = store.search("unrelated omega zzq", "default", "project", 5, None, None)
    filtered = store.search(
        "unrelated omega zzq", "default", "project", 5, None, None, min_score=0.99
    )

    # Assert
    assert unfiltered["count"] >= 1
    assert filtered["count"] == 0


def test_supersede_excludes_old_memory_from_search():
    # Arrange
    store = make_store()
    original = store.save(
        "Elevate maybe another account",
        "default",
        "project",
        ["investigation"],
        None,
        "agent",
        None,
    )

    # Act
    replaced = store.supersede(
        original["memory_id"],
        "Elevate CONFIRMED it was the other account, user 422872",
    )
    live = store.search("Elevate account", "default", "project", 5, None, None)
    listed = store.list_memories("default", "project", 20, None, None, None)
    archived = store.list_memories(
        "default", "project", 20, None, None, None, include_superseded=True
    )

    # Assert
    ids = {item["memory_id"] for item in live["results"]}
    assert replaced["new_memory_id"] in ids
    assert original["memory_id"] not in ids
    assert listed["count"] == 1
    assert archived["count"] == 2


def test_update_tags_does_not_reembed_but_text_does():
    # Arrange
    store = make_store()
    saved = store.save("stable body", "default", "project", ["old"], None, "agent", None)

    # Act
    tagged = store.update(saved["memory_id"], tags=["new"])
    rewritten = store.update(saved["memory_id"], text="rewritten body")

    # Assert
    assert tagged["reembedded"] is False
    assert rewritten["reembedded"] is True
    listed = store.list_memories("default", "project", 5, None, None, None, include_text=True)
    assert listed["items"][0]["tags"] == ["new"]
    assert listed["items"][0]["text"] == "rewritten body"


def test_delete_removes_memory():
    # Arrange
    store = make_store()
    saved = store.save("to delete", "default", "project", None, None, "agent", None)

    # Act
    deleted = store.delete(saved["memory_id"])
    missing = store.delete(saved["memory_id"])

    # Assert
    assert deleted["deleted"] is True
    assert missing["deleted"] is False
    assert store.list_memories("default", "project", 5, None, None, None)["count"] == 0


def test_stats_truncation_exposure_before_and_after_reindex_shape():
    # Arrange
    store = make_store(max_tokens=8, chunk_overlap=2)
    store.save(_words(20) + " long investigation tail", "default", "project", ["x"], None, "agent", None)

    # Act
    stats = store.stats("default", "project")

    # Assert
    assert stats["embed_max_tokens"] == 8
    assert stats["truncation_exposure"]["over_window"] == 1
    assert stats["truncation_exposure"]["over_window_unchunked"] == 0
    assert stats["live_count"] == 1


def test_legacy_unchunked_point_counts_as_truncation_exposure():
    # Arrange
    store = make_store(max_tokens=8, chunk_overlap=2)
    saved = store.save("short", "default", "project", None, None, "agent", None)
    _scope, collection, points = store._locate(saved["memory_id"])
    for point in points:
        point.payload["token_count"] = 400
        point.payload["chunk_count"] = 1
        store.client.set_payload(
            collection_name=collection,
            payload={"token_count": 400, "chunk_count": 1},
            points=[point.id],
        )

    # Act
    stats = store.stats("default", "project")

    # Assert
    assert stats["truncation_exposure"]["over_window_unchunked"] == 1
    assert stats["truncation_exposure"]["ratio"] == 1.0


def test_related_returns_neighbor_not_self():
    # Arrange
    store = make_store()
    first = store.save("red apple fruit", "default", "project", None, None, "agent", None)
    second = store.save("red apple juice", "default", "project", None, None, "agent", None)

    # Act
    related = store.related(first["memory_id"], limit=5)

    # Assert
    ids = {item["memory_id"] for item in related["results"]}
    assert second["memory_id"] in ids
    assert first["memory_id"] not in ids


def test_dedupe_reports_cluster_without_deleting():
    # Arrange
    store = make_store()
    first = store.save("same payload here", "default", "project", None, None, "agent", None)
    second = store.save("same payload here", "default", "project", None, None, "agent", None)

    # Act
    report = store.dedupe("default", "project", threshold=0.95)
    listed = store.list_memories("default", "project", 10, None, None, None)

    # Assert
    assert report["cluster_count"] == 1
    assert set(report["clusters"][0]["memory_ids"]) == {first["memory_id"], second["memory_id"]}
    assert listed["count"] == 2


def test_reindex_dry_run_does_not_rewrite():
    # Arrange
    store = make_store(max_tokens=8, chunk_overlap=2)
    saved = store.save(_words(18), "default", "project", None, None, "agent", None)
    before_ids = {point.id for point in store._points_for_memory("memory_project", saved["memory_id"])}

    # Act
    result = store.reindex(scope="project", dry_run=True)
    after_ids = {point.id for point in store._points_for_memory("memory_project", saved["memory_id"])}

    # Assert
    assert result["dry_run"] is True
    assert result["reembedded"] == 1
    assert result["chunks_created"] > 1
    assert before_ids == after_ids


def test_update_rejected_on_superseded_memory():
    # Arrange
    store = make_store()
    original = store.save("old claim", "default", "project", None, None, "agent", None)
    store.supersede(original["memory_id"], "corrected claim")

    # Act / Assert
    with pytest.raises(ValueError, match="reemplazada"):
        store.update(original["memory_id"], text="should fail")
