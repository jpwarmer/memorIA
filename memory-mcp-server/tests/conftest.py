from qdrant_client import QdrantClient

from store import MemoryStore
from tests.fakes import FakeEmbedder, FakeTokenizer


def make_store(max_tokens: int = 8, chunk_overlap: int = 2) -> MemoryStore:
    return MemoryStore(
        client=QdrantClient(location=":memory:"),
        embedder=FakeEmbedder(),
        tokenizer=FakeTokenizer(),
        max_tokens=max_tokens,
        chunk_overlap=chunk_overlap,
    )
