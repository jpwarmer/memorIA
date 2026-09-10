import os
import shutil
from pathlib import Path
from typing import Any, Protocol

from fastembed import TextEmbedding
from fastembed.common.utils import define_cache_dir


SPECIAL_TOKEN_RESERVE = 2
DEFAULT_MAX_TOKENS = 128
DEFAULT_CHUNK_OVERLAP = 16
_PERSISTENT_CACHE_DIR = str(Path.home() / ".cache" / "fastembed")


class TokenEncoder(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> Any: ...
    def decode(self, ids: list[int]) -> str: ...
    def enable_truncation(self, max_length: int, **kwargs: Any) -> None: ...
    def no_truncation(self) -> None: ...

    @property
    def truncation(self) -> dict[str, Any] | None: ...


def embed_max_tokens_from_env() -> int:
    raw = os.getenv("EMBED_MAX_TOKENS", str(DEFAULT_MAX_TOKENS))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"EMBED_MAX_TOKENS invalido: {raw}") from exc
    if value < 8:
        raise ValueError("EMBED_MAX_TOKENS debe ser >= 8")
    return value


def chunk_overlap_from_env() -> int:
    raw = os.getenv("EMBED_CHUNK_OVERLAP", str(DEFAULT_CHUNK_OVERLAP))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"EMBED_CHUNK_OVERLAP invalido: {raw}") from exc
    return max(0, value)


def _fastembed_hf_repo_id(model_name: str) -> str | None:
    lower = model_name.lower()
    for desc in TextEmbedding.list_supported_models():
        if desc["model"].lower() == lower:
            hf = (desc.get("sources") or {}).get("hf")
            if isinstance(hf, str) and hf.strip():
                return hf
    return None


def _clear_fastembed_cache_for_hf_repo(hf_repo_id: str) -> None:
    marker = f"models--{hf_repo_id.replace('/', '--')}"
    for cache_root in [Path(_PERSISTENT_CACHE_DIR), Path(define_cache_dir(None))]:
        stale = cache_root / marker
        if stale.exists():
            shutil.rmtree(stale, ignore_errors=True)


def _model_cache_corrupted(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "no_suchfile" in msg
        or ("file doesn't exist" in msg and ".onnx" in msg)
        or "could not find tokenizer_config.json" in msg
    )


def tokenizer_of(embedder: TextEmbedding) -> TokenEncoder:
    tokenizer = getattr(getattr(embedder, "model", None), "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("el embedder no expone tokenizer; no se puede fijar truncacion")
    return tokenizer


def apply_truncation(tokenizer: TokenEncoder, max_tokens: int) -> None:
    tokenizer.enable_truncation(max_length=max_tokens)


def make_text_embedder(model_name: str, max_tokens: int) -> TextEmbedding:
    try:
        embedder = TextEmbedding(model_name=model_name, cache_dir=_PERSISTENT_CACHE_DIR)
    except Exception as e:
        if not _model_cache_corrupted(e):
            raise
        hf = _fastembed_hf_repo_id(model_name)
        if hf is None:
            raise
        _clear_fastembed_cache_for_hf_repo(hf)
        embedder = TextEmbedding(model_name=model_name, cache_dir=_PERSISTENT_CACHE_DIR)
    apply_truncation(tokenizer_of(embedder), max_tokens)
    return embedder


def restore_truncation(tokenizer: TokenEncoder, previous: dict[str, Any] | None, max_tokens: int) -> None:
    if previous:
        tokenizer.enable_truncation(**previous)
        return
    tokenizer.enable_truncation(max_length=max_tokens)


def token_count(tokenizer: TokenEncoder, text: str, max_tokens: int) -> int:
    previous = tokenizer.truncation
    tokenizer.no_truncation()
    try:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)
    finally:
        restore_truncation(tokenizer, previous, max_tokens)


def content_window(max_tokens: int) -> int:
    return max(1, max_tokens - SPECIAL_TOKEN_RESERVE)


def clamp_overlap(overlap: int, window: int) -> int:
    if window <= 1:
        return 0
    if overlap >= window:
        return max(0, window // 4)
    return max(0, overlap)


def chunk_id_ranges(n_tokens: int, window: int, overlap: int) -> list[tuple[int, int]]:
    if n_tokens <= 0:
        return [(0, 0)]
    if n_tokens <= window:
        return [(0, n_tokens)]
    step = max(1, window - overlap)
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < n_tokens:
        end = min(n_tokens, start + window)
        ranges.append((start, end))
        if end >= n_tokens:
            break
        start += step
        if start >= n_tokens:
            break
    return ranges


def chunk_text(tokenizer: TokenEncoder, text: str, max_tokens: int, overlap: int) -> list[str]:
    window = content_window(max_tokens)
    overlap = clamp_overlap(overlap, window)
    previous = tokenizer.truncation
    tokenizer.no_truncation()
    try:
        ids = list(tokenizer.encode(text, add_special_tokens=False).ids)
    finally:
        restore_truncation(tokenizer, previous, max_tokens)
    ranges = chunk_id_ranges(len(ids), window, overlap)
    if len(ranges) == 1:
        return [text]
    chunks = []
    for start, end in ranges:
        piece = tokenizer.decode(ids[start:end]).strip()
        chunks.append(piece or text)
    return chunks
