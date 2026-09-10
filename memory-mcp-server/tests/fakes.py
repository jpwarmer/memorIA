from __future__ import annotations

import math


class _Encoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class FakeTokenizer:
    def __init__(self) -> None:
        self._truncation: dict | None = None
        self.word_to_id: dict[str, int] = {}
        self.id_to_word: dict[int, str] = {}

    @property
    def truncation(self) -> dict | None:
        return self._truncation

    def enable_truncation(self, max_length: int, **kwargs) -> None:
        self._truncation = {"max_length": max_length, **kwargs}

    def no_truncation(self) -> None:
        self._truncation = None

    def encode(self, text: str, add_special_tokens: bool = False) -> _Encoding:
        words = text.split() if text.strip() else []
        ids = []
        for word in words:
            if word not in self.word_to_id:
                index = len(self.word_to_id)
                self.word_to_id[word] = index
                self.id_to_word[index] = word
            ids.append(self.word_to_id[word])
        if self._truncation and add_special_tokens:
            ids = ids[: self._truncation["max_length"]]
        return _Encoding(ids)

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.id_to_word[item] for item in ids)


class FakeEmbedder:
    dim = 32

    def embed(self, texts):
        for text in texts:
            yield _Vec(self._vector(text))

    def _vector(self, text: str) -> list[float]:
        values = [0.0] * self.dim
        words = text.lower().split() or ["_"]
        for word in words:
            values[hash(word) % self.dim] += 1.0
        norm = math.sqrt(sum(item * item for item in values)) or 1.0
        return [item / norm for item in values]


class _Vec(list):
    def tolist(self) -> list[float]:
        return list(self)
