"""retrieve(queries, k) -> top-k (doc_id, score) per query, memoised.

Queries already in the memo cost nothing. All misses in one call are encoded together and
scored in a single pass over the index, so callers should send large batches (a whole
split, or a whole tree level in Stage 6) rather than one query at a time.
"""

from __future__ import annotations

from pathlib import Path

from .e5 import E5QueryEncoder
from .flat_index import FlatIndexFile
from .memo import RetrievalMemo


class E5Retriever:
    name = "e5"

    def __init__(self, index_parts: list[Path], memo_path: Path, model_name: str = "intfloat/e5-base-v2",
                 store_k: int = 10):
        self.index_parts = index_parts
        self.model_name = model_name
        self.memo = RetrievalMemo(memo_path)
        self.store_k = store_k
        self._encoder: E5QueryEncoder | None = None
        self._index: FlatIndexFile | None = None

    @property
    def encoder(self) -> E5QueryEncoder:
        if self._encoder is None:
            self._encoder = E5QueryEncoder(self.model_name)
        return self._encoder

    @property
    def index(self) -> FlatIndexFile:
        if self._index is None:
            self._index = FlatIndexFile(self.index_parts)
        return self._index

    def retrieve(self, queries: list[str], k: int = 10, progress=None) -> list[list[tuple[int, float]]]:
        if k > self.store_k:
            raise ValueError(f"k={k} is above the stored depth {self.store_k}")
        hits = self.memo.get_many(self.name, queries, self.store_k)
        missing = [q for q in dict.fromkeys(queries) if q not in hits]
        if missing:
            emb = self.encoder.encode(missing)
            scores, ids = self.index.search(emb, self.store_k, progress=progress)
            new = {q: (ids[j].tolist(), scores[j].tolist()) for j, q in enumerate(missing)}
            self.memo.put_many(self.name, self.store_k, new)
            hits.update(new)
        return [list(zip(hits[q][0][:k], hits[q][1][:k])) for q in queries]

    def close(self) -> None:
        self.memo.close()
        if self._index is not None:
            self._index.close()
