"""Persistent retrieval memo: (retriever, exact query string) -> top-k doc ids and scores.

Every query the project ever sends is retrieved once and stored here, so later stages
(and the simulator) never search again, and repeated runs give identical passages.
SQLite keeps appends cheap and safe across crashes; export() writes the Parquet copy.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """CREATE TABLE IF NOT EXISTS hits (
    retriever TEXT NOT NULL, query TEXT NOT NULL, k INTEGER NOT NULL,
    doc_ids TEXT NOT NULL, scores TEXT NOT NULL,
    PRIMARY KEY (retriever, query))"""


class RetrievalMemo:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute(SCHEMA)

    def get_many(self, retriever: str, queries: list[str], k: int) -> dict[str, tuple[list[int], list[float]]]:
        """Stored hits for the queries that have at least k results (sliced to k)."""
        found = {}
        uniq = list(dict.fromkeys(queries))
        for i in range(0, len(uniq), 500):
            part = uniq[i : i + 500]
            marks = ",".join("?" * len(part))
            rows = self.db.execute(
                f"SELECT query, k, doc_ids, scores FROM hits WHERE retriever=? AND query IN ({marks})",
                [retriever, *part])
            for q, kk, ids, sc in rows:
                if kk >= k:
                    found[q] = (json.loads(ids)[:k], json.loads(sc)[:k])
        return found

    def put_many(self, retriever: str, k: int, items: dict[str, tuple[list[int], list[float]]]) -> None:
        with self.db:
            self.db.executemany(
                "INSERT OR REPLACE INTO hits VALUES (?, ?, ?, ?, ?)",
                [(retriever, q, k, json.dumps([int(x) for x in ids]), json.dumps([round(float(s), 6) for s in sc]))
                 for q, (ids, sc) in items.items()])

    def __len__(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM hits").fetchone()[0]

    def export(self, path: Path) -> int:
        df = pd.read_sql_query("SELECT * FROM hits", self.db)
        df["doc_ids"] = df["doc_ids"].map(json.loads)
        df["scores"] = df["scores"].map(json.loads)
        df.to_parquet(path, index=False)
        return len(df)

    def close(self) -> None:
        self.db.close()
