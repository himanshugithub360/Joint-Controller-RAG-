"""Random access to Search-R1's wiki-18 corpus (21M 100-word passages).

A passage's id is its 0-based line number in wiki-18.jsonl, the same id the E5 index and
Search-R1's load_docs use. prepare() decompresses the .gz once and writes a byte-offset
table, so any passage is one seek away and the 14 GB file is never loaded into RAM.
"""

from __future__ import annotations

import gzip
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np


def _open_payload(gz_path: Path):
    """The JSONL byte stream inside gz_path.

    Search-R1's wiki-18.jsonl.gz is a gzipped *tar* holding one member (wiki_dump.jsonl),
    not a plain gzipped JSONL. Plain `gzip -d` leaves a 512-byte tar header glued to
    passage 0 and tar padding as an extra last line; reading the member avoids both.
    A plain .jsonl.gz is also accepted.
    """
    with gzip.open(gz_path, "rb") as probe:
        head = probe.read(512)
    if head[257:262] == b"ustar":
        tar = tarfile.open(gz_path, mode="r|gz")
        member = next(m for m in tar if m.isfile())
        return tar.extractfile(member), tar
    return gzip.open(gz_path, "rb"), None


def prepare(gz_path: Path, jsonl_path: Path, offsets_path: Path, progress_every: int = 1_000_000,
            check_every: int = 10_000) -> int:
    """Decompress gz_path to jsonl_path and save line start offsets (int64) to offsets_path.

    Checks, on every `check_every`-th line and the last one, that the record's "id" equals
    its line number, since that line number is the passage id the E5 index returns.
    """
    if jsonl_path.exists() and offsets_path.exists():
        return len(np.load(offsets_path, mmap_mode="r"))
    tmp = jsonl_path.with_suffix(".tmp")
    offsets = []
    pos = 0
    last = b""
    src, tar = _open_payload(gz_path)
    with open(tmp, "wb") as dst:
        for line in src:
            if not line.strip():
                continue
            i = len(offsets)
            if check_every and i % check_every == 0:
                _check_id(line, i)
            offsets.append(pos)
            dst.write(line)
            pos += len(line)
            last = line
            if progress_every and len(offsets) % progress_every == 0:
                print(f"  {len(offsets):,} passages", flush=True)
    src.close()
    if tar is not None:
        tar.close()
    if offsets:
        _check_id(last, len(offsets) - 1)
    np.save(offsets_path, np.asarray(offsets, dtype=np.int64))
    shutil.move(tmp, jsonl_path)
    return len(offsets)


def _check_id(line: bytes, i: int) -> None:
    rec_id = json.loads(line)["id"]
    if str(rec_id) != str(i):
        raise ValueError(f"line {i} holds passage id {rec_id!r}; ids must equal line numbers")


def split_contents(contents: str) -> tuple[str, str]:
    """Search-R1 stores '"Title"\\ntext'; split it the way its _passages2string does."""
    title, _, text = contents.partition("\n")
    return title, text


class Corpus:
    def __init__(self, jsonl_path: Path, offsets_path: Path):
        self.offsets = np.load(offsets_path, mmap_mode="r")
        self.fh = open(jsonl_path, "rb")

    def __len__(self) -> int:
        return len(self.offsets)

    def get(self, doc_ids) -> list[dict]:
        """[{doc_id, title, text}] in the order given."""
        out = []
        for i in doc_ids:
            self.fh.seek(int(self.offsets[int(i)]))
            rec = json.loads(self.fh.readline())
            title, text = split_contents(rec["contents"])
            out.append({"doc_id": int(i), "title": title, "text": text})
        return out

    def close(self) -> None:
        self.fh.close()
