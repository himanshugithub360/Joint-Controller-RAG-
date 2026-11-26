"""Exact inner-product search over a FAISS IndexFlatIP file without loading it into RAM.

Search-R1 ships its E5 index (21M x 768 float32, ~60 GB) as a FAISS flat index split into
part files. A flat index is a short header followed by the raw vectors, so this module
reads the vectors in chunks straight from the part files (no need to join them) and keeps
a running top-k. All queries are scored in one pass over the file, so a batch of
thousands of queries costs one disk read of the index. Results equal faiss
IndexFlatIP.search up to floating-point ties.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch

FLAT_FOURCC = {b"IxFI": "ip", b"IxF2": "l2"}


class MultiFileReader:
    """Read byte ranges from several files as if they were one concatenated file."""

    def __init__(self, paths: list[Path]):
        self.paths = [Path(p) for p in paths]
        self.sizes = [p.stat().st_size for p in self.paths]
        self.starts = np.cumsum([0] + self.sizes[:-1]).tolist()
        self.total = sum(self.sizes)
        self.handles = [open(p, "rb") for p in self.paths]

    def read(self, offset: int, n: int) -> bytes:
        if offset + n > self.total:
            raise ValueError("read past the end of the index files")
        buf = bytearray(n)
        view, done = memoryview(buf), 0
        for fh, start, size in zip(self.handles, self.starts, self.sizes):
            lo, hi = max(offset + done, start), min(offset + n, start + size)
            if lo >= hi:
                continue
            fh.seek(lo - start)
            got = fh.readinto(view[lo - offset : hi - offset])
            if got != hi - lo:
                raise OSError(f"short read from {fh.name}")
            done = hi - offset
        return bytes(buf)

    def close(self) -> None:
        for fh in self.handles:
            fh.close()


class FlatIndexFile:
    """Header of a FAISS flat index stored in one or more part files."""

    def __init__(self, parts: list[Path]):
        self.reader = MultiFileReader(parts)
        head = self.reader.read(0, 64)
        metric = FLAT_FOURCC.get(head[:4])
        if metric != "ip":
            raise ValueError(f"expected an IndexFlatIP file, found fourcc {head[:4]!r}")
        d, ntotal = struct.unpack_from("<iq", head, 4)
        # header: fourcc(4) d(4) ntotal(8) dummy(8) dummy(8) is_trained(1) metric_type(4)
        pos = 4 + 4 + 8 + 8 + 8 + 1
        (metric_type,) = struct.unpack_from("<i", head, pos)
        pos += 4
        if metric_type > 1:
            pos += 4  # metric_arg (float)
        (nbytes,) = struct.unpack_from("<q", head, pos)
        pos += 8
        if nbytes != ntotal * d * 4:
            # older FAISS writes the vector length in floats, not bytes
            if nbytes == ntotal * d:
                nbytes *= 4
            else:
                raise ValueError(f"unexpected codes size {nbytes} for {ntotal} x {d}")
        if pos + nbytes != self.reader.total:
            raise ValueError("index file size does not match its header")
        self.d, self.ntotal, self.data_offset = d, ntotal, pos

    def rows(self, start: int, stop: int) -> np.ndarray:
        raw = self.reader.read(self.data_offset + start * self.d * 4, (stop - start) * self.d * 4)
        return np.frombuffer(raw, dtype=np.float32).reshape(stop - start, self.d)

    def search(self, queries: np.ndarray, k: int, chunk_rows: int = 262_144,
               query_block: int = 2048, progress=None) -> tuple[np.ndarray, np.ndarray]:
        """Exact top-k by inner product. Returns (scores, ids), each (n_queries, k), best first."""
        q = torch.from_numpy(np.ascontiguousarray(queries, dtype=np.float32))
        nq = q.shape[0]
        best_s = torch.full((nq, k), -float("inf"))
        best_i = torch.full((nq, k), -1, dtype=torch.int64)
        for start in range(0, self.ntotal, chunk_rows):
            stop = min(start + chunk_rows, self.ntotal)
            block = torch.from_numpy(self.rows(start, stop).copy())
            kk = min(k, stop - start)
            for qs in range(0, nq, query_block):
                qe = min(qs + query_block, nq)
                s, i = torch.topk(q[qs:qe] @ block.T, kk, dim=1)
                cat_s = torch.cat([best_s[qs:qe], s], dim=1)
                cat_i = torch.cat([best_i[qs:qe], i + start], dim=1)
                top_s, pos = torch.topk(cat_s, k, dim=1)
                best_s[qs:qe], best_i[qs:qe] = top_s, torch.gather(cat_i, 1, pos)
            if progress:
                progress(stop, self.ntotal)
        return best_s.numpy(), best_i.numpy()

    def close(self) -> None:
        self.reader.close()
