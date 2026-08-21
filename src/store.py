"""
Vector store, hand-rolled.

No Chroma, no FAISS, no Pinecone. At 2k-20k chunks a brute-force numpy dot
product over a normalised matrix takes single-digit milliseconds. An ANN index
would be SLOWER here once you count build time, and it would introduce
approximation error into an experiment whose entire purpose is measurement.

Know this argument. "Why no vector DB?" is a near-certain interview question
and "brute force is exact and fast enough at my scale" is the right answer.

    cosine(a, b) = (a . b) / (|a| * |b|)

Normalise every vector once at insert time, and cosine similarity collapses to
a plain dot product -- so the whole search is one matrix-vector multiply.
"""
from __future__ import annotations

import json
import os
from typing import List, Tuple

import numpy as np

from .chunkers import Chunk


class VectorStore:
    def __init__(self):
        self.chunks: List[Chunk] = []
        self.matrix: np.ndarray | None = None   # (n_chunks, dim), L2-normalised
        self._id_to_pos = {}

    def add(self, chunks: List[Chunk], vectors: np.ndarray):
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks vs {len(vectors)} vectors")

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normed = (vectors / np.maximum(norms, 1e-9)).astype(np.float32)

        start = len(self.chunks)
        self.chunks.extend(chunks)
        for i, c in enumerate(chunks):
            self._id_to_pos[c.chunk_id] = start + i

        self.matrix = normed if self.matrix is None else np.vstack([self.matrix, normed])

    def search(self, query_vec: np.ndarray, k: int = 10) -> List[Tuple[Chunk, float]]:
        """One matrix-vector multiply gives every similarity at once."""
        if self.matrix is None or not len(self.chunks):
            return []
        q = query_vec / max(float(np.linalg.norm(query_vec)), 1e-9)
        sims = self.matrix @ q                       # (n_chunks,)

        k = min(k, len(sims))
        # argpartition finds the top k in O(n) without sorting all n
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]            # then sort only those k
        return [(self.chunks[i], float(sims[i])) for i in top]

    def get(self, chunk_id: str) -> Chunk | None:
        pos = self._id_to_pos.get(chunk_id)
        return self.chunks[pos] if pos is not None else None

    def texts(self) -> List[str]:
        return [c.text for c in self.chunks]

    def ids(self) -> List[str]:
        return [c.chunk_id for c in self.chunks]

    def __len__(self):
        return len(self.chunks)

    # ---- persistence: .npy for the matrix, .jsonl for the chunks ----
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "matrix.npy"), self.matrix)
        with open(os.path.join(path, "chunks.jsonl"), "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        s = cls()
        s.matrix = np.load(os.path.join(path, "matrix.npy"))
        with open(os.path.join(path, "chunks.jsonl"), encoding="utf-8") as f:
            for i, line in enumerate(f):
                c = Chunk(**json.loads(line))
                s.chunks.append(c)
                s._id_to_pos[c.chunk_id] = i
        return s
