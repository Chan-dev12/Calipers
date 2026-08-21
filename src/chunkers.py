"""
Chunking strategies.

Three variants, all returning List[Chunk]. This is Ablation #1:
how you cut documents often matters more than which retriever you use.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, asdict
from typing import List, Callable

import numpy as np


@dataclass
class Chunk:
    chunk_id: str          # stable ID -- your answer key references these
    doc_id: str            # source filename
    text: str
    start: int             # char offset in source doc (for provenance)
    end: int
    meta: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def _mk_id(doc_id: str, start: int, text: str) -> str:
    """Stable chunk ID. If chunking params change, IDs change -- that is intentional.
    Your answer key stores (doc_id, quote) as the durable ground truth; chunk IDs
    are resolved from that at eval time. See tools/make_evalset.py."""
    h = hashlib.md5(f"{doc_id}|{start}|{text[:80]}".encode()).hexdigest()[:10]
    return f"{doc_id}::{start}::{h}"


# ---------------------------------------------------------------- 1. fixed
def fixed_size(doc_id: str, text: str, size: int = 512, overlap: int = 64) -> List[Chunk]:
    """Naive baseline: slice every N characters. Will cut sentences in half.
    That is the point -- it is the control condition."""
    out, i = [], 0
    step = max(1, size - overlap)
    while i < len(text):
        piece = text[i:i + size]
        if piece.strip():
            out.append(Chunk(_mk_id(doc_id, i, piece), doc_id, piece, i, i + len(piece)))
        i += step
    return out


# ------------------------------------------------------------ 2. recursive
_SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", ". ", " "]


def recursive(doc_id: str, text: str, size: int = 512, overlap: int = 64) -> List[Chunk]:
    """Split on the most semantic separator that fits, falling back down the list.
    This is what LangChain's RecursiveCharacterTextSplitter does -- written out so
    you can explain it."""

    def _split(s: str, seps: List[str]) -> List[str]:
        if len(s) <= size or not seps:
            return [s]
        sep, rest = seps[0], seps[1:]
        parts = s.split(sep)
        merged, buf = [], ""
        for p in parts:
            cand = (buf + sep + p) if buf else p
            if len(cand) <= size:
                buf = cand
            else:
                if buf:
                    merged.append(buf)
                buf = p
        if buf:
            merged.append(buf)
        out = []
        for m in merged:
            out.extend(_split(m, rest) if len(m) > size else [m])
        return out

    pieces = [p for p in _split(text, _SEPARATORS) if p.strip()]

    # re-attach overlap and recover char offsets
    out, cursor = [], 0
    for idx, p in enumerate(pieces):
        start = text.find(p, cursor)
        if start == -1:
            start = cursor
        prefix = pieces[idx - 1][-overlap:] if (overlap and idx > 0) else ""
        body = (prefix + " " + p).strip() if prefix else p
        out.append(Chunk(_mk_id(doc_id, start, body), doc_id, body, start, start + len(p)))
        cursor = start + len(p)
    return out


# ------------------------------------------------------------- 3. semantic
_SENT_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def semantic(doc_id: str, text: str, embed_fn: Callable[[List[str]], np.ndarray],
             percentile: float = 90.0, max_chars: int = 1200) -> List[Chunk]:
    """Embed each sentence, measure cosine distance between neighbours, and cut
    where the distance spikes above the Nth percentile -- i.e. where the topic shifts.

    Costs one embedding call per sentence, so it is the slowest chunker. Worth
    reporting that cost in your ablation table alongside the accuracy gain.
    """
    sents = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
    if len(sents) < 3:
        return fixed_size(doc_id, text, max_chars, 0)

    vecs = embed_fn(sents)
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    dists = 1.0 - np.sum(vecs[:-1] * vecs[1:], axis=1)  # distance between neighbours
    threshold = float(np.percentile(dists, percentile))

    out, buf, buf_start = [], [], 0
    cursor = 0
    for i, s in enumerate(sents):
        pos = text.find(s, cursor)
        if pos == -1:
            pos = cursor
        if not buf:
            buf_start = pos
        buf.append(s)
        cursor = pos + len(s)
        too_long = sum(len(x) for x in buf) >= max_chars
        breakpoint_here = i < len(dists) and dists[i] > threshold
        if breakpoint_here or too_long:
            body = " ".join(buf)
            out.append(Chunk(_mk_id(doc_id, buf_start, body), doc_id, body, buf_start, cursor))
            buf = []
    if buf:
        body = " ".join(buf)
        out.append(Chunk(_mk_id(doc_id, buf_start, body), doc_id, body, buf_start, cursor))
    return out


CHUNKERS = {"fixed": fixed_size, "recursive": recursive, "semantic": semantic}
