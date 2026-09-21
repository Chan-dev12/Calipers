"""
Corpus loading + index building.

One index per chunking strategy. Because the embedding cache is keyed on text,
building E1's index after E0's is cheap for any chunk whose text repeats -- and
completely free on a re-run.
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List

from . import llm
from .chunkers import CHUNKERS, Chunk
from .store import VectorStore

EXTENSIONS = ("*.md", "*.txt", "*.py", "*.rst", "*.sql", "*.java", "*.js", "*.ts")


def load_corpus(corpus_dir: str, min_chars: int = 200) -> Dict[str, str]:
    docs = {}
    for ext in EXTENSIONS:
        for path in glob.glob(os.path.join(corpus_dir, "**", ext), recursive=True):
            rel = os.path.relpath(path, corpus_dir).replace("\\", "/")
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except OSError:
                continue
            if len(text.strip()) >= min_chars:
                docs[rel] = text
    return docs


def chunk_corpus(docs: Dict[str, str], config: Dict) -> List[Chunk]:
    kind = config.get("chunker", "recursive")
    fn = CHUNKERS[kind]
    chunks: List[Chunk] = []

    for doc_id, text in docs.items():
        if kind == "semantic":
            chunks.extend(fn(doc_id, text,
                             embed_fn=lambda xs: llm.embed(xs),
                             percentile=config.get("percentile", 90.0)))
        elif kind == "recursive":
            chunks.extend(fn(doc_id, text,
                             size=config.get("chunk_size", 512),
                             overlap=config.get("overlap", 64),
                             separators=config.get("separators", "prose")))
        else:
            chunks.extend(fn(doc_id, text,
                             size=config.get("chunk_size", 512),
                             overlap=config.get("overlap", 64)))
    return chunks


def build_store(chunks: List[Chunk], verbose: bool = True) -> VectorStore:
    store = VectorStore()
    if not chunks:
        return store
    if verbose:
        print(f"  embedding {len(chunks)} chunks (cached after first run)...")
    vectors = llm.embed([c.text for c in chunks], batch_log=verbose)
    store.add(chunks, vectors)
    return store


def corpus_stats(docs: Dict[str, str], chunks: List[Chunk]) -> Dict:
    lengths = [len(c.text) for c in chunks]
    return {
        "documents": len(docs),
        "chunks": len(chunks),
        "mean_chunk_chars": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "min_chunk_chars": min(lengths) if lengths else 0,
        "max_chunk_chars": max(lengths) if lengths else 0,
    }
