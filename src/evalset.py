"""
Eval set loading, and the function that makes the whole study possible.

Ground truth is stored as (doc_id + verbatim quote), never as a chunk_id.
Chunk IDs change every time you re-chunk, so a chunk-ID answer key would be
invalid the moment you ran your second experiment. resolve_gold() maps the
stored quote onto whatever chunking is currently active.

That one decision is what lets E0, E1 and E2 -- three different chunkers -- be
scored against the same answer key.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Set


def load_questions(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No eval set at {path}. Build one first:\n"
            f"  python tools/make_evalset.py draft --n 40\n"
            f"  python tools/make_evalset.py review")
    with open(path, encoding="utf-8") as f:
        qs = [json.loads(line) for line in f if line.strip()]
    if not qs:
        raise ValueError(f"{path} is empty.")
    return qs


def resolve_gold(question: Dict, chunks: List, min_overlap: int = 40) -> Set[str]:
    """Map an evidence quote to chunk_ids under the ACTIVE chunking.

    Two passes:
      1. exact substring match on the first `min_overlap` chars of the quote
      2. if that fails (the quote straddles a chunk boundary), fall back to
         word-overlap and take the best chunk above 0.6

    Returns an empty set when neither works. Do not silently drop those --
    report them. A question whose evidence cannot be located under a given
    chunker is itself a finding about that chunker.
    """
    quote = " ".join(question["quote"].split())
    probe = quote[:min_overlap]
    gold = set()

    for c in chunks:
        if c.doc_id != question["doc_id"]:
            continue
        if probe and probe in " ".join(c.text.split()):
            gold.add(c.chunk_id)

    if not gold:
        qwords = set(quote.lower().split())
        best, best_score = None, 0.0
        for c in chunks:
            if c.doc_id != question["doc_id"]:
                continue
            overlap = len(qwords & set(c.text.lower().split())) / max(1, len(qwords))
            if overlap > best_score:
                best, best_score = c.chunk_id, overlap
        if best and best_score > 0.6:
            gold.add(best)

    return gold


def resolve_all(questions: List[Dict], chunks: List) -> tuple:
    """Returns (resolved, unresolved_qids). Print the unresolved count every run
    -- if it climbs when you change chunkers, that is a real result."""
    resolved, unresolved = [], []
    for q in questions:
        gold = resolve_gold(q, chunks)
        if gold:
            resolved.append({**q, "gold": sorted(gold)})
        else:
            unresolved.append(q["qid"])
    return resolved, unresolved
