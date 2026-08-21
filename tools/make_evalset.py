#!/usr/bin/env python3
"""
Answer-key builder.

THE PROBLEM THIS SOLVES
Writing 100 question/answer pairs by hand takes a full day and you will quit at
question 30. So: the local LLM DRAFTS candidates from real chunks, and you
VERIFY them. Verifying is roughly 10x faster than authoring.

Be honest about this in your writeup. "LLM-drafted, human-verified, with N
rejected" is a legitimate, publishable methodology. Silently shipping unverified
LLM-generated ground truth is not -- your eval set would inherit the model's
blind spots and you would be grading the system with its own answer key.

THE CRITICAL DESIGN DECISION
Ground truth is stored as (doc_id + verbatim evidence quote), NOT as a chunk_id.

Why: you are about to compare three chunking strategies. Chunk IDs change every
time you re-chunk, so a chunk-ID answer key would be invalid the moment you run
your second experiment. A quote is stable -- at eval time you resolve it to
whichever chunks currently contain it. This one decision is what makes the
chunking ablation possible at all.

USAGE
    python tools/make_evalset.py draft   --n 60      # generate candidates
    python tools/make_evalset.py review              # verify them, one by one
    python tools/make_evalset.py stats               # see where you are
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.chunkers import recursive          # noqa: E402
from src import llm                          # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
CORPUS_DIR = os.path.join(ROOT, "corpus")
CANDIDATES = os.path.join(ROOT, "evalset", "candidates.jsonl")
QUESTIONS = os.path.join(ROOT, "evalset", "questions.jsonl")

DRAFT_PROMPT = """You are helping build an evaluation set for a retrieval system.

Below is an excerpt from a document. Write ONE specific question that:
- can be answered using ONLY this excerpt
- a real user of these documents might plausibly ask
- is NOT answerable from general knowledge without the excerpt
- avoids phrases like "according to the text" or "in this passage"

Also give the shortest verbatim span from the excerpt that proves the answer.
The quote MUST be copied character-for-character from the excerpt.

EXCERPT:
---
{chunk}
---

Return ONLY this JSON:
{{"question": "...", "answer": "...", "quote": "...", "difficulty": "easy|medium|hard"}}"""


def load_corpus():
    docs = {}
    patterns = ("**/*.md", "**/*.txt", "**/*.py", "**/*.rst", "**/*.java", "**/*.sql")
    for pat in patterns:
        for path in glob.glob(os.path.join(CORPUS_DIR, pat), recursive=True):
            rel = os.path.relpath(path, CORPUS_DIR)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read()
                if len(txt.strip()) > 200:
                    docs[rel] = txt
            except Exception as e:
                print(f"  skip {rel}: {e}")
    return docs


def cmd_draft(args):
    if not llm.health():
        sys.exit("Ollama not reachable. Start it with: ollama serve")

    docs = load_corpus()
    if not docs:
        sys.exit(f"No documents found in {CORPUS_DIR}/")
    print(f"Loaded {len(docs)} documents")

    all_chunks = []
    for doc_id, text in docs.items():
        all_chunks.extend(recursive(doc_id, text, size=900, overlap=0))
    print(f"Split into {len(all_chunks)} chunks")

    # Only chunks with real substance, and spread across documents so you do not
    # end up with 40 questions from your one longest file.
    usable = [c for c in all_chunks if len(c.text) > 300]
    random.Random(42).shuffle(usable)

    by_doc, picked = {}, []
    cap = max(2, args.n // max(1, len(docs)) + 2)
    for c in usable:
        if by_doc.get(c.doc_id, 0) < cap:
            picked.append(c)
            by_doc[c.doc_id] = by_doc.get(c.doc_id, 0) + 1
        if len(picked) >= args.n:
            break

    os.makedirs(os.path.dirname(CANDIDATES), exist_ok=True)
    written, rejected = 0, 0
    with open(CANDIDATES, "w", encoding="utf-8") as out:
        for i, c in enumerate(picked, 1):
            print(f"  [{i}/{len(picked)}] {c.doc_id}", flush=True)
            r = llm.generate_json(DRAFT_PROMPT.format(chunk=c.text[:2500]))
            if not r.get("question") or not r.get("quote"):
                rejected += 1
                continue
            # auto-reject hallucinated quotes before a human ever sees them
            if r["quote"].strip()[:60] not in c.text:
                rejected += 1
                continue
            out.write(json.dumps({
                "qid": f"q{written:03d}",
                "question": r["question"].strip(),
                "answer": r.get("answer", "").strip(),
                "doc_id": c.doc_id,
                "quote": r["quote"].strip(),
                "difficulty": r.get("difficulty", "medium"),
                "verified": None,
            }, ensure_ascii=False) + "\n")
            written += 1

    print(f"\n{written} candidates -> {CANDIDATES}")
    print(f"{rejected} auto-rejected (bad JSON or quote not found in source)")
    print("Next:  python tools/make_evalset.py review")


def cmd_review(args):
    if not os.path.exists(CANDIDATES):
        sys.exit("No candidates. Run: python tools/make_evalset.py draft")

    cands = [json.loads(l) for l in open(CANDIDATES, encoding="utf-8")]
    approved = []
    if os.path.exists(QUESTIONS):
        approved = [json.loads(l) for l in open(QUESTIONS, encoding="utf-8")]
    done = {q["qid"] for q in approved}

    todo = [c for c in cands if c["qid"] not in done]
    print(f"{len(approved)} approved, {len(todo)} left.  [k]eep / [s]kip / [e]dit / [q]uit\n")

    for c in todo:
        print("=" * 70)
        print(f"{c['qid']}  ({c['doc_id']}, {c['difficulty']})")
        print(f"Q: {c['question']}")
        print(f"A: {c['answer']}")
        print(f"EVIDENCE: {c['quote'][:300]}")
        choice = input("> ").strip().lower()
        if choice == "q":
            break
        if choice == "s":
            continue
        if choice == "e":
            nq = input("  new question (blank = keep): ").strip()
            if nq:
                c["question"] = nq
            na = input("  new answer   (blank = keep): ").strip()
            if na:
                c["answer"] = na
        c["verified"] = True
        approved.append(c)
        with open(QUESTIONS, "w", encoding="utf-8") as f:
            for q in approved:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"\n{len(approved)} verified questions in {QUESTIONS}")


def resolve_gold(question: dict, chunks: list, min_overlap: int = 40) -> set:
    """Map an evidence quote to chunk_ids under WHATEVER chunking is active.

    Called by the eval runner, not by you directly. This is the function that
    keeps your answer key valid across chunking experiments.
    """
    quote = " ".join(question["quote"].split())
    probe = quote[:min_overlap]
    gold = set()
    for c in chunks:
        if c.doc_id != question["doc_id"]:
            continue
        norm = " ".join(c.text.split())
        if probe and probe in norm:
            gold.add(c.chunk_id)
    if not gold:  # quote straddles a chunk boundary -- fall back to word overlap
        qwords = set(quote.lower().split())
        best, best_score = None, 0.0
        for c in chunks:
            if c.doc_id != question["doc_id"]:
                continue
            cwords = set(c.text.lower().split())
            score = len(qwords & cwords) / max(1, len(qwords))
            if score > best_score:
                best, best_score = c.chunk_id, score
        if best and best_score > 0.6:
            gold.add(best)
    return gold


def cmd_stats(args):
    for path, label in ((CANDIDATES, "candidates"), (QUESTIONS, "verified")):
        if os.path.exists(path):
            rows = [json.loads(l) for l in open(path, encoding="utf-8")]
            print(f"{label:12} {len(rows)}")
            if rows:
                diffs = {}
                docs = set()
                for r in rows:
                    diffs[r.get("difficulty", "?")] = diffs.get(r.get("difficulty", "?"), 0) + 1
                    docs.add(r["doc_id"])
                print(f"             by difficulty: {diffs}")
                print(f"             across {len(docs)} documents")
        else:
            print(f"{label:12} (none yet)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draft"); d.add_argument("--n", type=int, default=60)
    sub.add_parser("review")
    sub.add_parser("stats")
    a = p.parse_args()
    {"draft": cmd_draft, "review": cmd_review, "stats": cmd_stats}[a.cmd](a)
