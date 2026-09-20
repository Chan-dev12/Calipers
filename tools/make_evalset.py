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
# resolve_gold lives in ONE place. This file used to carry its own near-copy,
# already drifted from the original, with a docstring claiming the eval runner
# called it -- it did not; the runner imports src.evalset. Two implementations
# of the function that defines ground truth is exactly the kind of quiet
# divergence that produces confident wrong numbers.
from src.evalset import resolve_gold          # noqa: E402,F401

ROOT = os.path.join(os.path.dirname(__file__), "..")
CORPUS_DIR = os.path.join(ROOT, "corpus")
CANDIDATES = os.path.join(ROOT, "evalset", "candidates.jsonl")
QUESTIONS = os.path.join(ROOT, "evalset", "questions.jsonl")

DRAFT_PROMPT = """You are building an evaluation set for a code retrieval system.

Below is an excerpt from the file: {doc_id}

Write ONE question a developer might ask about this code.

HARD RULES -- breaking any of these makes the question useless for evaluation:

1. NAME THE EXACT TARGET. Write "StringUtils.substringBefore" or the constant
   "IS_OS_MAC_OSX_CHEETAH". NEVER write "the method", "this class", "the
   function" or "this file" -- those match dozens of places in the codebase.

2. IF THE METHOD IS OVERLOADED, give the parameter type. "ArrayUtils.contains"
   exists for Object[], int[], long[] and more, all with identical Javadoc.
   Write "ArrayUtils.contains(int[], int)" instead.

3. DO NOT REUSE THE EXCERPT'S WORDING. Ask it the way a developer would type it
   into a search box. If the excerpt says "returns false if a null array is
   passed in", do NOT ask "what is returned if a null array is passed in".

4. NO JAVADOC MARKUP anywhere. No {{@link}}, no {{@code}}, no <p>, no @return.
   Plain prose only.

5. IT MUST REQUIRE THIS CODE. If general Java knowledge answers it ("what does
   isEmpty do", "what does super() do"), it tests memorisation, not retrieval.
   Prefer specific return values, edge cases, exception conditions, version
   notes, constant values.

6. THE QUOTE MUST EXPLAIN, NOT MERELY EXIST. A line like
   `setDefaultFullDetail(true)` or `throw new UnsupportedOperationException();`
   appears in dozens of files and proves nothing. Quote the Javadoc sentence
   that STATES the fact -- not a line of code that happens to do something.

7. NEVER ask about a name that exists everywhere -- remove(), super(),
   toString(), equals(), iterator(), hashCode(). Ask about something unique to
   this file.

The "quote" field must be copied character-for-character from the excerpt and
must actually prove the answer.

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

    # Only sample chunks that actually contain documented facts.
    #
    # Without this the drafter samples raw code bodies -- constructor calls,
    # `throw new UnsupportedOperationException();` -- and manufactures questions
    # from method names, citing a line that exists but proves nothing. Those
    # quotes pass the hallucination check (the text IS in the chunk) while being
    # useless as evidence, so only human review catches them. Cheaper to never
    # generate them.
    #
    # Requiring a Javadoc block plus a minimum amount of prose filters the
    # undocumented code out. Report this: it means the eval set covers the
    # DOCUMENTED surface of the corpus, not all of it.
    def has_documented_content(text: str) -> bool:
        if "/**" not in text:
            return False
        # prose inside the Javadoc, not just @param/@return tags or code
        doc_lines = [ln.strip().lstrip("*").strip()
                     for ln in text.splitlines() if ln.strip().startswith("*")]
        prose = [ln for ln in doc_lines
                 if len(ln.split()) >= 6 and not ln.startswith("@")]
        return len(prose) >= 3

    usable = [c for c in all_chunks
              if len(c.text) > 300 and has_documented_content(c.text)]
    print(f"{len(usable)} chunks carry enough documentation to draft from "
          f"({len(all_chunks) - len(usable)} skipped as undocumented)")

    # Exclude chunks that already produced an approved question.
    #
    # An earlier version seeded the shuffle with a constant (random.Random(42)),
    # which meant every batch sampled the SAME chunks in the SAME order -- so a
    # second draft re-generated the first batch's questions, including ones the
    # human had already rejected. Deterministic sampling is right for an
    # experiment and wrong for incremental data collection.
    used_quotes = set()
    if os.path.exists(QUESTIONS):
        with open(QUESTIONS, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    q = json.loads(line)
                    used_quotes.add(" ".join(q["quote"].split())[:60])

    fresh = [c for c in usable
             if not any(u in " ".join(c.text.split()) for u in used_quotes)]
    print(f"{len(fresh)} unused ({len(usable) - len(fresh)} already covered by "
          f"approved questions)")

    random.shuffle(fresh)   # unseeded: each batch draws different chunks
    usable = fresh

    by_doc, picked = {}, []
    cap = max(2, args.n // max(1, len(docs)) + 2)
    for c in usable:
        if by_doc.get(c.doc_id, 0) < cap:
            picked.append(c)
            by_doc[c.doc_id] = by_doc.get(c.doc_id, 0) + 1
        if len(picked) >= args.n:
            break

    os.makedirs(os.path.dirname(CANDIDATES), exist_ok=True)

    # Batch-stamped IDs. An earlier version numbered every batch q000, q001...
    # so a second batch collided with the first, and `review` -- which skips
    # any qid it has already approved -- silently dropped the new questions.
    # Nothing errored; the batch just disappeared.
    import datetime
    batch = datetime.datetime.now().strftime("b%m%d%H%M")

    written, rejected, non_unique = 0, 0, 0
    with open(CANDIDATES, "w", encoding="utf-8") as out:
        for i, c in enumerate(picked, 1):
            print(f"  [{i}/{len(picked)}] {c.doc_id}", flush=True)
            r = llm.generate_json(DRAFT_PROMPT.format(chunk=c.text[:2500], doc_id=c.doc_id))
            if not r.get("question") or not r.get("quote"):
                rejected += 1
                continue
            # auto-reject hallucinated quotes before a human ever sees them
            if r["quote"].strip()[:60] not in c.text:
                rejected += 1
                continue

            # auto-reject evidence that is not unique inside its own file.
            #
            # This is where the project's worst defect was born. resolve_gold()
            # maps a quote onto EVERY chunk containing it, so a boilerplate
            # Javadoc sentence -- one repeats 193 times in SystemProperties.java
            # -- produces 115 gold chunks and caps recall@5 at 0.043. Four such
            # questions were enough to invert BM25's measured result against the
            # baseline. The quote is verbatim and the question can name an exact
            # method, so neither the hallucination check above nor any check on
            # the question's wording can see it. Only counting in the source can.
            probe = " ".join(r["quote"].split())[:40]
            occurrences = " ".join(docs[c.doc_id].split()).count(probe)
            if occurrences > 3:
                non_unique += 1
                rejected += 1
                continue
            out.write(json.dumps({
                "qid": f"{batch}_{written:03d}",
                "question": r["question"].strip(),
                "answer": r.get("answer", "").strip(),
                "doc_id": c.doc_id,
                "quote": r["quote"].strip(),
                "difficulty": r.get("difficulty", "medium"),
                "verified": None,
            }, ensure_ascii=False) + "\n")
            written += 1

    print(f"\n{written} candidates -> {CANDIDATES}")
    print(f"{rejected} auto-rejected (bad JSON, or quote absent from source)")
    print(f"  of which {non_unique} cited evidence repeating >3x in its own "
          f"file -- the defect that silently caps recall")
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
