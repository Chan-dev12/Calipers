#!/usr/bin/env python3
"""
Run an experiment and score it.

    python tools/run_eval.py                    # every config in configs/
    python tools/run_eval.py E0_baseline        # just one
    python tools/run_eval.py E0_baseline E4_hybrid

Results land in results/<name>.json. Then:  python tools/report.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import llm, retrievers                                   # noqa: E402
from src.evalset import load_questions, resolve_all               # noqa: E402
from src.indexing import build_store, chunk_corpus, corpus_stats, load_corpus  # noqa: E402
from src.metrics import evaluate_run, bootstrap_ci, ndcg_at_k     # noqa: E402

import hashlib                                                     # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
CORPUS = os.path.join(ROOT, "corpus")
EVALSET = os.path.join(ROOT, "evalset", "questions.jsonl")
RESULTS = os.path.join(ROOT, "results")

# one store per chunking signature, reused across retriever variants in a run
_store_cache = {}


def evalset_fingerprint(questions: list) -> str:
    """Short hash of the answer key a result was scored against.

    Results accumulate in results/ across weeks while the eval set keeps being
    pruned. Without a fingerprint, report.py will happily put a row scored on 36
    questions next to a row scored on 28 and draw a delta between them -- which
    is how a pruning fix quietly turns into a fabricated comparison. Stamping it
    here lets the report refuse to do that.
    """
    blob = "|".join(f"{q['qid']}:{q['doc_id']}:{' '.join(q['quote'].split())}"
                    for q in sorted(questions, key=lambda x: x["qid"]))
    return hashlib.md5(blob.encode("utf-8")).hexdigest()[:10]


def _chunk_key(cfg: dict) -> tuple:
    # separators MUST be part of the key. Without it E1 and E1b -- same
    # chunker, same size, different separator list -- would share one cached
    # store, and the "experiment" would compare a config against itself while
    # reporting a plausible-looking number.
    return (cfg.get("chunker", "recursive"), cfg.get("chunk_size", 512),
            cfg.get("overlap", 64), cfg.get("percentile", 90.0),
            cfg.get("separators", "prose"))


def get_store(cfg: dict, docs: dict):
    key = _chunk_key(cfg)
    if key not in _store_cache:
        print(f"  chunking [{key[0]}]...")
        chunks = chunk_corpus(docs, cfg)
        stats = corpus_stats(docs, chunks)
        print(f"  {stats}")
        _store_cache[key] = (build_store(chunks), chunks, stats)
    return _store_cache[key]


def run_one(cfg: dict, docs: dict, questions: list) -> dict:
    name = cfg["name"]
    print(f"\n=== {name} ===")
    t0 = time.time()

    store, chunks, chunk_stats = get_store(cfg, docs)
    resolved, unresolved = resolve_all(questions, chunks)
    if unresolved:
        # Do not hide this. If a chunker cannot host the evidence for some
        # questions, that is a property of the chunker worth reporting.
        print(f"  WARNING: {len(unresolved)}/{len(questions)} questions have no "
              f"locatable gold chunk under this chunking: {unresolved[:5]}")
    if not resolved:
        raise RuntimeError("No questions could be resolved. Is corpus/ the same "
                           "content the eval set was built from?")

    retriever = retrievers.build(cfg, store)
    k = cfg.get("top_k", 10)

    runs = []
    for i, q in enumerate(resolved, 1):
        if i % 10 == 0 or i == 1:
            print(f"  [{i}/{len(resolved)}] {q['qid']}", flush=True)
        ranked = retriever.retrieve(q["question"], k)
        runs.append({"qid": q["qid"],
                     "retrieved": [cid for cid, _ in ranked],
                     "gold": q["gold"]})

    metrics = evaluate_run(runs)
    lo, hi = bootstrap_ci(runs, lambda r, g: ndcg_at_k(r, g, 10))
    elapsed = time.time() - t0

    result = {
        "name": name,
        "config": cfg,
        "evalset_fingerprint": evalset_fingerprint(questions),
        "evalset_size": len(questions),
        "n_questions": len(resolved),
        "n_unresolved": len(unresolved),
        "unresolved_qids": unresolved,
        "n_chunks": len(chunks),
        # Reported because chunk_size is held constant in the CONFIG but not in
        # effect: fixed/recursive/java produce means of ~987/860/781 chars at a
        # nominal 1000. Score tracks mean size, so a chunker comparison that
        # hides this cannot separate "better boundaries" from "more context".
        "chunk_stats": chunk_stats,
        "metrics": {m: round(v, 4) for m, v in metrics.items()},
        "ndcg@10_ci95": [round(lo, 4), round(hi, 4)],
        "elapsed_sec": round(elapsed, 1),
        "models": {"embed": llm.EMBED_MODEL, "gen": llm.GEN_MODEL},
        "per_question": runs,
    }

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    m = metrics
    print(f"  ndcg@10={m['ndcg@10']:.3f} [{lo:.3f}, {hi:.3f}]  "
          f"recall@5={m['recall@5']:.3f}  mrr={m['mrr']:.3f}  ({elapsed:.0f}s)")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*", help="config names, or blank for all")
    args = ap.parse_args()

    if not llm.health():
        sys.exit("Ollama unreachable. Start it, then retry.")
    try:
        # Fail now, not after the corpus has been indexed.
        llm.require_models(llm.EMBED_MODEL, llm.GEN_MODEL)
    except RuntimeError as e:
        sys.exit(str(e))

    docs = load_corpus(CORPUS)
    if not docs:
        sys.exit(f"No documents in {CORPUS}/. Add 30-80 files first.")
    print(f"corpus: {len(docs)} documents")

    questions = load_questions(EVALSET)
    print(f"evalset: {len(questions)} questions")

    if args.configs:
        paths = [os.path.join(ROOT, "configs", f"{n}.json") for n in args.configs]
    else:
        paths = sorted(glob.glob(os.path.join(ROOT, "configs", "E*.json")))

    for p in paths:
        if not os.path.exists(p):
            print(f"missing config: {p}")
            continue
        with open(p, encoding="utf-8") as f:
            cfg = json.load(f)
        try:
            run_one(cfg, docs, questions)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")

    print("\nDone. Build the table:  python tools/report.py")


if __name__ == "__main__":
    main()
