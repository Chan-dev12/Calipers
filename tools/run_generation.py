#!/usr/bin/env python3
"""
Generate answers over a completed retrieval run, then judge them.

    python tools/run_generation.py E4_hybrid
    python tools/run_generation.py E4_hybrid --top-n 5 --limit 10

Reads results/<name>.json (so it scores the EXACT ranking that experiment
produced -- no re-retrieval, no drift), generates an answer per question, scores
faithfulness and correctness, and writes results/<name>.generation.json.

WHY THIS IS A SEPARATE TOOL

Retrieval scoring is seconds and needs no LLM; generation plus judging is two
LLM calls per question on CPU. Welding them together would make the cheap,
frequently-run half hostage to the expensive, rarely-run half -- and the whole
reason the retrieval ablation is usable is that it runs in seconds.

Keeping them separate also keeps the attribution clean: this tool cannot change
retrieval, so any movement in the generation numbers is generation's doing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import judge, llm                                        # noqa: E402
from src.evalset import load_questions                            # noqa: E402
from src.generation import answer_one, is_refusal                 # noqa: E402
from src.indexing import chunk_corpus, load_corpus                # noqa: E402
from src.store import VectorStore                                 # noqa: E402
from src.chunkers import Chunk                                    # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
CORPUS = os.path.join(ROOT, "corpus")
EVALSET = os.path.join(ROOT, "evalset", "questions.jsonl")
RESULTS = os.path.join(ROOT, "results")


class _TextOnlyStore(VectorStore):
    """Chunk lookup without the embedding matrix.

    Generation needs chunk TEXT for ids the retrieval run already chose. It does
    not need vectors, so building the full store would mean embedding the whole
    corpus again to answer a question that is really just a dictionary lookup.
    """

    def __init__(self, chunks):
        super().__init__()
        self.chunks = list(chunks)
        self._id_to_pos = {c.chunk_id: i for i, c in enumerate(self.chunks)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="name of a completed run in results/")
    ap.add_argument("--top-n", type=int, default=5,
                    help="passages to put in the prompt (default 5)")
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N questions (for a smoke test)")
    ap.add_argument("--model", default=None, help="override the generator")
    args = ap.parse_args()

    path = os.path.join(RESULTS, f"{args.config}.json")
    if not os.path.exists(path):
        sys.exit(f"No retrieval run at {path}.\n"
                 f"Run it first:  python tools/run_eval.py {args.config}")
    with open(path, encoding="utf-8") as f:
        run = json.load(f)

    if not llm.health():
        sys.exit("Ollama unreachable. Start it, then retry.")
    try:
        llm.require_models(args.model or llm.GEN_MODEL, llm.JUDGE_MODEL)
    except RuntimeError as e:
        sys.exit(str(e))

    # Rebuild the SAME chunking this run used, so the chunk ids in the stored
    # ranking resolve to the same text they did at retrieval time.
    docs = load_corpus(CORPUS)
    chunks = chunk_corpus(docs, run["config"])
    store = _TextOnlyStore(chunks)

    questions = {q["qid"]: q for q in load_questions(EVALSET)}
    per_q = run["per_question"]
    if args.limit:
        per_q = per_q[:args.limit]

    gen_model = args.model or llm.GEN_MODEL
    print(f"generating with {gen_model}, judging with {llm.JUDGE_MODEL}")
    print(f"{len(per_q)} questions, top-{args.top_n} passages each\n")
    llm.warmup(gen_model)

    t0 = time.time()
    records = []
    for i, r in enumerate(per_q, 1):
        q = questions.get(r["qid"])
        if q is None:
            # The eval set was pruned after this run was scored. Skip rather
            # than guess -- the answer key is the authority.
            print(f"  [{i}/{len(per_q)}] {r['qid']} SKIPPED (not in current evalset)")
            continue

        out = answer_one(q["question"], r["retrieved"], store,
                         top_n=args.top_n, model=gen_model)
        refused = is_refusal(out["answer"])

        # A refusal is not scored for faithfulness: "NOT_IN_CONTEXT" makes no
        # claims, so there is nothing for the evidence to support. Scoring it
        # would reward refusing everything.
        faith = ({"score": None, "reason": "refusal -- no claims to check"}
                 if refused else
                 judge.faithfulness(out["answer"], out["context"]))
        correct = judge.correctness(q["question"], out["answer"], q["answer"])

        # Did the retriever actually supply the evidence? Lets a wrong answer be
        # attributed to retrieval or to generation instead of just "wrong".
        gold_in_context = bool(set(r["retrieved"][:args.top_n]) & set(r["gold"]))

        records.append({
            "qid": r["qid"], "question": q["question"],
            "reference": q["answer"], "answer": out["answer"],
            "refused": refused, "gold_in_context": gold_in_context,
            "empty_context": out["empty_context"],
            # Stored so a human can check faithfulness against the SAME text
            # the generator and judge saw. Without it, judge_agreement.py asked
            # people to rate "stays within the evidence" while never showing
            # them the evidence.
            "context": out["context"],
            "faithfulness": faith, "correctness": correct,
        })
        print(f"  [{i}/{len(per_q)}] {r['qid']}  "
              f"faith={faith['score']} correct={correct['score']}"
              f"{' REFUSED' if refused else ''}", flush=True)

    summary = summarise(records)
    elapsed = time.time() - t0
    out_path = os.path.join(RESULTS, f"{args.config}.generation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"name": args.config, "top_n": args.top_n,
                   "models": {"gen": gen_model, "judge": llm.JUDGE_MODEL},
                   "elapsed_sec": round(elapsed, 1),
                   "summary": summary, "per_question": records},
                  f, indent=2)

    print(f"\n{json.dumps(summary, indent=2)}")
    print(f"\nwritten: {out_path}")
    print(f"Validate the judge before trusting these:  "
          f"python tools/judge_agreement.py {args.config}")


def summarise(records) -> dict:
    """Aggregate, keeping unscored items out of the averages.

    Judge scores of None (a parse failure, or a refusal with nothing to check)
    are EXCLUDED rather than counted as zero. Counting them as zero would let a
    flaky judge look like a bad generator.
    """
    def mean(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    n = len(records)
    faith = [r["faithfulness"]["score"] for r in records]
    corr = [r["correctness"]["score"] for r in records]
    grounded = [r for r in records if r["gold_in_context"]]

    return {
        "n": n,
        "faithfulness_mean": mean(faith),
        "correctness_mean": mean(corr),
        "faithful_rate": round(sum(1 for s in faith if s == 2)
                               / max(1, len([s for s in faith if s is not None])), 4),
        "correct_rate": round(sum(1 for s in corr if s == 2)
                              / max(1, len([s for s in corr if s is not None])), 4),
        "refusal_rate": round(sum(1 for r in records if r["refused"]) / max(1, n), 4),
        # The interesting split: when the gold chunk WAS in the prompt, a wrong
        # answer is squarely generation's fault. This is the number that tells
        # you whether to fix the retriever or the generator.
        "n_gold_in_context": len(grounded),
        "correctness_given_gold": mean([r["correctness"]["score"] for r in grounded]),
        "judge_parse_failures": sum(
            1 for r in records
            if r["correctness"]["score"] is None
            or (r["faithfulness"]["score"] is None and not r["refused"])),
    }


if __name__ == "__main__":
    main()
