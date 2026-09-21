#!/usr/bin/env python3
"""
Audit the judge against human labels.

    python tools/judge_agreement.py E4_hybrid              # label 25 items
    python tools/judge_agreement.py E4_hybrid --n 15
    python tools/judge_agreement.py E4_hybrid --report     # scores only, no labelling

An LLM judge is an instrument, and this project's claim is that it measures
rather than demos. "Faithfulness 0.88" is worth nothing on its own; "0.88, from
a judge that agrees with a human 84% of the time, kappa 0.71" is a measurement.

HOW IT WORKS

Samples N items from a completed generation run, hides the judge's verdict, and
asks you to score each one yourself. Then it reports raw agreement, agreement
within one point, and Cohen's kappa.

WHY KAPPA AND NOT JUST AGREEMENT
If 90% of answers are faithful, a judge that says "faithful" unconditionally
scores 90% raw agreement while containing no information at all. Kappa corrects
for that chance agreement and would score the same degenerate judge at 0.0. Any
raw agreement number reported without it is flattering itself.

YOUR LABELS ARE THE RECORD
They are written to evalset/human_labels.json and reused on later runs, so the
judge can be re-validated after a prompt or model change without re-labelling
from scratch. Labelling is the expensive part; throwing it away each time is why
judge validation usually gets skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.judge import agreement_report                            # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(ROOT, "results")
LABELS = os.path.join(ROOT, "evalset", "human_labels.json")

# The human and the judge MUST score against the same rubric, or kappa
# measures rubric mismatch rather than judge quality. These are the judge's
# own wordings from src/judge.py, shown verbatim. An earlier version printed
# one vague scale merging both metrics, so the two raters were not actually
# answering the same question.
CORRECTNESS_SCALE = """  Does the ANSWER mean the same as the REFERENCE?
  Wording may differ -- judge meaning, not phrasing.
    2 = same meaning as the reference
    1 = partially correct, or missing an essential part
    0 = wrong, or contradicts the reference
    s = skip      q = save and quit"""

FAITHFULNESS_SCALE = """  Is every claim in the ANSWER stated in the PASSAGES?
  Judge only whether the passages support it, NOT whether it is true.
    2 = every claim is stated in the passages
    1 = mostly supported, but adds a detail the passages do not state
    0 = contradicts the passages, or invents information
    s = skip      q = save and quit"""


def load_labels() -> dict:
    if os.path.exists(LABELS):
        with open(LABELS, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_labels(d: dict):
    with open(LABELS, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)


def ask(prompt: str):
    """Read one label. Returns int, 'skip', or 'quit'."""
    while True:
        try:
            raw = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "quit"
        if raw in ("0", "1", "2"):
            return int(raw)
        if raw == "s":
            return "skip"
        if raw == "q":
            return "quit"
        print("    enter 0, 1, 2, s or q")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--n", type=int, default=25,
                    help="how many items to hand-label (default 25)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", action="store_true",
                    help="just score existing labels, do not prompt")
    args = ap.parse_args()

    path = os.path.join(RESULTS, f"{args.config}.generation.json")
    if not os.path.exists(path):
        sys.exit(f"No generation run at {path}.\n"
                 f"Run it first:  python tools/run_generation.py {args.config}")
    with open(path, encoding="utf-8") as f:
        gen = json.load(f)

    records = gen["per_question"]
    # Fixed seed: the sample must be the same set on every invocation, or you
    # would be labelling a different subset each time and the agreement number
    # would drift for reasons that have nothing to do with the judge.
    rng = random.Random(args.seed)
    sample = rng.sample(records, min(args.n, len(records)))

    labels = load_labels()
    store = labels.setdefault(args.config, {})

    if not args.report:
        todo = [r for r in sample if r["qid"] not in store]
        print(f"\n{len(sample)} sampled, {len(sample) - len(todo)} already labelled, "
              f"{len(todo)} to go.")
        print("The judge's own scores are hidden until the end.\n")

        for i, r in enumerate(todo, 1):
            print("=" * 72)
            print(f"[{i}/{len(todo)}]  {r['qid']}")
            print(f"\nQUESTION:  {r['question']}")
            print(f"REFERENCE: {r['reference']}")
            print(f"\nANSWER:    {r['answer']}")
            print(f"\n{CORRECTNESS_SCALE}")

            c = ask("\n  CORRECTNESS (vs reference) > ")
            if c == "quit":
                break
            f_ = None
            if not r["refused"]:
                if "context" not in r:
                    sys.exit("This generation run has no stored context, so "
                             "faithfulness cannot be labelled. Re-run: "
                             f"python tools/run_generation.py {args.config}")
                print("\nPASSAGES THE MODEL WAS GIVEN:")
                print("-" * 72)
                print(r["context"])
                print("-" * 72)
                print(f"\n{FAITHFULNESS_SCALE}")
                f_ = ask("  FAITHFULNESS (is every claim in the ANSWER stated "
                         "in the passages above?) > ")
                if f_ == "quit":
                    break
            entry = {}
            if c != "skip":
                entry["correctness"] = c
            if f_ not in (None, "skip"):
                entry["faithfulness"] = f_
            if entry:
                store[r["qid"]] = entry
                save_labels(labels)

        save_labels(labels)
        print(f"\nlabels saved: {LABELS}")

    # ------------------------------------------------------------- scoring
    pairs = {"correctness": ([], []), "faithfulness": ([], [])}
    for r in sample:
        h = store.get(r["qid"])
        if not h:
            continue
        for metric in ("correctness", "faithfulness"):
            if metric in h and r[metric]["score"] is not None:
                pairs[metric][0].append(h[metric])
                pairs[metric][1].append(r[metric]["score"])

    print("\n" + "=" * 72)
    print(f"JUDGE AGREEMENT -- {args.config}")
    print(f"judge model: {gen['models']['judge']}")
    print("=" * 72)

    out = {}
    for metric, (human, machine) in pairs.items():
        rep = agreement_report(human, machine)
        out[metric] = rep
        if not rep.get("n"):
            print(f"\n{metric}: no labelled items yet.")
            continue
        print(f"\n{metric}  (n={rep['n']})")
        print(f"  exact agreement   {rep['exact_agreement']:.3f}")
        print(f"  within one point  {rep['within_one']:.3f}")
        k = rep["cohens_kappa"]
        print(f"  Cohen's kappa     "
              f"{f'{k:.3f}  ({interpret(k)})' if k is not None else 'undefined'}")
        print(f"  human mean {rep['human_mean']:.2f}  vs  "
              f"judge mean {rep['judge_mean']:.2f}")
        if k is None:
            print("    kappa is undefined when both raters used a single label "
                  "-- the sample is too uniform to be informative.")

    path_out = os.path.join(RESULTS, f"{args.config}.judge_agreement.json")
    with open(path_out, "w", encoding="utf-8") as f:
        json.dump({"config": args.config,
                   "judge_model": gen["models"]["judge"],
                   "agreement": out}, f, indent=2)
    print(f"\nwritten: {path_out}")
    print("\nReport this next to the faithfulness score. A judge whose kappa is "
          "low has not validated your generation numbers -- it has replaced one "
          "unknown with another.")


def interpret(k: float) -> str:
    """Landis & Koch (1977) bands. Conventional, not laws of nature."""
    if k < 0.0:
        return "worse than chance"
    if k < 0.21:
        return "slight"
    if k < 0.41:
        return "fair"
    if k < 0.61:
        return "moderate"
    if k < 0.81:
        return "substantial"
    return "almost perfect"


if __name__ == "__main__":
    main()
