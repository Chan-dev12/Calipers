#!/usr/bin/env python3
"""
Point the downstream experiments at the chunker that actually won.

    python tools/pick_chunker.py            # report and update E3-E6
    python tools/pick_chunker.py --dry-run  # report only

WHY THIS IS A TOOL AND NOT A NOTE IN THE README

The experiment matrix says E3-E6 use "the best chunker from E0-E2". That is a
dependency between experiments, and it was silently wrong for the entire life of
the project: E0 (fixed) beat E1 (recursive) by 7 NDCG points, and every one of
E3, E4, E5 and E6 was configured with `"chunker": "recursive"` -- the loser. Four
rows of the ablation were built on a foundation the ablation itself had already
rejected, and nothing complained, because a JSON file cannot check a claim made
in a markdown table.

So the rule is executable now. It reads the chunker experiments' results, picks
the winner on the headline metric, and rewrites the downstream configs to match,
recording in each file which run the choice came from.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(ROOT, "results")
CONFIGS = os.path.join(ROOT, "configs")

# The chunker ablation: same retriever, same everything, chunking varies.
CHUNKER_RUNS = ("E0_baseline", "E1_recursive", "E1b_recursive_java",
                "E1c_recursive_java_matched", "E2_semantic")
# The rows that inherit the winner.
DOWNSTREAM = ("E3_bm25", "E4_hybrid", "E5_rerank", "E6_query_rewrite")

METRIC = "ndcg@10"

# Keys that define a chunking. Copied wholesale onto the downstream configs so
# the inherited setup is identical, not merely similar.
CHUNK_KEYS = ("chunker", "chunk_size", "overlap", "separators", "percentile")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    runs = {}
    for name in CHUNKER_RUNS:
        p = os.path.join(RESULTS, f"{name}.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                runs[name] = json.load(f)

    if not runs:
        sys.exit("No chunker results yet. Run E0/E1/E1b/E2 first.")

    missing = [n for n in CHUNKER_RUNS if n not in runs]
    if missing:
        # Not fatal, but the choice is only as good as the field it ran against.
        print(f"WARNING: choosing without {', '.join(missing)} -- "
              f"the winner may change once they land.")

    # All rows must have been scored on the same answer key, or the comparison
    # is meaningless and so is anything inherited from it.
    fps = {r.get("evalset_fingerprint", "unstamped") for r in runs.values()}
    if len(fps) > 1:
        sys.exit(f"Chunker runs were scored against {len(fps)} different eval "
                 f"sets. Re-run them before picking a winner:\n"
                 f"  python tools/run_eval.py {' '.join(runs)}")

    print(f"chunker ablation, by {METRIC}:")
    ranked = sorted(runs.values(), key=lambda r: -r["metrics"].get(METRIC, 0.0))
    for r in ranked:
        stats = r.get("chunk_stats") or {}
        print(f"  {r['name']:20} {METRIC}={r['metrics'].get(METRIC, 0):.3f}  "
              f"chunks={r.get('n_chunks', '?'):>5}  "
              f"mean_chars={stats.get('mean_chunk_chars', '?')}")

    best = ranked[0]
    lo, hi = best.get(f"{METRIC}_ci95", [0, 0])
    print(f"\nwinner: {best['name']}  ({METRIC}={best['metrics'][METRIC]:.3f}, "
          f"95% CI [{lo:.3f}, {hi:.3f}])")

    # Honesty check. If the runner-up's interval overlaps the winner's, the
    # choice is not statistically supported -- say so rather than implying the
    # downstream rows rest on a demonstrated result.
    if len(ranked) > 1:
        r2 = ranked[1]
        lo2, hi2 = r2.get(f"{METRIC}_ci95", [0, 0])
        if lo2 < hi and lo < hi2:
            print(f"  NOTE: {r2['name']}'s interval overlaps it. The winner is "
                  f"not statistically distinguishable from the runner-up at "
                  f"n={best.get('n_questions', '?')}; this picks the point "
                  f"estimate so the pipeline is at least consistent, but do not "
                  f"report it as 'chunker X is better'.")

    chunking = {k: best["config"][k] for k in CHUNK_KEYS if k in best["config"]}
    print(f"\ninherited chunking: {json.dumps(chunking)}")

    changed = []
    for name in DOWNSTREAM:
        p = os.path.join(CONFIGS, f"{name}.json")
        if not os.path.exists(p):
            print(f"  missing config: {name}")
            continue
        with open(p, encoding="utf-8") as f:
            cfg = json.load(f)

        before = {k: cfg.get(k) for k in CHUNK_KEYS}
        for k in CHUNK_KEYS:
            cfg.pop(k, None)
        cfg.update(chunking)
        cfg["_chunker_from"] = (
            f"{best['name']} -- winner of the chunker ablation on {METRIC}. "
            f"Set by tools/pick_chunker.py; do not edit by hand.")

        after = {k: cfg.get(k) for k in CHUNK_KEYS}
        if before == after:
            print(f"  {name}: already correct")
            continue
        changed.append(name)
        print(f"  {name}: {before} -> {after}")
        if not args.dry_run:
            # Rewrite with the name first so the file stays readable.
            ordered = {"name": cfg.pop("name")}
            ordered.update(cfg)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(ordered, f, indent=2)
                f.write("\n")

    if args.dry_run:
        print("\n(dry run -- nothing written)")
    elif changed:
        print(f"\nupdated {len(changed)} config(s). Their existing results are "
              f"now stale -- re-run them:\n  python tools/run_eval.py "
              f"{' '.join(changed)}")
    else:
        print("\nall downstream configs already use the winning chunker.")


if __name__ == "__main__":
    main()
