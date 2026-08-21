#!/usr/bin/env python3
"""
Build the ablation table. This file produces the actual deliverable.

    python tools/report.py              # print markdown
    python tools/report.py --save       # also write results/ABLATION.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(ROOT, "results")

COLUMNS = [("recall@5", "Recall@5"), ("mrr", "MRR"),
           ("ndcg@10", "NDCG@10"), ("precision@5", "P@5")]


def load_results():
    out = []
    for p in sorted(glob.glob(os.path.join(RESULTS, "E*.json"))):
        with open(p, encoding="utf-8") as f:
            out.append(json.load(f))
    return sorted(out, key=lambda r: r["name"])


def table(results) -> str:
    if not results:
        return "No results yet. Run: python tools/run_eval.py"

    baseline = next((r for r in results if r["name"].startswith("E0")), results[0])
    base_m = baseline["metrics"]

    head = ["Experiment"] + [label for _, label in COLUMNS] + ["NDCG@10 95% CI", "Time"]
    rows = []
    for r in results:
        m = r["metrics"]
        cells = [r["name"]]
        for key, _ in COLUMNS:
            val = m.get(key, 0.0)
            if r is baseline:
                cells.append(f"{val:.3f}")
            else:
                d = val - base_m.get(key, 0.0)
                cells.append(f"{val:.3f} ({d:+.3f})")
        lo, hi = r.get("ndcg@10_ci95", [0, 0])
        cells.append(f"[{lo:.3f}, {hi:.3f}]")
        cells.append(f"{r['elapsed_sec']:.0f}s")
        rows.append(cells)

    widths = [max(len(str(row[i])) for row in [head] + rows) for i in range(len(head))]
    def fmt(cells):
        return "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    lines = [fmt(head), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    lines += [fmt(r) for r in rows]

    n = baseline["n_questions"]
    lines.append("")
    lines.append(f"n = {n} questions. Deltas are vs {baseline['name']}. "
                 f"CI by bootstrap resampling, 1000 iterations.")

    # honesty check: flag deltas that the confidence interval does not support
    notes = []
    for r in results:
        if r is baseline:
            continue
        lo, hi = r.get("ndcg@10_ci95", [0, 0])
        b_lo, b_hi = baseline.get("ndcg@10_ci95", [0, 0])
        if lo < b_hi and b_lo < hi:
            notes.append(f"- {r['name']}: NDCG@10 interval overlaps the baseline's. "
                         f"With n={n} this difference is not statistically distinguishable.")
        if r.get("n_unresolved"):
            notes.append(f"- {r['name']}: {r['n_unresolved']} questions had no "
                         f"locatable gold chunk under this chunking.")
    if notes:
        lines.append("")
        lines.append("**Caveats**")
        lines.extend(notes)

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    results = load_results()
    md = table(results)
    print(md)

    if args.save and results:
        path = os.path.join(RESULTS, "ABLATION.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Ablation results\n\n" + md + "\n")
        print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()
