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


# Sidecar outputs that live in results/ but are NOT retrieval runs. The glob
# below is "E*.json", which happily matches E4_hybrid.generation.json -- loading
# one as an experiment row would crash on the missing "metrics" key, or worse,
# quietly contribute a malformed row to the table.
_SIDECARS = (".generation.json", ".judge_agreement.json")


def load_results():
    out = []
    for p in sorted(glob.glob(os.path.join(RESULTS, "E*.json"))):
        if p.endswith(_SIDECARS):
            continue
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        if "metrics" not in r:
            print(f"  skipping {os.path.basename(p)}: not a retrieval run")
            continue
        out.append(r)
    return sorted(out, key=lambda r: r["name"])


def partition_by_evalset(results):
    """Split results by the answer key they were scored against.

    A delta between two rows only means something if both were scored on the
    same questions. Results scored on an older key are not deleted -- they are
    separated, reported, and kept out of the comparison.
    """
    fps = {}
    for r in results:
        fps.setdefault(r.get("evalset_fingerprint", "unstamped"), []).append(r)
    if len(fps) == 1:
        return results, []
    # the key most results agree on wins; the rest are stale
    current = max(fps.values(), key=len)
    stale = [r for group in fps.values() if group is not current for r in group]
    return current, stale


def table(results) -> str:
    if not results:
        return "No results yet. Run: python tools/run_eval.py"

    baseline = next((r for r in results if r["name"].startswith("E0")), results[0])
    base_m = baseline["metrics"]

    head = ["Experiment"] + [label for _, label in COLUMNS] + [
        "NDCG@10 95% CI", "Chunks", "Mean chars", "Time"]
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
        # chunk count and MEAN CHUNK LENGTH are reported next to every score,
        # because they are the confound in a chunker ablation: chunk_size is
        # fixed in the config but the realised mean is not, and score tracks the
        # realised mean. Showing both lets a reader see that for themselves.
        stats = r.get("chunk_stats") or {}
        cells.append(str(r.get("n_chunks", "?")))
        mc = stats.get("mean_chunk_chars")
        cells.append(f"{mc:.0f}" if isinstance(mc, (int, float)) else "?")
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

    # A row scored on a different number of questions cannot be compared to the
    # baseline even if it shares the fingerprint check above.
    mismatched = [r["name"] for r in results if r["n_questions"] != n]
    if mismatched:
        lines.append("")
        lines.append(f"**WARNING** these rows scored a different number of "
                     f"questions than the baseline and are NOT comparable: "
                     f"{', '.join(mismatched)}")

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


README = os.path.join(ROOT, "README.md")
_README_START = "<!-- ABLATION:START -->"
_README_END = "<!-- ABLATION:END -->"


def update_readme(md: str) -> bool:
    """Replace the README's Results block with the freshly generated table.

    The README carried a "(populated by report.py)" placeholder for the whole
    life of the project, so the one artefact a reader actually opens was the one
    thing never updated. Copying a table across by hand is how a README ends up
    quoting numbers that no longer exist in results/ -- so it happens
    mechanically, between markers, or not at all.
    """
    if not os.path.exists(README):
        return False
    with open(README, encoding="utf-8") as f:
        text = f.read()
    if _README_START not in text or _README_END not in text:
        print(f"  README has no {_README_START} markers; leaving it alone.")
        return False
    head, rest = text.split(_README_START, 1)
    _, tail = rest.split(_README_END, 1)
    nl = chr(10)
    new_text = head + _README_START + nl + nl + md + nl + nl + _README_END + tail
    if new_text == text:
        return False
    with open(README, "w", encoding="utf-8") as f:
        f.write(new_text)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true",
                    help="write results/ABLATION.md")
    ap.add_argument("--readme", action="store_true",
                    help="also splice the table into README.md")
    args = ap.parse_args()

    results = load_results()
    results, stale = partition_by_evalset(results)
    md = table(results)
    if stale:
        names = ", ".join(f"{r['name']} (n={r.get('n_questions','?')})"
                          for r in stale)
        md += (
            "\n\n**Excluded as stale**\n"
            f"- {names} were scored against a different answer key and "
            "are not comparable to the rows above. Re-run them:\n"
            f"  `python tools/run_eval.py "
            f"{' '.join(r['name'] for r in stale)}`")
    print(md)

    if args.save and results:
        path = os.path.join(RESULTS, "ABLATION.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Ablation results\n\n" + md + "\n")
        print(f"\nwritten: {path}")

    if args.readme and results:
        print("  README.md updated" if update_readme(md)
              else "  README.md already current")


if __name__ == "__main__":
    main()
