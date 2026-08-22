#!/usr/bin/env python3
"""
Audit the eval set for defects that would quietly corrupt your results.

    python tools/audit_evalset.py           # report
    python tools/audit_evalset.py --prune   # drop everything flagged CRITICAL

An eval set is an instrument. An uncalibrated instrument produces confident
numbers that are wrong, which is worse than no numbers -- so check it before
you measure anything with it.

WHAT IT LOOKS FOR

  LEAKAGE (critical)
    The answer is copied verbatim from the evidence quote, and the question
    reuses the same wording. Retrieval collapses into string matching, BM25
    scores near-perfectly, and hybrid retrieval appears to add nothing. Your
    ablation table would show flat lines and you would conclude, wrongly, that
    none of the techniques help.

  AMBIGUITY (critical)
    The question does not name a specific method or class. In overloaded
    utility code -- contains(int[]), contains(Object[]), contains(long[]) --
    the same Javadoc sentence appears in a dozen chunks. Only one is recorded
    as gold, so a retriever is punished for returning an equally correct chunk.

  MARKUP (warning)
    Raw Javadoc syntax ({@link}, {@code}, <p>) left in the question. Nobody
    types that. The retriever ends up matching on formatting artifacts.

  DUPLICATE EVIDENCE (warning)
    Two questions pointing at the same quote. Not fatal, but it weights that
    passage twice in your averages.

  TRIVIAL (warning)
    Very short or generic questions.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.join(os.path.dirname(__file__), "..")
QUESTIONS = os.path.join(ROOT, "evalset", "questions.jsonl")

MARKUP = re.compile(r"\{@\w+|</?\w+>|@return|@param|@throws")

# What makes a question specific enough to have ONE right answer chunk.
# The first version of this only matched CamelCase and method(), which threw
# false positives on constants (IS_OS_MAC_OSX_CHEETAH), dotted references
# (Boolean.TRUE) and backticked names (`fill`). All of those are specific.
SPECIFIC = re.compile(
    r"\b[A-Z][a-z]+[A-Z]\w*"          # StringUtils, ArrayFill
    r"|\b[A-Z][A-Z0-9]+(?:_[A-Z0-9]+)+\b"   # IS_OS_MAC_OSX_CHEETAH, JAVA_RECENT
    r"|\b[A-Z][A-Za-z0-9]*\.[A-Za-z_]\w*"   # Boolean.TRUE, SystemUtils.IS_OS
    r"|\w+\(\)"                        # secure(), toString()
    r"|`\w+`"                          # `fill`
    r"|\b[a-z]+[A-Z]\w*\b"             # substringBefore, indexOf
)

# Method names that are heavily overloaded in utility code: naming one of these
# without a parameter type is still ambiguous, because the same Javadoc is
# duplicated across every primitive variant.
OVERLOADED = re.compile(
    r"\b(contains|indexOf|lastIndexOf|toString|isEmpty|isNotEmpty|add|remove|"
    r"toArray|clone|equals|hashCode|toObject|toPrimitive|fill|reverse|"
    r"defaultIfNull|getLength|subarray)\b", re.I)


def norm(s: str) -> str:
    return " ".join(s.lower().split())


def overlap(a: str, b: str) -> float:
    aw, bw = set(norm(a).split()), set(norm(b).split())
    return len(aw & bw) / max(1, len(aw))


def audit(questions):
    quote_counts = Counter(norm(q["quote"])[:80] for q in questions)
    findings = []

    for q in questions:
        issues = []
        ans, quote, ques = q.get("answer", ""), q["quote"], q["question"]

        # Leakage: answer lifted verbatim from the quote.
        # Only meaningful for substantive answers -- a short factual answer
        # ("-1", "abc", "false") is necessarily a substring of its evidence and
        # that is fine. The problem is a full sentence copied across, which
        # means no reformulation happened.
        if ans and len(ans.split()) >= 7:
            if norm(ans) in norm(quote):
                issues.append(("CRITICAL", "answer sentence copied verbatim from evidence"))
            elif overlap(ans, quote) > 0.85:
                issues.append(("CRITICAL", "answer ~85% lexical overlap with evidence"))

        # leakage: question reuses the source wording
        if overlap(ques, quote) > 0.6:
            issues.append(("CRITICAL", "question reuses evidence wording"))

        # ambiguity: no specific identifier named at all
        if not SPECIFIC.search(ques):
            issues.append(("CRITICAL", "no method/class/constant named -- ambiguous target"))
        # ambiguity: names an overloaded method with no parameter type.
        # ArrayUtils.contains exists for Object[], int[], long[], double[]...
        # each carrying identical Javadoc, so several chunks are equally correct
        # while only one is recorded as gold.
        elif OVERLOADED.search(ques) and not re.search(r"\(.+\)|\bint\b|\blong\b|"
                                                       r"\bdouble\b|\bObject\b|\bchar\b|"
                                                       r"\bboolean\b|\bbyte\b|\bfloat\b", ques):
            issues.append(("CRITICAL", "overloaded method without parameter type -- "
                                       "several chunks would be equally correct"))

        # vague framing: refers to "the method"/"this file" instead of naming it
        if re.search(r"\bthe (method|function|class|file|object)\b|\bthis (file|class)\b",
                     ques, re.I):
            issues.append(("WARN", "refers to 'the method/this file' rather than naming it"))

        # markup artifacts
        if MARKUP.search(ques):
            issues.append(("WARN", "raw Javadoc markup in question"))

        # duplicate evidence
        if quote_counts[norm(quote)[:80]] > 1:
            issues.append(("WARN", "evidence shared with another question"))

        # trivial
        if len(ques.split()) < 7:
            issues.append(("WARN", "very short question"))

        if issues:
            findings.append((q, issues))

    return findings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prune", action="store_true",
                    help="write a cleaned file, keeping a backup")
    args = ap.parse_args()

    with open(QUESTIONS, encoding="utf-8") as f:
        questions = [json.loads(l) for l in f if l.strip()]

    findings = audit(questions)
    flagged = {q["qid"] for q, iss in findings}
    critical = {q["qid"] for q, iss in findings
                if any(lvl == "CRITICAL" for lvl, _ in iss)}

    for q, issues in findings:
        worst = "CRITICAL" if any(l == "CRITICAL" for l, _ in issues) else "WARN"
        print(f"\n[{worst}] {q['qid']}  ({q['doc_id']})")
        print(f"  Q: {q['question'][:100]}")
        for lvl, msg in issues:
            print(f"     - {lvl}: {msg}")

    print("\n" + "=" * 60)
    print(f"total questions   {len(questions)}")
    print(f"clean             {len(questions) - len(flagged)}")
    print(f"flagged           {len(flagged)}  ({len(critical)} critical)")

    if len(questions) - len(critical) < 15:
        print("\nFewer than 15 clean questions. Draft another batch:")
        print("  python tools/make_evalset.py draft --n 40")

    if args.prune and critical:
        backup = QUESTIONS.replace(".jsonl", ".prepruned.jsonl")
        os.replace(QUESTIONS, backup)
        kept = [q for q in questions if q["qid"] not in critical]
        with open(QUESTIONS, "w", encoding="utf-8") as f:
            for q in kept:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
        print(f"\npruned {len(critical)} critical -> {len(kept)} remain")
        print(f"backup: {backup}")
        print("Report both numbers in your writeup. The rejection rate is a "
              "result, not an embarrassment.")


if __name__ == "__main__":
    main()
