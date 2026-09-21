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

  NON-UNIQUE EVIDENCE (critical)
    The evidence quote is not unique inside its own source file. This is the
    defect that silently destroys a retrieval score, so it gets its own check.

    resolve_gold() maps a quote onto every chunk that contains it. If the quote
    is boilerplate -- `SystemProperties.java` repeats "Returns {@code null} if
    the property cannot be read..." 193 times -- then EVERY one of those chunks
    becomes gold. With 115 gold chunks, recall@5 is capped at 5/115 = 0.043 and
    the retriever is punished no matter what it returns.

    It is not a question-wording problem, so the specificity check above cannot
    see it: the question can name an exact method and still cite evidence that
    appears verbatim in eighty other places. The defect lives in the QUOTE, and
    the only way to see it is to go back to the source document and count.

    Four such questions dragged BM25's measured NDCG@10 from +0.069 to -0.010
    against the baseline -- i.e. they inverted the conclusion of the ablation.

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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.indexing import load_corpus                            # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
QUESTIONS = os.path.join(ROOT, "evalset", "questions.jsonl")
CORPUS = os.path.join(ROOT, "corpus")

# How many times a quote may appear in its own source file before the question
# is unusable. Calibrated against this corpus: occurrences map almost 1:1 onto
# resolved gold chunks (1 -> 1, 2 -> 2-3, 7 -> 8, 193 -> 115), so anything
# above 3 means the answer key points at a crowd rather than a passage.
MAX_QUOTE_OCCURRENCES = 3

# resolve_gold() matches on the first 40 characters of the normalised quote.
# Use the same probe here, or the audit would measure something the scorer does
# not actually do.
PROBE_CHARS = 40

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


def quote_occurrences(question, docs) -> int:
    """How many times this question's evidence probe appears in its source doc.

    Mirrors resolve_gold(): same normalisation, same 40-char probe. Counting in
    the DOCUMENT rather than in chunks keeps the check chunking-independent --
    a question is either answerable or it is not, and that should not depend on
    which experiment happens to be running.
    """
    doc = docs.get(question["doc_id"])
    if doc is None:
        return -1                      # unknown doc; reported separately
    probe = norm(question["quote"])[:PROBE_CHARS]
    return norm(doc).count(probe) if probe else 0


def audit(questions, docs=None):
    quote_counts = Counter(norm(q["quote"])[:80] for q in questions)
    docs = docs if docs is not None else {}
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

        # non-unique evidence: the quote is boilerplate within its own file
        if docs:
            occ = quote_occurrences(q, docs)
            if occ == -1:
                issues.append(("CRITICAL", f"source document {q['doc_id']!r} "
                                           f"not found in corpus/"))
            elif occ == 0:
                issues.append(("CRITICAL", "evidence quote does not appear in "
                                           "its source document"))
            elif occ > MAX_QUOTE_OCCURRENCES:
                issues.append(("CRITICAL", f"evidence appears {occ}x in "
                                           f"{q['doc_id']} -- every copy becomes "
                                           f"gold, so recall is capped near "
                                           f"{5.0 / occ:.3f} at k=5"))
            elif occ > 1:
                issues.append(("WARN", f"evidence appears {occ}x in "
                                       f"{q['doc_id']}"))

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

    docs = load_corpus(CORPUS)
    if not docs:
        print(f"WARNING: no documents in {CORPUS}/ -- skipping the "
              f"evidence-uniqueness check, which is the one that matters most.")
    findings = audit(questions, docs)
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
        # Never overwrite an existing backup. The first version wrote a fixed
        # `.prepruned.jsonl`, so a SECOND prune silently destroyed the record of
        # the first -- losing the original drafted set, and with it the rejection
        # rate this project reports as a result. Find a free slot instead.
        backup = QUESTIONS.replace(".jsonl", ".prepruned.jsonl")
        n = 2
        while os.path.exists(backup):
            backup = QUESTIONS.replace(".jsonl", f".prepruned.{n}.jsonl")
            n += 1
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
