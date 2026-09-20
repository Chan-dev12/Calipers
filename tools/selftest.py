#!/usr/bin/env python3
"""
Self-test for the measuring apparatus. No Ollama, no corpus, runs in a second.

    python tools/selftest.py

WHY THIS FILE EXISTS

This project's claim is that it measures rather than demos, which makes every
scoring function a load-bearing part of the result. A bug in ndcg_at_k does not
crash anything -- it produces a plausible number that is wrong, and a wrong
number in an ablation table is indistinguishable from a finding.

The project already learned this twice the expensive way. The audit tool's
specificity check wrongly rejected valid questions until it was itself audited.
The answer key silently mapped one boilerplate Javadoc sentence onto 115 gold
chunks, which capped recall at 0.043 and inverted the measured ranking of BM25
against the baseline. Both were defects in the instrument, not the system, and
both survived because nothing checked the checker.

So: the metrics get hand-computed expected values, and the known-bad cases get
regression tests.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import judge                                             # noqa: E402
from src.bm25 import BM25, reciprocal_rank_fusion, tokenize       # noqa: E402
from src.chunkers import fixed_size, recursive                    # noqa: E402
from src.evalset import resolve_gold                              # noqa: E402
from src.metrics import (bootstrap_ci, hit_rate_at_k, ndcg_at_k,  # noqa: E402
                         precision_at_k, recall_at_k, reciprocal_rank)

PASS, FAIL = [], []


def check(name, got, want, tol=1e-9):
    ok = (abs(got - want) < tol) if isinstance(want, float) else (got == want)
    (PASS if ok else FAIL).append(f"{name}: got {got!r}, want {want!r}")
    print(("  ok   " if ok else "  FAIL ") + name +
          ("" if ok else f"   got {got!r}, want {want!r}"))


def section(t):
    print(f"\n{t}\n" + "-" * len(t))


# ------------------------------------------------------------------- metrics
section("metrics: hand-computed values")
R = ["a", "b", "c", "d", "e"]

check("hit@1 when gold is at rank 1", hit_rate_at_k(R, {"a"}, 1), 1.0)
check("hit@1 when gold is at rank 2", hit_rate_at_k(R, {"b"}, 1), 0.0)
check("hit@3 when gold is at rank 3", hit_rate_at_k(R, {"c"}, 3), 1.0)

check("recall@5, 1 of 2 gold found", recall_at_k(R, {"a", "z"}, 5), 0.5)
check("recall with empty gold is 0", recall_at_k(R, set(), 5), 0.0)

check("precision@5, 1 hit", precision_at_k(R, {"a"}, 5), 0.2)
check("precision@2, 2 hits", precision_at_k(R, {"a", "b"}, 2), 1.0)

check("RR, gold at rank 1", reciprocal_rank(R, {"a"}), 1.0)
check("RR, gold at rank 4", reciprocal_rank(R, {"d"}), 0.25)
check("RR, gold absent", reciprocal_rank(R, {"zzz"}), 0.0)

# NDCG by hand. Single gold at rank 2: DCG = 1/log2(3) = 0.63093,
# IDCG (one gold, perfectly placed) = 1/log2(2) = 1. So NDCG = 0.63093.
check("ndcg@5, single gold at rank 2", ndcg_at_k(R, {"b"}, 5), 0.6309297535714574, 1e-9)
check("ndcg@5, gold at rank 1 is perfect", ndcg_at_k(R, {"a"}, 5), 1.0)
# Two gold at ranks 1 and 2 is also a perfect ranking.
check("ndcg@5, gold at ranks 1+2 is perfect", ndcg_at_k(R, {"a", "b"}, 5), 1.0)
check("ndcg@5, no gold retrieved", ndcg_at_k(R, {"zzz"}, 5), 0.0)
# IDCG must cap at k: 3 gold but k=1 means the best possible is 1 hit.
check("ndcg@1 with 3 gold caps IDCG at k", ndcg_at_k(R, {"a", "b", "c"}, 1), 1.0)

lo, hi = bootstrap_ci([{"retrieved": R, "gold": {"a"}}] * 20,
                      lambda r, g: ndcg_at_k(r, g, 10))
check("bootstrap CI on a constant metric is degenerate", (lo, hi), (1.0, 1.0))

# ---------------------------------------------------------------------- bm25
section("bm25")
corpus = ["the cat sat on the mat", "dogs are loyal animals",
          "the cat chased the dog", "snake_case identifiers survive"]
bm = BM25(corpus)
top = bm.search("cat", 2)
check("bm25 finds both docs containing 'cat'", len(top), 2)
check("bm25 ranks a 'cat' doc first", top[0][0] in (0, 2), True)
check("bm25 ignores docs with no query term", len(bm.search("zebra", 5)), 0)
check("tokenizer keeps snake_case whole", "snake_case" in tokenize("snake_case x"), True)

# RRF uses rank only, never raw score -- that is the whole point of it.
#
# A property worth pinning, because the intuitive guess is wrong: a document
# ranked 2nd by BOTH retrievers does NOT beat one ranked 1st by one and 3rd by
# the other. 1/(k+1) + 1/(k+3) > 2/(k+2) for every k > 0, because 1/x is convex,
# so RRF always prefers the consensus-splitting candidate to the compromise one.
# This test was originally written asserting the opposite and the code was
# right. Fusion rewards a confident first place more than broad agreement.
fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "b", "a"]], k=60)
check("RRF: 1st+3rd beats 2nd+2nd (convexity of 1/x)", fused[0][0] in ("a", "c"), True)
check("RRF: the 2nd-by-both candidate comes last", fused[-1][0], "b")
# And the headline property: identical rankings preserve their order.
same = reciprocal_rank_fusion([["a", "b", "c"], ["a", "b", "c"]], k=60)
check("RRF: agreeing retrievers preserve order", [d for d, _ in same], ["a", "b", "c"])
weighted = reciprocal_rank_fusion([["x", "y"], ["y", "x"]], k=60, weights=[2.0, 1.0])
check("RRF respects weights", weighted[0][0], "x")

# ------------------------------------------------------------------ chunkers
section("chunkers")
text = "A" * 250
cs = fixed_size("d.txt", text, size=100, overlap=20)
check("fixed_size covers the whole document", cs[-1].end >= len(text), True)
check("fixed_size overlaps by the requested amount", cs[1].start, 80)
check("fixed_size ids are unique", len({c.chunk_id for c in cs}), len(cs))

prose = "\n\n".join(f"Paragraph {i}. " + "word " * 40 for i in range(6))
rc = recursive("d.md", prose, size=300, overlap=50)
check("recursive produces chunks", len(rc) > 1, True)
check("recursive respects the size bound (plus overlap)",
      max(len(c.text) for c in rc) <= 300 + 50 + 1, True)
check("recursive ids are unique", len({c.chunk_id for c in rc}), len(rc))

java = "class X {\n    public void a() {\n        int i = 1;\n    }\n\n    /**\n     * doc\n     */\n    public void b() {\n        int j = 2;\n    }\n}\n"
jr = recursive("X.java", java, size=60, overlap=0, separators="java")
check("java separators are actually selected", len(jr) > 1, True)
# An unknown separator name must fall back, not crash.
check("unknown separator set falls back to prose",
      len(recursive("X.java", java, size=60, overlap=0, separators="nope")) > 0, True)

# ------------------------------------------------------- gold resolution
section("evalset: gold resolution")


class _C:
    def __init__(self, cid, doc, text):
        self.chunk_id, self.doc_id, self.text = cid, doc, text


quote = "Returns null if the property cannot be read due to a SecurityException."
chunks = [_C("c1", "A.java", "prelude " + quote + " tail"),
          _C("c2", "A.java", "unrelated content entirely"),
          _C("c3", "B.java", quote)]                      # different doc
g = resolve_gold({"doc_id": "A.java", "quote": quote}, chunks)
check("gold resolves to the chunk in the right document", g, {"c1"})
check("gold never crosses document boundaries", "c3" in g, False)

# THE REGRESSION TEST FOR THE 115-GOLD-CHUNK DEFECT.
# A boilerplate quote resolves to EVERY chunk repeating it. resolve_gold is
# behaving correctly here -- this is the behaviour the audit must catch upstream,
# and this test pins the fact that it happens so nobody "fixes" the symptom.
many = [_C(f"c{i}", "A.java", quote) for i in range(20)]
check("boilerplate evidence really does explode the gold set",
      len(resolve_gold({"doc_id": "A.java", "quote": quote}, many)), 20)

# ----------------------------------------------------------------- the judge
section("judge validation")
check("kappa: identical raters", judge.cohens_kappa([2, 1, 0], [2, 1, 0]), 1.0)
check("kappa: perfectly inverted raters", judge.cohens_kappa([2, 2, 0, 0], [0, 0, 2, 2]), -1.0)
# The property that justifies reporting kappa at all.
check("kappa: a judge that always says '2' scores 0 despite high raw agreement",
      judge.cohens_kappa([2, 2, 2, 2, 1, 0], [2, 2, 2, 2, 2, 2]), 0.0)
check("kappa: undefined when only one label is used", judge.cohens_kappa([2, 2], [2, 2]), None)
check("kappa: mismatched lengths are refused", judge.cohens_kappa([2], [2, 1]), None)

rep = judge.agreement_report([2, 1, 2, 0], [2, 2, 2, 0])
check("agreement: exact", rep["exact_agreement"], 0.75)
check("agreement: within one point", rep["within_one"], 1.0)
check("agreement: unlabelled items are dropped",
      judge.agreement_report([2, None, 1], [2, 1, None])["n"], 1)

# ---------------------------------------------------- the audit's own check
section("audit: evidence uniqueness")
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from audit_evalset import quote_occurrences, audit, MAX_QUOTE_OCCURRENCES  # noqa: E402

boiler = "Returns null if the property cannot be read due to a SecurityException."
docs = {"A.java": "intro\n" + (boiler + "\n") * 9 + "outro",
        "B.java": "a unique sentence appears exactly once here."}

check("counts every repetition of the evidence",
      quote_occurrences({"doc_id": "A.java", "quote": boiler}, docs), 9)
check("a unique quote counts once",
      quote_occurrences({"doc_id": "B.java",
                         "quote": "a unique sentence appears exactly once here."}, docs), 1)
check("a missing document is reported, not silently passed",
      quote_occurrences({"doc_id": "GONE.java", "quote": "x"}, docs), -1)

# The end-to-end regression: this is the exact defect that inverted the BM25
# result, and the audit must flag it CRITICAL.
q_bad = {"qid": "bad", "doc_id": "A.java", "quote": boiler,
         "question": "What does SystemProperties.getProperty return on failure?",
         "answer": "null"}
q_ok = {"qid": "ok", "doc_id": "B.java",
        "quote": "a unique sentence appears exactly once here.",
        "question": "What does StringUtils.substringBefore return for a missing separator?",
        "answer": "the original string"}
found = dict((q["qid"], [lvl for lvl, _ in iss]) for q, iss in audit([q_bad, q_ok], docs))
check("boilerplate evidence is flagged CRITICAL",
      "CRITICAL" in found.get("bad", []), True)
check("a unique-evidence question is not flagged critical",
      "CRITICAL" in found.get("ok", []), False)
check("threshold is documented and low", MAX_QUOTE_OCCURRENCES <= 3, True)


# --------------------------------------------------- E5/E6 retrieval stacks
section("retriever stacks (LLM stubbed)")
# E5 and E6 are the two slowest rows in the table -- tens of minutes each -- so
# a bug in how the stack is assembled is worst discovered after the run, not
# during it. The LLM is stubbed so the WIRING is tested without the latency.
import numpy as _np                                                # noqa: E402
from src import llm as _llm, retrievers as _r                      # noqa: E402
from src.chunkers import Chunk as _Chunk                           # noqa: E402

_saved = (_llm.generate_json, _llm.generate, _llm.embed)
_calls = {"rerank": 0, "rewrite": 0}


def _fake_json(prompt, model=None, system=None, retries=2):
    if "Rate how well each passage" in prompt:
        _calls["rerank"] += 1
        return {"scores": [{"id": i, "score": 3 if i == 2 else 1} for i in range(1, 7)]}
    if "alternative search queries" in prompt:
        _calls["rewrite"] += 1
        return {"queries": ["alt one", "alt two"]}
    return {}


class _FakeStore:
    def __init__(self, n=40):
        self.cs = [_Chunk(f"c{i}", "d.java", f"passage {i} text", 0, 10) for i in range(n)]
        self.d = {c.chunk_id: c for c in self.cs}

    def ids(self):
        return [c.chunk_id for c in self.cs]

    def texts(self):
        return [c.text for c in self.cs]

    def get(self, cid):
        return self.d.get(cid)

    def search(self, qv, k):
        return [(c, 1.0 / (i + 1)) for i, c in enumerate(self.cs[:k])]


try:
    _llm.generate_json = _fake_json
    _llm.generate = lambda p, **k: "a hypothetical passage"
    _llm.embed = lambda xs, **k: _np.ones((len(xs), 768), dtype="float32")
    _store = _FakeStore()

    _cfg5 = {"retriever": "hybrid", "rrf_k": 60, "rerank": True,
             "rerank_pool": 12, "rerank_batch": 6, "top_k": 10}
    _out5 = _r.build(_cfg5, _store).retrieve("q", 10)
    check("E5 returns k results", len(_out5), 10)
    # pool 12 at batch 6 is exactly two calls per question -- the figure the
    # runtime estimate in the README is built on.
    check("E5 makes pool/batch re-rank calls", _calls["rerank"], 2)

    _calls["rerank"] = 0
    _cfg6 = dict(_cfg5, query_rewrite="expand", rewrite_n=2)
    _out6 = _r.build(_cfg6, _store).retrieve("q", 10)
    check("E6 returns k results", len(_out6), 10)
    check("E6 rewrites the query exactly once", _calls["rewrite"], 1)
    check("E6 re-ranks every variant", _calls["rerank"] >= 3, True)

    # A malformed judge response must degrade to the base ranking, never to an
    # empty or all-zero result -- otherwise a flaky model silently zeroes a row.
    _llm.generate_json = lambda p, **k: {}
    check("malformed re-rank output still returns a full ranking",
          len(_r.build(_cfg5, _store).retrieve("q", 10)), 10)
finally:
    _llm.generate_json, _llm.generate, _llm.embed = _saved


# ------------------------------------------------------------------- verdict
print("\n" + "=" * 60)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("\nFAILURES:")
    for f in FAIL:
        print("  " + f)
    sys.exit(1)
print("The instrument checks out.")
