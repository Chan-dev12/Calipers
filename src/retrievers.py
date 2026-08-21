"""
Retrievers -- the contestants in your ablation table.

Every one of them exposes the same method:

    retrieve(query: str, k: int) -> List[(chunk_id, score)]

That uniform interface is what lets the eval runner treat them interchangeably.
Swapping E4 for E5 in your experiment matrix becomes a config change, not a
code change -- which is the difference between running 7 clean experiments and
running 7 messy ones.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from . import llm
from .bm25 import BM25, reciprocal_rank_fusion
from .store import VectorStore


class DenseRetriever:
    """Embed the query, cosine-search the matrix. This is E0, the control.

    Strength: understands paraphrase. "how do I cancel" finds "termination
    procedure" with no shared words.
    Weakness: exact identifiers. Ask for "Clause 7.2" and it returns clauses
    7.1, 7.3 and 8.4 -- all semantically adjacent, all wrong.
    """
    name = "dense"

    def __init__(self, store: VectorStore):
        self.store = store

    def retrieve(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        qv = llm.embed([query])[0]
        return [(c.chunk_id, s) for c, s in self.store.search(qv, k)]


class BM25Retriever:
    """Pure keyword. The mirror image of dense: nails exact tokens, blind to
    paraphrase. Worth running alone as E3 -- the number is usually higher than
    people expect, and that surprise is a good thing to report."""
    name = "bm25"

    def __init__(self, store: VectorStore):
        self.store = store
        self.ids = store.ids()
        self.index = BM25(store.texts())

    def retrieve(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        return [(self.ids[i], s) for i, s in self.index.search(query, k)]


class HybridRetriever:
    """Dense + BM25, merged with Reciprocal Rank Fusion.

    The reason this works is NOT that two retrievers are better than one. It is
    that these two fail on DIFFERENT queries. Fusing correlated retrievers buys
    you nothing; fusing complementary ones buys you a lot.

    Worth proving rather than asserting: log which queries only dense got right
    and which only BM25 got right. If that overlap is small, you have evidence
    for why hybrid helps. Put that in your writeup.

    Over-fetch (k * pool_mult) before fusing, or you throw away candidates that
    RRF would have promoted.
    """
    name = "hybrid"

    def __init__(self, store: VectorStore, rrf_k: int = 60,
                 weights: Tuple[float, float] = (1.0, 1.0), pool_mult: int = 3):
        self.dense = DenseRetriever(store)
        self.sparse = BM25Retriever(store)
        self.rrf_k, self.weights, self.pool_mult = rrf_k, list(weights), pool_mult

    def retrieve(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        pool = k * self.pool_mult
        d_ids = [cid for cid, _ in self.dense.retrieve(query, pool)]
        s_ids = [cid for cid, _ in self.sparse.retrieve(query, pool)]
        fused = reciprocal_rank_fusion([d_ids, s_ids], k=self.rrf_k, weights=self.weights)
        return fused[:k]


# --------------------------------------------------------------- re-ranking
RERANK_PROMPT = """Rate how well each passage answers the question.

0 = irrelevant
1 = same topic, does not answer it
2 = partially answers it
3 = directly and fully answers it

QUESTION: {query}

PASSAGES:
{passages}

Score every passage. Return ONLY:
{{"scores": [{{"id": 1, "score": 0}}, {{"id": 2, "score": 0}}]}}"""


class LLMReranker:
    """Two-stage retrieval: cheap retriever proposes, expensive scorer disposes.

    THE IDEA
    Dense retrieval is a bi-encoder -- query and passage are embedded separately
    and never meet. Fast, because passages are embedded once in advance, but the
    model never gets to look at the pair together. A cross-encoder does look at
    them together, which is far more accurate and far too slow to run over the
    whole corpus. So: fetch a small candidate pool cheaply, then sort that pool
    carefully. Standard in production RAG.

    LISTWISE, NOT POINTWISE -- AND WHY
    The obvious implementation sends one LLM call per candidate. With a pool of
    30 over 45 questions that is 1350 calls, which on CPU inference is roughly
    two hours per experiment. Unrunnable.

    Instead this scores a BATCH of passages in a single call. Fewer calls means
    less per-call overhead and the instruction block is not re-processed 30
    times. Combined with a smaller pool it turns hours into tens of minutes.

    This is not a hack to save time -- listwise re-ranking is a real technique
    (RankGPT and similar), and it has a genuine advantage: the model sees the
    candidates side by side and can compare them, rather than judging each in
    isolation against an imagined standard. It has a genuine disadvantage too:
    scores can drift with batch composition and long batches strain a small
    model's attention. Say both in your writeup.

    WHAT TO WATCH FOR IN YOUR RESULTS
    Re-ranking usually barely moves recall@10 (the same chunks come back) but
    moves MRR and NDCG a lot (they come back in a better order). If your table
    shows exactly that pattern, call it out -- it proves you know what each
    metric actually measures.
    """
    name = "llm-rerank"

    def __init__(self, base, store: VectorStore, pool: int = 12,
                 batch: int = 6, model: str = None, passage_chars: int = 700):
        self.base, self.store = base, store
        self.pool, self.batch, self.model = pool, batch, model
        self.passage_chars = passage_chars

    def retrieve(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        candidates = self.base.retrieve(query, self.pool)
        chunks = [(cid, self.store.get(cid)) for cid, _ in candidates]
        chunks = [(cid, c) for cid, c in chunks if c is not None]

        scored = []
        for start in range(0, len(chunks), self.batch):
            group = chunks[start:start + self.batch]
            listing = "\n\n".join(
                f"[{i}] {c.text[:self.passage_chars]}"
                for i, (_, c) in enumerate(group, start=1))
            r = llm.generate_json(
                RERANK_PROMPT.format(query=query, passages=listing),
                model=self.model)

            by_id = {}
            for item in (r.get("scores") or []):
                try:
                    by_id[int(item["id"])] = float(item["score"])
                except (KeyError, TypeError, ValueError):
                    continue

            for i, (cid, _) in enumerate(group, start=1):
                prior_rank = start + i - 1
                # fall back to original rank order if the model skipped this id,
                # so a malformed response degrades to the base retriever rather
                # than silently zeroing out good candidates
                s = by_id.get(i, 0.0)
                scored.append((cid, s - prior_rank * 1e-4))

        scored.sort(key=lambda x: -x[1])
        return scored[:k]


# ---------------------------------------------------------- query rewriting
REWRITE_PROMPT = """Rewrite this question into {n} alternative search queries.
Use vocabulary a formal document would use, not casual phrasing.
Keep every proper noun, identifier and number exactly as written.

QUESTION: {query}

Return ONLY: {{"queries": ["...", "..."]}}"""

HYDE_PROMPT = """Write a short passage (2-4 sentences) that would plausibly answer
this question, as if extracted from a reference document. Do not hedge, do not
say you are unsure -- just write the passage.

QUESTION: {query}"""


class QueryRewriter:
    """Wraps any retriever and expands the query before searching.

    THE PROBLEM
    Users write "what if I pay late". The document says "default and remedies
    upon delinquent payment". Zero shared vocabulary. Retrieval fails not
    because the index is bad but because the question and the answer are written
    in different registers.

    TWO STRATEGIES, both here:

    'expand' -- generate N paraphrases, retrieve for each, fuse with RRF.

    'hyde'   -- Hypothetical Document Embeddings. Ask the LLM to HALLUCINATE an
                answer, then embed that instead of the question. Sounds absurd,
                works well: a fake answer looks structurally much more like a
                real passage than a question does, so it lands closer in
                embedding space. The hallucination is never shown to anyone --
                it is used only as a search key.

    HONEST WARNING
    This is the technique most likely to fail on your corpus, especially stacked
    on top of hybrid + re-ranking, which have already fixed the vocabulary
    problem from the other side. If E6 comes out flat or worse than E5, report
    it. "Query rewriting added nothing once hybrid retrieval was in place, and
    here is why" is a better finding than a fifth incremental win.
    """
    name = "query-rewrite"

    def __init__(self, base, mode: str = "expand", n: int = 3, model: str = None):
        self.base, self.mode, self.n, self.model = base, mode, n, model

    def _variants(self, query: str) -> List[str]:
        if self.mode == "hyde":
            doc = llm.generate(HYDE_PROMPT.format(query=query), model=self.model)
            return [query, doc] if doc else [query]
        r = llm.generate_json(
            REWRITE_PROMPT.format(query=query, n=self.n), model=self.model)
        alts = [q for q in r.get("queries", []) if isinstance(q, str) and q.strip()]
        return [query] + alts[:self.n]

    def retrieve(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        variants = self._variants(query)
        if len(variants) == 1:
            return self.base.retrieve(query, k)
        rankings = [[cid for cid, _ in self.base.retrieve(v, k * 2)] for v in variants]
        # original query weighted double -- the rewrites are hypotheses, not gospel
        weights = [2.0] + [1.0] * (len(rankings) - 1)
        return reciprocal_rank_fusion(rankings, k=60, weights=weights)[:k]


def build(config: dict, store: VectorStore):
    """Assemble a retrieval stack from a plain dict. This is what makes each
    experiment a config file instead of a code edit."""
    r = config.get("retriever", "dense")
    if r == "dense":
        base = DenseRetriever(store)
    elif r == "bm25":
        base = BM25Retriever(store)
    elif r == "hybrid":
        base = HybridRetriever(store,
                               rrf_k=config.get("rrf_k", 60),
                               weights=tuple(config.get("rrf_weights", (1.0, 1.0))))
    else:
        raise ValueError(f"unknown retriever: {r}")

    if config.get("rerank"):
        base = LLMReranker(base, store,
                           pool=config.get("rerank_pool", 12),
                           batch=config.get("rerank_batch", 6))
    if config.get("query_rewrite"):
        base = QueryRewriter(base, mode=config.get("query_rewrite"),
                             n=config.get("rewrite_n", 3))
    return base
