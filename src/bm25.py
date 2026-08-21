"""
BM25 (Okapi), written out rather than imported.

Why it exists in a RAG system: dense embeddings are good at meaning and bad at
exact tokens. Ask for "Clause 7.2" or "get_billing_account()" and an embedding
model gives you something semantically nearby but wrong. BM25 nails exact terms.
They fail differently, which is exactly why fusing them works.

The formula:

    score(q, d) = sum over terms t in q of:

                              f(t,d) * (k1 + 1)
        IDF(t) * -----------------------------------------
                  f(t,d) + k1 * (1 - b + b * |d| / avgdl)

    IDF(t) = ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5))

    f(t,d)  = how often term t appears in document d  (term frequency)
    |d|     = length of d in tokens
    avgdl   = average document length in the corpus
    n(t)    = how many documents contain t
    N       = total documents
    k1 ~1.5 = term-frequency saturation. 10 occurrences is not 10x better than 1.
    b  ~0.75= length normalisation. Stops long documents winning by volume alone.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> List[str]:
    """Lowercase + alphanumeric. Underscore kept on purpose so snake_case
    identifiers survive intact if you are indexing code."""
    return _TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in corpus]
        self.N = len(self.docs)
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0

        self.tf: List[Counter] = [Counter(d) for d in self.docs]
        df = Counter()
        for d in self.docs:
            df.update(set(d))
        self.idf = {
            t: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
            for t, n in df.items()
        }

    def score(self, query: str, idx: int) -> float:
        s = 0.0
        norm = self.k1 * (1 - self.b + self.b * self.doc_len[idx] / (self.avgdl or 1))
        for t in tokenize(query):
            f = self.tf[idx].get(t, 0)
            if not f:
                continue
            s += self.idf.get(t, 0.0) * (f * (self.k1 + 1)) / (f + norm)
        return s

    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Returns [(doc_index, score)] sorted descending. Only scores documents
        that contain at least one query term -- everything else is 0 anyway."""
        candidates = set()
        for t in tokenize(query):
            for i, tf in enumerate(self.tf):
                if t in tf:
                    candidates.add(i)
        scored = [(i, self.score(query, i)) for i in candidates]
        scored.sort(key=lambda x: -x[1])
        return scored[:k]


def reciprocal_rank_fusion(rankings: List[List[int]], k: int = 60,
                           weights: List[float] = None) -> List[Tuple[int, float]]:
    """Merge several ranked ID lists into one.

        RRF_score(d) = sum over lists of  weight_L / (k + rank_of_d_in_L)

    The trick is that it only uses RANK, never the raw score. BM25 scores and
    cosine similarities live on completely different scales, so averaging them
    directly is meaningless. Ranks are comparable by construction.

    k=60 is the value from the original Cormack et al. paper. It dampens the
    influence of the very top position so one confident-but-wrong retriever
    cannot dominate the merged list.
    """
    weights = weights or [1.0] * len(rankings)
    scores = {}
    for w, ranking in zip(weights, rankings):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])
