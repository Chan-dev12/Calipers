"""
Retrieval metrics. This file is the scoreboard -- everything else is a contestant.

Each function takes:
    retrieved : list of chunk_ids, ranked best-first
    gold      : set of chunk_ids that are actually correct
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Set


def hit_rate_at_k(retrieved: Sequence[str], gold: Set[str], k: int) -> float:
    """Did AT LEAST ONE correct chunk appear in the top k?  1.0 or 0.0.
    The most forgiving metric. Answers: 'did the system have a chance?'"""
    return 1.0 if set(retrieved[:k]) & gold else 0.0


def recall_at_k(retrieved: Sequence[str], gold: Set[str], k: int) -> float:
    """What FRACTION of the correct chunks appeared in the top k?
    Stricter than hit rate when a question needs evidence from several places."""
    if not gold:
        return 0.0
    return len(set(retrieved[:k]) & gold) / len(gold)


def precision_at_k(retrieved: Sequence[str], gold: Set[str], k: int) -> float:
    """What fraction of what you returned was actually useful?
    Matters because every junk chunk you stuff into the prompt costs context
    and gives the generator another chance to get distracted."""
    if k == 0:
        return 0.0
    return len(set(retrieved[:k]) & gold) / k


def reciprocal_rank(retrieved: Sequence[str], gold: Set[str]) -> float:
    """1 / (position of the first correct result). 1st -> 1.0, 2nd -> 0.5,
    5th -> 0.2, never found -> 0.0. Averaged over all questions this is MRR.

    Use it to catch the failure mode where the right chunk IS retrieved but
    sits at position 9 -- so a re-ranker, not a different retriever, is your fix."""
    for i, cid in enumerate(retrieved, start=1):
        if cid in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], gold: Set[str], k: int) -> float:
    """Normalised Discounted Cumulative Gain, binary relevance.

    DCG  = sum of  rel_i / log2(i + 1)   -- a hit at rank 1 is worth more than at rank 5
    IDCG = the DCG of a perfect ranking  -- all gold chunks packed at the top
    NDCG = DCG / IDCG, so 1.0 means a perfect ordering

    Unlike MRR it credits EVERY correct chunk, not just the first one, while
    still caring about position. It is the metric to lead with in your table."""
    dcg = sum(1.0 / math.log2(i + 1)
              for i, cid in enumerate(retrieved[:k], start=1) if cid in gold)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return (dcg / idcg) if idcg > 0 else 0.0


def evaluate_run(runs: List[Dict], ks: Sequence[int] = (1, 3, 5, 10)) -> Dict[str, float]:
    """runs = [{'retrieved': [chunk_id, ...], 'gold': {chunk_id, ...}}, ...]
    Returns averaged metrics across all questions."""
    if not runs:
        return {}
    agg: Dict[str, List[float]] = {}

    def add(name, val):
        agg.setdefault(name, []).append(val)

    for r in runs:
        ret, gold = r["retrieved"], set(r["gold"])
        add("mrr", reciprocal_rank(ret, gold))
        for k in ks:
            add(f"hit@{k}", hit_rate_at_k(ret, gold, k))
            add(f"recall@{k}", recall_at_k(ret, gold, k))
            add(f"precision@{k}", precision_at_k(ret, gold, k))
            add(f"ndcg@{k}", ndcg_at_k(ret, gold, k))

    return {name: sum(v) / len(v) for name, v in sorted(agg.items())}


def bootstrap_ci(runs: List[Dict], metric_fn, n_boot: int = 1000,
                 seed: int = 0) -> tuple:
    """95% confidence interval by bootstrap resampling.

    Include this. With only 50-100 questions, a 2-point difference between two
    systems is probably noise. Reporting an interval instead of a bare number is
    the single easiest way to look like you know what you are doing -- and it
    stops you from claiming a win you did not actually get.
    """
    import random
    rng = random.Random(seed)
    scores = [metric_fn(r["retrieved"], set(r["gold"])) for r in runs]
    n = len(scores)
    means = []
    for _ in range(n_boot):
        sample = [scores[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return (means[int(0.025 * n_boot)], means[int(0.975 * n_boot)])
