"""
LLM-as-judge: faithfulness and correctness, plus the machinery to check the
judge itself.

TWO METRICS, DELIBERATELY SEPARATE

  faithfulness -- is every claim in the answer supported by the context it was
                  given? Scored WITHOUT the reference answer, because that is
                  the question being asked: did the model stay inside its
                  evidence. An answer can be faithful and wrong (the retriever
                  supplied the wrong passage and the generator reported it
                  accurately) and that is a retrieval failure, not a generation
                  one.

  correctness  -- does the answer match the human-verified reference? Scored
                  WITHOUT the context, so the judge cannot be talked into
                  accepting a plausible-looking passage in place of the truth.

Keeping them apart is what lets the ablation attribute a failure. Collapsed into
one "quality" score, a retrieval regression and a generation regression look
identical.

WHY THE JUDGE GETS ITS OWN AUDIT

An LLM judge is an instrument, and this project's whole claim is that it
measures rather than demos. An unvalidated judge produces a confident number
with no error bar and no way to tell whether it tracks human opinion at all. So
`tools/judge_agreement.py` hand-labels a sample of the same items and reports
raw agreement AND Cohen's kappa.

Kappa matters more than raw agreement here. If 90% of answers are faithful, a
judge that blindly answers "faithful" every time scores 90% agreement while
carrying zero information. Kappa corrects for exactly that chance agreement:
it would score that same degenerate judge at 0.0.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from . import llm

FAITHFULNESS_PROMPT = """You are grading whether an ANSWER stays within its EVIDENCE.

Score:
2 = every claim in the answer is stated in the evidence
1 = mostly supported, but contains a detail the evidence does not state
0 = contradicts the evidence, or invents information not present

Judge ONLY whether the evidence supports the answer.
Do NOT judge whether the answer is true in general.

EVIDENCE:
{context}

ANSWER: {answer}

Return ONLY: {{"score": 0, "reason": "..."}}"""

CORRECTNESS_PROMPT = """You are grading whether an ANSWER matches a REFERENCE answer.

Score:
2 = same meaning as the reference
1 = partially correct, or missing an essential part
0 = wrong, or contradicts the reference

Wording may differ. Judge meaning, not phrasing.

QUESTION: {question}
REFERENCE: {reference}
ANSWER: {answer}

Return ONLY: {{"score": 0, "reason": "..."}}"""


def _score(prompt: str, model: Optional[str]) -> Dict:
    """Run one judge call and normalise the result.

    An unparseable judge response is recorded as score None, never as 0. A
    parse failure is a broken measurement, and averaging it in as a zero would
    silently push the reported score down in proportion to how often the judge
    malfunctioned -- turning an instrument fault into what looks like a model
    result.
    """
    r = llm.generate_json(prompt, model=model or llm.JUDGE_MODEL)
    raw = r.get("score")
    try:
        s = int(float(raw))
    except (TypeError, ValueError):
        return {"score": None, "reason": "unparseable judge response"}
    if s not in (0, 1, 2):
        return {"score": None, "reason": f"out-of-range judge score: {s}"}
    return {"score": s, "reason": str(r.get("reason", ""))[:300]}


def faithfulness(answer: str, context: str, model: Optional[str] = None) -> Dict:
    if not context.strip():
        return {"score": None, "reason": "no context to be faithful to"}
    return _score(FAITHFULNESS_PROMPT.format(
        context=context[:4000], answer=answer), model)


def correctness(question: str, answer: str, reference: str,
                model: Optional[str] = None) -> Dict:
    return _score(CORRECTNESS_PROMPT.format(
        question=question, reference=reference, answer=answer), model)


# ----------------------------------------------------------- judge validation
def cohens_kappa(a: List[int], b: List[int]) -> Optional[float]:
    """Agreement between two raters, corrected for agreement by chance.

        kappa = (p_observed - p_chance) / (1 - p_chance)

    1.0 = perfect, 0.0 = no better than guessing at the base rate, < 0 = worse
    than chance. Landis & Koch call 0.41-0.60 moderate, 0.61-0.80 substantial.

    Returns None when the two raters between them used only one label: p_chance
    is then 1.0 and kappa is 0/0. That case is undefined, not zero, and it is
    worth surfacing rather than printing a number -- it usually means the sample
    was too easy or too small to tell you anything.
    """
    if not a or len(a) != len(b):
        return None
    n = len(a)
    labels = sorted(set(a) | set(b))
    if len(labels) < 2:
        return None
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    chance = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    if abs(1.0 - chance) < 1e-12:
        return None
    return (observed - chance) / (1.0 - chance)


def agreement_report(human: List[int], machine: List[int]) -> Dict:
    """Compare hand labels against judge labels on the same items."""
    pairs = [(h, m) for h, m in zip(human, machine)
             if h is not None and m is not None]
    if not pairs:
        return {"n": 0}
    h, m = [p[0] for p in pairs], [p[1] for p in pairs]
    exact = sum(1 for x, y in pairs if x == y) / len(pairs)
    # Adjacent agreement: 1-vs-2 is a disagreement about degree, 0-vs-2 is a
    # disagreement about kind. On a 3-point scale those are not the same error
    # and reporting only exact match overstates how badly the judge does.
    within_one = sum(1 for x, y in pairs if abs(x - y) <= 1) / len(pairs)
    return {
        "n": len(pairs),
        "exact_agreement": round(exact, 4),
        "within_one": round(within_one, 4),
        "cohens_kappa": (round(k, 4)
                         if (k := cohens_kappa(h, m)) is not None else None),
        "human_mean": round(sum(h) / len(h), 4),
        "judge_mean": round(sum(m) / len(m), 4),
    }
