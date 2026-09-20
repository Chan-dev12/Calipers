"""
Answer generation over retrieved context.

Retrieval metrics answer "did the right passage come back?". They do not answer
"did the system then say something true?" -- and those come apart constantly. A
retriever can put the gold chunk at rank 1 and the generator can still contradict
it, or ignore it and answer from the model's own memory. Measuring only
retrieval hides that failure entirely.

So this module is deliberately thin. It does exactly one thing -- turn ranked
chunks into an answer -- and it does it the same way for every experiment, so
that when generation quality moves between rows, RETRIEVAL is the only thing
that changed. A generator that varied per experiment would confound the table it
is supposed to be measuring.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from . import llm
from .store import VectorStore

# Grounding instructions, not politeness instructions.
#
# "I don't know" has to be an explicitly permitted answer. Without that licence a
# small model will always produce something, and a confabulated answer over an
# empty context scores as a generation failure when it is really a retrieval
# failure. Allowing the refusal is what separates those two cases in the numbers.
ANSWER_PROMPT = """Answer the question using ONLY the passages below.

Rules:
- Use only what the passages state. Do not add outside knowledge.
- If the passages do not contain the answer, reply exactly: NOT_IN_CONTEXT
- Be brief. One or two sentences.

PASSAGES:
{context}

QUESTION: {question}

ANSWER:"""

NOT_IN_CONTEXT = "NOT_IN_CONTEXT"


def build_context(chunk_ids: List[str], store: VectorStore,
                  top_n: int = 5, passage_chars: int = 700) -> str:
    """Format the top-n retrieved chunks as a numbered context block.

    top_n is smaller than the retrieval k on purpose. The table scores retrieval
    at k=10, but stuffing ten passages into a 4096-token window on CPU is both
    slow and counterproductive -- precision@5 is around 0.25 here, so ranks 6-10
    are mostly distractors, and every distractor is another chance for the model
    to answer from the wrong passage.
    """
    parts = []
    for i, cid in enumerate(chunk_ids[:top_n], start=1):
        c = store.get(cid)
        if c is None:
            continue
        parts.append(f"[{i}] {c.text[:passage_chars]}")
    return "\n\n".join(parts)


def answer_one(question: str, chunk_ids: List[str], store: VectorStore,
               top_n: int = 5, model: Optional[str] = None) -> Dict:
    """Generate one answer. Returns the answer plus the context it was given.

    The context is returned, not just the answer, because the judge has to score
    faithfulness against the EXACT text the generator saw. Re-deriving it later
    from chunk ids would risk scoring against a different context than the one
    that produced the answer -- and a judge scoring the wrong evidence is worse
    than no judge.
    """
    context = build_context(chunk_ids, store, top_n=top_n)
    if not context.strip():
        # No retrievable context at all. Record it as a refusal rather than
        # asking the model to answer from nothing -- otherwise a pure retrieval
        # failure would be logged as a hallucination.
        return {"answer": NOT_IN_CONTEXT, "context": "", "empty_context": True}

    answer = llm.generate(
        ANSWER_PROMPT.format(context=context, question=question),
        model=model, temperature=0.0)
    return {"answer": (answer or "").strip(), "context": context,
            "empty_context": False}


def is_refusal(answer: str) -> bool:
    """Did the generator decline to answer?

    Counted separately from a wrong answer throughout. A refusal on missing
    context is CORRECT behaviour and the system should not be penalised for it;
    a refusal on good context is a generation failure. Collapsing the two into
    one "wrong" bucket would make a well-behaved system look identical to a
    broken one.
    """
    return NOT_IN_CONTEXT in (answer or "").upper()
