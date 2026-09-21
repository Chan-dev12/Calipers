# Calipers

*Measuring RAG instead of demoing it.*

A retrieval system built from scratch, and — the actual point — an evaluation
harness that shows which design choices helped and by how much.

No LangChain. No LlamaIndex. No vector database. Two pip packages total.
Runs entirely on a laptop; nothing leaves the machine.

---

## The claim

Everyone builds RAG. Almost noone measures it. This repo produces an ablation
table where each row is a controlled experiment against a human-verified answer
key, reported with bootstrap confidence intervals.

The engineering here is ordinary on purpose. The discipline is not.

---

## Stack

| Component | Choice | Notes |
|---|---|---|
| Embeddings | `nomic-embed-text` via Ollama | 768-dim, ~270 MB, fits in 2 GB VRAM |
| Generation | `gemma3:1b` (dev) / `gemma4:e2b` (final) | CPU |
| Judge | `gemma4:e2b` | at least as strong as the generator, on purpose |
| Judge validation | Cohen's kappa vs hand labels | `tools/judge_agreement.py` |
| Vector index | normalised numpy matrix | exact search, no ANN |
| BM25 | hand-written, `src/bm25.py` | ~40 lines |
| Re-ranking | listwise, via the local LLM | no cross-encoder dependency |

Dependencies: `numpy`, `requests`. That is the entire list.

---

## Setup

```bash
# 1. Ollama (https://ollama.com/download)
ollama serve
ollama pull nomic-embed-text
ollama pull gemma3:1b        # fast dev loop
ollama pull gemma4:e2b       # final runs + judge

# 2. Python
python -m venv .venv && source .venv/bin/activate    # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Corpus (see below)
```

Verify — in this order, because each check rules out a different failure:
```bash
python tools/selftest.py        # the metrics, offline. no Ollama, ~1s
python tools/probe_ollama.py    # endpoints, models, GPU clock, throughput
```

`probe_ollama.py` reports the **GPU clock against its boost clock**. Do not skip
that line. On a laptop the same embedding call is ~7× slower on battery, nothing
in the stack reports it, and it looks exactly like slow code — it is the single
biggest source of wasted time in this project's history.

---

## Corpus

47 Java source files from **Apache Commons Lang**
(`src/main/java/org/apache/commons/lang3/`), ~1.83 M characters. Chunk count
depends on the chunker under test — 2,263 fixed, 2,647 recursive, 2,953 with
Java separators — which is itself one of the things the table has to account for.

Not vendored into this repo — reproduce it:

```bash
git clone --depth 1 https://github.com/apache/commons-lang.git
cp commons-lang/src/main/java/org/apache/commons/lang3/*.java corpus/
```

**Why this corpus.** Dense Javadoc gives real evidence spans, and the library is
well documented but not so famous that a small model can answer from memory. It
also turned out to have a property worth knowing about — see *Findings*.

---

## Pipeline

```
corpus/ ──> chunkers.py ──> llm.embed ──> store.py (numpy matrix)
                                               │
evalset/questions.jsonl ──> resolve_gold ──────┤
        │                                      ▼
        │     retrievers.py ──> metrics.py ──> results/*.json ──> report.py
        │                                      │
        │                                      ▼  (optional second half)
        └──────────────────> generation.py ──> judge.py ──> *.generation.json
                                                   │
                              human labels ──> judge_agreement.py (kappa)
```

The lower branch is optional: five of the eight retrieval rows never call an LLM
at all, and the retrieval table is complete without it. Generation is measured
separately so that a change there can never be mistaken for a retrieval result.

---

## Usage

```bash
# build the answer key
python tools/make_evalset.py draft --n 40     # LLM drafts candidates
python tools/make_evalset.py review           # human verifies, one at a time
python tools/audit_evalset.py                 # mechanical defect check
python tools/audit_evalset.py --prune         # drop the critical ones

# check the harness itself before measuring anything with it
python tools/selftest.py

# experiments -- chunkers first, since everything downstream inherits the winner
python tools/run_eval.py E0_baseline E1_recursive E1b_recursive_java E1c_recursive_java_matched
python tools/run_eval.py E2_semantic                  # slow: one embed per sentence
python tools/pick_chunker.py                          # rewrites E3-E6 to match
python tools/run_eval.py E3_bm25 E4_hybrid
python tools/run_eval.py E5_rerank E6_query_rewrite   # slow: LLM in the loop
python tools/report.py --save

# generation + judging (optional second half; retrieval rows do not need it)
python tools/run_generation.py E4_hybrid          # answer, then score faithfulness
python tools/judge_agreement.py E4_hybrid         # you hand-label; reports kappa
```

---

## Experiment matrix

| ID | Chunker | Retriever | Re-rank | Query | Question it answers |
|----|---------|-----------|---------|-------|---------------------|
| E0 | fixed | dense | — | raw | Control |
| E1 | recursive | dense | — | raw | Does structure-aware splitting help? |
| E1b | recursive (Java separators) | dense | — | raw | Was E1's loss the algorithm, or its separator list? |
| E1c | recursive (Java separators, size matched) | dense | — | raw | Control for E1b: same realised chunk length as E1, only separators differ |
| E2 | semantic | dense | — | raw | Is the embedding cost worth it? |
| E3 | best of E0–E2 | bm25 | — | raw | How far does keyword-only get? |
| E4 | best | hybrid + RRF | — | raw | Do they fail differently enough to fuse? |
| E5 | best | hybrid + RRF | LLM | raw | Does re-ranking fix ordering? |
| E6 | best | hybrid + RRF | LLM | rewritten | Does query rewriting add anything on top? |

One variable per row. That is the whole discipline — it is what lets you say
"recursive chunking gave +9 points" instead of "I changed some things and it got
better."

"Best" in rows E3–E6 is a dependency between experiments, so it is enforced by
`tools/pick_chunker.py` rather than by hand. It had been wrong from the start:
E0 (fixed) beat E1 (recursive), yet all four downstream configs specified
`recursive`, building every one of them on the chunker the ablation had already
rejected. A markdown table cannot check a JSON file, so the rule is executable
now.

---

## Results

Generated by `python tools/report.py --save --readme`, spliced in between
markers so this table cannot drift from `results/`.

<!-- ABLATION:START -->

| Experiment                 | Recall@5       | MRR            | NDCG@10        | P@5            | NDCG@10 95% CI | Chunks | Mean chars | Time  |
|----------------------------|----------------|----------------|----------------|----------------|----------------|--------|------------|-------|
| E0_baseline                | 0.684          | 0.532          | 0.558          | 0.186          | [0.414, 0.704] | 2263   | 987        | 2s    |
| E1_recursive               | 0.589 (-0.095) | 0.500 (-0.032) | 0.525 (-0.033) | 0.171 (-0.014) | [0.395, 0.647] | 2647   | 860        | 2s    |
| E1b_recursive_java         | 0.452 (-0.232) | 0.463 (-0.070) | 0.442 (-0.116) | 0.150 (-0.036) | [0.319, 0.569] | 2953   | 780        | 2s    |
| E1c_recursive_java_matched | 0.542 (-0.143) | 0.407 (-0.125) | 0.436 (-0.123) | 0.150 (-0.036) | [0.324, 0.563] | 2640   | 857        | 636s  |
| E2_semantic                | 0.696 (+0.012) | 0.452 (-0.080) | 0.534 (-0.024) | 0.164 (-0.021) | [0.409, 0.665] | 2205   | 836        | 1993s |
| E3_bm25                    | 0.643 (-0.042) | 0.509 (-0.023) | 0.553 (-0.006) | 0.150 (-0.036) | [0.404, 0.692] | 2263   | 987        | 1s    |
| E4_hybrid                  | 0.732 (+0.048) | 0.654 (+0.122) | 0.695 (+0.137) | 0.186 (+0.000) | [0.558, 0.822] | 2263   | 987        | 7s    |
| E5_rerank                  | 0.732 (+0.048) | 0.603 (+0.071) | 0.663 (+0.105) | 0.186 (+0.000) | [0.531, 0.785] | 2263   | 987        | 3216s |
| E6_query_rewrite           | 0.750 (+0.066) | 0.626 (+0.094) | 0.689 (+0.131) | 0.193 (+0.007) | [0.574, 0.797] | 2263   | 987        | 72s   |

n = 28 questions. Deltas are vs E0_baseline. CI by bootstrap resampling, 1000 iterations.

**Caveats**
- E1_recursive: NDCG@10 interval overlaps the baseline's. With n=28 this difference is not statistically distinguishable.
- E1b_recursive_java: NDCG@10 interval overlaps the baseline's. With n=28 this difference is not statistically distinguishable.
- E1c_recursive_java_matched: NDCG@10 interval overlaps the baseline's. With n=28 this difference is not statistically distinguishable.
- E2_semantic: NDCG@10 interval overlaps the baseline's. With n=28 this difference is not statistically distinguishable.
- E3_bm25: NDCG@10 interval overlaps the baseline's. With n=28 this difference is not statistically distinguishable.
- E4_hybrid: NDCG@10 interval overlaps the baseline's. With n=28 this difference is not statistically distinguishable.
- E5_rerank: NDCG@10 interval overlaps the baseline's. With n=28 this difference is not statistically distinguishable.
- E6_query_rewrite: NDCG@10 interval overlaps the baseline's. With n=28 this difference is not statistically distinguishable.

<!-- ABLATION:END -->

**How to read it.** `Mean chars` is there because `chunk_size` is held constant
in the configs but the *realised* chunk length is not, and score tracks the
realised length — so the chunker rows cannot separate "better boundaries" from
"more context per chunk". The CI column is the honest one: at n≈28 most of these
intervals overlap, and the report says so underneath rather than letting a
point estimate imply a result.

---

## Design decisions

**Ground truth is a quote, not a chunk ID.**
Chunk IDs change every time you re-chunk, so a chunk-ID answer key would be
invalid the moment the second experiment ran. `resolve_gold()` maps a stored
verbatim evidence span onto whatever chunking is currently active. This single
decision is what makes the chunking ablation possible at all.

**`temperature=0` everywhere.**
Non-deterministic generation makes an ablation table meaningless — you cannot
tell whether a delta came from your change or from sampling noise.

**The judge gets judged.**
An LLM scores faithfulness at scale (`src/judge.py`); `tools/judge_agreement.py`
then has you hand-label a sample of the same items and reports raw agreement,
agreement within one point, and Cohen's kappa. The difference between "scores
0.88 on faithfulness" and "scores 0.88 on a metric shown to be trustworthy."

Kappa rather than raw agreement, because raw agreement flatters a lazy judge: if
90% of answers are faithful, a judge that answers "faithful" unconditionally
scores 90% agreement while carrying no information. Kappa scores that same judge
at 0.0, and `tools/selftest.py` pins exactly that case.

**Faithfulness and correctness are scored separately, and never together.**
Faithfulness asks whether the answer stayed inside the evidence it was given, and
is scored *without* the reference. Correctness asks whether the answer matches
the human-verified reference, and is scored *without* the evidence. An answer can
be faithful and wrong — the retriever supplied the wrong passage and the
generator reported it accurately — and that is a retrieval failure, not a
generation one. Collapsed into a single "quality" score the two become
indistinguishable, and the ablation loses the ability to attribute a regression.

**An unscoreable item is `None`, never `0`.**
A judge that emits unparseable JSON, or a refusal that makes no claims for the
evidence to support, is excluded from the average rather than counted as zero.
Averaging a broken measurement in as a zero converts an instrument fault into
what looks like a model result — the score drops in proportion to how often the
judge malfunctioned, and nothing says so.

**Thinking mode off.**
Gemma 4 reasons before answering by default. Every call here is structured
extraction — score a passage, emit JSON — so the reasoning trace is pure cost on
CPU. `THINK=1` re-enables it if you want to test whether a reasoning judge
agrees with humans more often; that would be a legitimate extra row.

**No silent fallbacks.**
An early version caught request timeouts and retried a different endpoint,
turning "the model is still loading" into a misleading 404 from somewhere else.
In a project whose entire output is numbers you have to trust, a component that
hides failures is worse than one that crashes — a masked error becomes a wrong
number, and wrong numbers do not announce themselves.

**`chunk_size` 512 → 1000.**
512 characters is reasonable for prose and far too small for Java: a single
method with its Javadoc regularly exceeds it. Tuned to 1000/150 after inspecting
the chunk length distribution (mean 860, min 147, max 1151).

**Every result is stamped with the answer key that produced it.**
Results accumulate in `results/` over weeks while the eval set keeps being
pruned. Without a fingerprint, `report.py` will cheerfully place a row scored on
36 questions beside a row scored on 28 and print a delta between them — which is
how a *fix* to the answer key quietly becomes a fabricated comparison. Each run
records a hash of the questions it was scored against; the report groups by that
hash and excludes any row that does not match, naming it and the command to
re-run it. `pick_chunker.py` refuses to choose a winner across mismatched keys
at all.

**The harness has its own test suite.**
`tools/selftest.py` — 55 offline checks, no Ollama, no corpus, about a second. A
bug in `ndcg_at_k` does not crash anything; it produces a plausible number that
is wrong, and a wrong number in an ablation table is indistinguishable from a
finding. The metrics are checked against hand-computed values, and the two
defects that actually got through — the audit tool's false rejections and the
115-gold-chunk answer key — have regression tests.

---

## Why no libraries

| Dropped | Replaced by | Reason |
|---|---|---|
| RAGAS / DeepEval | `src/metrics.py` | Using an eval library in an eval project is circular — the evaluation *is* the contribution |
| `rank_bm25` | `src/bm25.py` | ~40 lines, and you should be able to derive the saturation term |
| Chroma / FAISS | `src/store.py` | At 2.6k chunks brute force is ~2 ms and **exact**; ANN is approximate and would cost recall the experiment would then misattribute |
| `sentence-transformers` | `LLMReranker` | On 2 GB VRAM, juggling three models is the real cost. A trade-off, not a principle |
| LangChain / LlamaIndex | `src/retrievers.py` | Writing the retrieval loop is the point of a learning project |

Ollama stays, because training an embedding model is a different project. The
honest boundary: **the models are borrowed, the RAG layer is mine.**

At production scale several of these invert — a vector DB above ~100k chunks, a
real cross-encoder, possibly LangChain. Knowing when *not* to build from scratch
is part of the point.

---

## Findings

Recorded as they came up. Several are about running on constrained hardware,
which surfaced failure modes the literature does not mention.

**The judge is harsher than a human, and its agreement is weak.**
The point of `judge_agreement.py` is that an unvalidated judge replaces one
unknown with another. Validated on 5 hand-labelled items (correctness) and 3
(faithfulness) it does not look trustworthy: exact agreement 0.60 on
correctness with **Cohen's kappa 0.286** -- "fair", barely above chance -- and
kappa **0.000** on faithfulness, meaning the judge carried no information about
faithfulness beyond guessing the base rate.

The direction is consistent: **human mean 1.20 against judge mean 0.40** on the
same correctness items. `gemma4:e2b` marks answers wrong that a person marks
right, so the reported `correctness_mean` of 0.8 is probably too pessimistic --
the generator may be better than the table says.

Two caveats, both of which cut against reading much into this. The sample is 5
items, far below the 25 intended, because the generation run was capped at 15
while the GPU was throttled. And four of the sampled items had their judge
verdicts discussed in the open before labelling, so they were not blind. What
this establishes is not "the judge is bad" but "the judge is not yet validated,
and the first evidence is not reassuring" -- which is exactly why the number is
reported rather than the faithfulness score alone.

**Four questions out of 36 inverted the headline result.**
The worst defect found so far, and the audit tool could not see it. Ground truth
is stored as a verbatim quote, and `resolve_gold()` maps that quote onto every
chunk containing it. Four questions cited a boilerplate Javadoc sentence —
*"Returns {@code null} if the property cannot be read due to a {@link
SecurityException}."* — which appears **193 times** in `SystemProperties.java`.
Each of those questions therefore had **115 gold chunks**, capping recall@5 at
5/115 = 0.043 no matter what the retriever returned.

Across the whole set the mean was 15 gold chunks per question; only 20 of 36 had
a unique one. Pruning the eight critical cases moved BM25 from **−0.010 NDCG@10
against the baseline to +0.069** — the degenerate questions had been masking the
result, and the table said keyword search was slightly worse than dense when it
is clearly better.

The existing checks could not catch this because it is not a question-wording
problem. A question can name an exact method, pass every specificity test, and
still cite evidence that appears verbatim in eighty other places. The defect
lives in the *quote*, and the only way to see it is to go back to the source
document and count. `tools/audit_evalset.py` now does, and `tools/selftest.py`
has the regression test.

**On a laptop, check the GPU clock before believing any benchmark.**
The single biggest source of wasted effort here. The same embedding call
measured 0.10 s plugged in and 1.54 s on battery: the MX550 drops to 300 MHz of
its 2100 MHz boost clock on battery power — a 7× derate at 5.3 W, with the GPU
at 57 °C, so it is power policy, not thermal throttling. Nothing in Ollama or the
client reports it. It looks exactly like the code got slower.

Combined with a second trap — a stray eval process still holding the server,
which made reads 10–100× slower — successive measurements of the *same*
operation ranged over three orders of magnitude and supported opposite
conclusions. `nvidia-smi --query-gpu=clocks.sm,clocks.max.sm --format=csv`, and
kill competing processes, before timing anything.

**Benchmarking on convenient data overstated a speedup by 10×.**
Batched embedding looked **14× faster** than one call at a time. That number came
from 55-character synthetic strings, where per-request overhead is most of the
cost and batching amortises nearly all of it. On real corpus sentences —
averaging 149 characters and running to 3,258 — the embedding model is
compute-bound and the honest figure is **1.36×**. Same project, same mistake the
eval set exists to prevent: measuring on a distribution that is not the one you
care about.

**2.9% of the corpus costs 59% of the embedding time.**
Semantic chunking splits on sentences, and `_SENT_RE` is prose-tuned — terminal
punctuation or a blank line. Java source offers neither for long stretches, so a
method body comes back as one 3,258-character "sentence". Because attention is
quadratic in sequence length, the 2.9% of sentences over 600 characters account
for **58.6%** of total embedding cost. They are also poor semantic units. Exactly
the same mismatch as E1's separator list: the algorithm is fine, its tokenisation
assumptions came from another document format.

**A chunker ablation does not hold chunk size constant -- so a control was run.**
`chunk_size: 1000` is identical across E0, E1 and E1b, but the *realised* mean
length is 987, 860 and 781 characters, and NDCG fell in exactly that order. So
"Java separators hurt" was not separable from "smaller chunks hurt", and the
first version of this finding said the question could not be answered.

E1c answers it. Setting `chunk_size: 1120` makes the Java separator set realise
a mean of **856.6** characters over **2,640** chunks, against prose E1's
**860.3** over **2,647** -- same realised length, same chunk count, only the
separator list differs. E1c scores **0.436** against E1's **0.525**.

So the separators really are the cause: holding size fixed costs 0.089 NDCG@10.
The supporting detail is that E1b (781 chars) and E1c (857 chars) land at 0.442
and 0.436 -- nearly identical despite a 76-character difference in mean length,
which is what you would expect if the separator list, not the size, is doing
the damage.

The uncomfortable part is that the Java separators were the *considered* choice.
They split on real Java boundaries -- method ends, Javadoc starts, declarations
-- and they lose to a list whose first two entries are markdown headings that
never occur in the corpus at all. A plausible mechanism: splitting at every
declaration severs each method from the Javadoc that explains it, and the
Javadoc is what the questions are actually asked about. Structure-aware is not
the same as retrieval-aware.

**`.gitignore` does not support trailing comments.****`.gitignore` does not support trailing comments.**
`.cache/           # regenerable` is not the pattern `.cache/` with a comment —
git treats the whole line as one literal pattern, which matches nothing. The
embedding cache and `index/` had therefore never been ignored, and a 23 MB
pickle was being tracked. Only a line *starting* with `#` is a comment.

**LLM-drafted questions hallucinate evidence at ~22%.**
Of 40 drafted candidates, 9 cited quotes that did not exist in the source chunk.
Caught automatically before human review by checking each quote against the
chunk text.

**Overloaded utility code produces ambiguous ground truth.**
`ArrayUtils.contains` exists for `Object[]`, `int[]`, `long[]` and more, each
carrying identical Javadoc. A question naming only "the contains method" has a
dozen equally-correct chunks while the key records one — so the retriever is
punished for being right. Questions on code corpora need method-signature-level
specificity. The drafting prompt now enforces this.

**Question wording leaks into retrieval scores.**
Where a drafted question reused the source's phrasing, retrieval collapsed into
string matching, which flatters BM25 and makes hybrid fusion look useless.
`tools/audit_evalset.py` flags lexical overlap above 0.6 as critical.

**The audit tool needed auditing.**
Its first specificity check only recognised CamelCase and `method()`, so it
wrongly flagged `IS_OS_MAC_OSX_CHEETAH` and `Boolean.TRUE` as ambiguous —
false-rejecting good questions. Same class of bug as a harness reporting wrong
numbers. The instrument that measures needs checking too.

**Model eviction under memory pressure.**
With 16 GB of RAM and three models in play, Ollama evicts the embedding model to
make room for a generation model; the next embed call then waits on a reload
that exceeds a short timeout and kills the whole indexing run. Fixed with
retry-with-backoff, a longer timeout, and explicit `keep_alive` residency.
Invisible on rented GPUs.

**KV cache is preallocated.**
`num_ctx: 8192` reserved memory that a 2,500-character prompt never used and
tipped a 7.2 GB model into a 500 error. Dropped to 4096.

---

## Running on modest hardware

Built and tuned on a 16 GB laptop with an MX550 (2 GB VRAM). Generation runs on
CPU; only the embedding model fits on the GPU.

This matters less than it sounds, because retrieval metrics need embeddings, not
generation — five of seven experiments never call the LLM:

| Experiment | LLM calls | Runtime (warm cache / cold) |
|---|---|---|
| E0 fixed + dense | none | 2 s / ~4 min to embed 2,263 chunks |
| E1 recursive + dense | none | 2 s / ~5 min |
| E1b recursive (Java seps) | none | 2 s / ~7 min |
| E2 semantic + dense | embeddings only | 3 s / **hours** — one call per *sentence*, 12,744 of them |
| E3 BM25 | none at all | 1 s, pure Python, no embeddings |
| E4 hybrid + RRF | none | 2 s |
| E5 + re-ranking | ~5 per question | 20–40 min |
| E6 + query rewriting | ~7 per question | 30–50 min |

Two things dominate the cold-cache numbers, and neither is the retrieval code.
E2 is slow because 2.9% of its "sentences" carry 59% of the attention cost (see
*Findings*). And every one of these figures roughly **7×** depending on whether
the laptop is plugged in — so they are only meaningful next to a recorded power
state.

Adjustments made for this hardware, reported as parameters rather than hidden:

- **Listwise re-ranking** — 6 passages per call instead of 1. Pointwise would be
  ~1,350 calls per run. Also a real technique with a real trade-off: the model
  can compare candidates side by side, but scores drift with batch composition.
- **`rerank_pool` 30 → 12**, **`rewrite_n` 3 → 2**.
- **`gemma4:12b` dropped** — too slow to load on this machine.

The disk cache in `src/llm.py` flushes every 50 items, so `Ctrl+C` is safe and a
re-run resumes rather than restarting.

---

## Files

```
src/chunkers.py         fixed / recursive / semantic
src/store.py            vector index — normalised numpy matrix, exact search
src/bm25.py             BM25 Okapi + reciprocal rank fusion
src/retrievers.py       dense / bm25 / hybrid / LLM re-rank / query rewrite
src/metrics.py          recall@k, MRR, NDCG, precision@k, bootstrap CI
src/indexing.py         corpus loading + index building
src/evalset.py          answer-key loading + gold resolution
src/llm.py              Ollama client + disk cache
src/generation.py       answer generation over retrieved context
src/judge.py            LLM faithfulness/correctness + Cohen's kappa

tools/make_evalset.py   LLM drafts questions, human verifies
tools/audit_evalset.py  mechanical defect check on the answer key
tools/pick_chunker.py   point E3-E6 at whichever chunker actually won
tools/run_eval.py       run a config and score retrieval
tools/run_generation.py generate answers over a run, then judge them
tools/judge_agreement.py hand-label a sample, score the judge against it
tools/selftest.py       offline checks on the metrics themselves
tools/report.py         build the ablation table
tools/probe_ollama.py   diagnose endpoint / model issues

configs/E*.json         one file per experiment
```

---

## On negative results

If semantic chunking loses to recursive here, that goes in the table with an
explanation. A study that only reports wins is a sales pitch; one that reports a
loss and accounts for it is evidence an experiment actually ran.

The same applies to the confidence intervals. `report.py` flags its own results
as statistically indistinguishable when intervals overlap — at n ≈ 30 a +0.02
NDCG gain usually is noise, and saying so is the point.
