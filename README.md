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

Verify:
```bash
python -c "from src import llm; print(llm.health())"
```

---

## Corpus

47 Java source files from **Apache Commons Lang**
(`src/main/java/org/apache/commons/lang3/`), ~1.9 M characters, ~2,650 chunks.

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
                                               ▼
              retrievers.py ──> metrics.py ──> results/*.json ──> report.py
```

---

## Usage

```bash
# build the answer key
python tools/make_evalset.py draft --n 40     # LLM drafts candidates
python tools/make_evalset.py review           # human verifies, one at a time
python tools/audit_evalset.py                 # mechanical defect check
python tools/audit_evalset.py --prune         # drop the critical ones

# experiments
python tools/run_eval.py E0_baseline E1_recursive E3_bm25 E4_hybrid   # seconds, no LLM
python tools/run_eval.py E2_semantic E5_rerank E6_query_rewrite       # slow, run overnight
python tools/report.py --save
```

---

## Experiment matrix

| ID | Chunker | Retriever | Re-rank | Query | Question it answers |
|----|---------|-----------|---------|-------|---------------------|
| E0 | fixed | dense | — | raw | Control |
| E1 | recursive | dense | — | raw | Does structure-aware splitting help? |
| E2 | semantic | dense | — | raw | Is the embedding cost worth it? |
| E3 | best of E0–E2 | bm25 | — | raw | How far does keyword-only get? |
| E4 | best | hybrid + RRF | — | raw | Do they fail differently enough to fuse? |
| E5 | best | hybrid + RRF | LLM | raw | Does re-ranking fix ordering? |
| E6 | best | hybrid + RRF | LLM | rewritten | Does query rewriting add anything on top? |

One variable per row. That is the whole discipline — it is what lets you say
"recursive chunking gave +9 points" instead of "I changed some things and it got
better."

---

## Results

*(populated by `tools/report.py --save` → `results/ABLATION.md`)*

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
An LLM scores faithfulness at scale; 25 of the same examples are then hand-
labelled and the agreement rate reported. The difference between "scores 0.88 on
faithfulness" and "scores 0.88 on a metric shown to be trustworthy."

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

| Experiment | LLM calls | Runtime |
|---|---|---|
| E0 fixed + dense | none | seconds |
| E1 recursive + dense | none | seconds |
| E2 semantic + dense | embeddings only | 2–5 min once, then cached |
| E3 BM25 | none at all | instant, pure Python |
| E4 hybrid + RRF | none | seconds |
| E5 + re-ranking | ~8 per question | 30–45 min |
| E6 + query rewriting | ~10 per question | 30–45 min |

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

tools/make_evalset.py   LLM drafts questions, human verifies
tools/audit_evalset.py  mechanical defect check on the answer key
tools/run_eval.py       run a config and score it
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
