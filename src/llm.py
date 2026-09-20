"""
Thin Ollama client: embeddings + generation, with an on-disk cache.

The cache matters more than you think. You will re-run the eval set dozens of
times while tuning. Without caching, every run re-embeds the whole corpus and
you will stop running experiments because they are slow -- which defeats the
entire point of the project.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from typing import List

import numpy as np
import requests

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Model choice resolves in three steps: environment variable, then
# configs/models.json, then the hardcoded default below.
#
# configs/models.json existed for weeks and NOTHING READ IT. It named
# "gemma4:e4b" as the judge -- a tag that is not even pulled on this machine --
# so anyone trusting the file would have believed the runs used a model they did
# not have, while every run quietly used the hardcoded defaults instead. A
# config file that silently does nothing is worse than no config file: it is a
# confident, checked-in, wrong answer to "which model produced these numbers?"
_MODELS_JSON = os.path.join(os.path.dirname(__file__), "..", "configs", "models.json")


def _model_defaults() -> dict:
    try:
        with open(_MODELS_JSON, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


_DEFAULTS = _model_defaults()


def _pick(env_var: str, key: str, fallback: str) -> str:
    return os.environ.get(env_var) or _DEFAULTS.get(key) or fallback


EMBED_MODEL = _pick("EMBED_MODEL", "embed_model", "nomic-embed-text")

# gemma3:1b (815MB) is the dev default because on 2GB VRAM the iteration loop
# matters more than raw quality -- you will run the eval set many times.
# Switch to the bigger model for the final run:
#     $env:GEN_MODEL = "gemma4:e2b"
GEN_MODEL = _pick("GEN_MODEL", "gen_model", "gemma3:1b")

# The judge should be at least as strong as the generator, never weaker --
# a judge that cannot follow the task grades noise.
JUDGE_MODEL = _pick("JUDGE_MODEL", "judge_model", "gemma4:e2b")

GEN_TIMEOUT = int(os.environ.get("GEN_TIMEOUT", "900"))
KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")   # hold weights in RAM
NUM_CTX = int(os.environ.get("NUM_CTX", "4096"))          # KV cache is preallocated
EMBED_TIMEOUT = int(os.environ.get("EMBED_TIMEOUT", "600"))  # model reload can be slow

# Gemma 4 has thinking mode ON by default. For this project every call is
# structured extraction -- score a passage, emit a JSON object -- and a chain of
# reasoning before the answer is pure cost: more tokens generated on CPU, more
# memory held, slower runs, and a longer response to parse. Turn it off.
# Set THINK=1 if you ever want to compare judge quality with reasoning enabled;
# that would be a legitimate extra ablation row.
THINK = os.environ.get("THINK", "0") not in ("0", "false", "False", "")

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".cache")
os.makedirs(_CACHE_DIR, exist_ok=True)


class _DiskCache:
    def __init__(self, name: str):
        self.path = os.path.join(_CACHE_DIR, f"{name}.pkl")
        self.d = {}
        if os.path.exists(self.path):
            try:
                with open(self.path, "rb") as f:
                    self.d = pickle.load(f)
            except Exception:
                self.d = {}
        self._dirty = 0

    def get(self, k):
        return self.d.get(k)

    def put(self, k, v):
        self.d[k] = v
        self._dirty += 1
        if self._dirty >= 50:
            self.flush()

    def flush(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.d, f)
        self._dirty = 0


_embed_cache = _DiskCache("embeddings")
_gen_cache = _DiskCache("generations")


def _key(*parts) -> str:
    return hashlib.md5("||".join(map(str, parts)).encode()).hexdigest()


# Ollama's /api/embed accepts a LIST as `input` and embeds it in one pass.
#
# MEASURED, ON THIS CORPUS: 1.36x faster than one call at a time
# (2454 vs 1799 chars/sec, arms interleaved on a quiet machine). Worth keeping,
# but nothing like the number the first benchmark here claimed.
#
# That first benchmark said 14x, and it was wrong in a way worth recording.
# It timed 55-character synthetic strings, where per-request overhead is most
# of the cost and batching therefore amortises almost everything. Real corpus
# sentences average ~149 characters and run to 3,258, and at that size the
# embedding model is compute-bound: the work per character is the same however
# you group it. Benchmarking on convenient fake data instead of the real
# distribution overstated the win by an order of magnitude -- the same mistake,
# in the same project, that the eval set is built to prevent.
#
# A second trap: every throughput number here is garbage if anything else is
# talking to Ollama. Measurements taken while a stray eval process was still
# running read 13-230 chars/sec, a 10-100x understatement, and they were noisy
# enough to point at the opposite conclusion.
#
# A third trap, and the one that actually wasted the most time: ON A LAPTOP,
# CHECK THE POWER STATE BEFORE BELIEVING ANY BENCHMARK. The same embedding call
# measured 0.10s plugged in and 1.54s on battery. The MX550 drops to 300 MHz of
# its 2100 MHz boost clock on battery -- a 7x derate at 5.3W, with the GPU at
# 57C, so it is power policy and not thermal throttling. Nothing in the server
# or the client reports this; it looks exactly like the code got slower.
#
#     nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw --format=csv
#
# Kill competing processes AND confirm the clock before timing anything.
#
# BATCHES ARE CAPPED BY CHARACTERS, NOT JUST BY COUNT.
# A count-only cap looks fine on average text and then falls over: a batch of
# 32 of this corpus's longest sentences (67k chars) never returned inside a 90s
# timeout, while 32 average sentences take about a second. Server cost tracks
# total tokens in the batch, so that is what has to be bounded.
EMBED_BATCH = int(os.environ.get("EMBED_BATCH", "32"))
EMBED_BATCH_CHARS = int(os.environ.get("EMBED_BATCH_CHARS", "8000"))

# A multi-item batch gets a SHORT timeout, unlike a single embed.
# EMBED_TIMEOUT is 600s because a cold model reload legitimately takes minutes --
# but that patience is wrong for a batch: a batch that has not returned in two
# minutes is too big, and the useful response is to split it, not to keep
# waiting. Using the long timeout here meant one bad batch stalled the run for
# ten minutes before making any progress at all.
EMBED_BATCH_TIMEOUT = int(os.environ.get("EMBED_BATCH_TIMEOUT", "120"))


def _plan_batches(indices: List[int], texts: List[str]) -> List[List[int]]:
    """Group indices into batches bounded by BOTH item count and total chars.

    A single text larger than the char budget is sent on its own rather than
    skipped -- it is still a legitimate chunk and must get a real vector.
    """
    batches, cur, cur_chars = [], [], 0
    for i in indices:
        n = len(texts[i])
        if cur and (len(cur) >= EMBED_BATCH or cur_chars + n > EMBED_BATCH_CHARS):
            batches.append(cur)
            cur, cur_chars = [], 0
        cur.append(i)
        cur_chars += n
    if cur:
        batches.append(cur)
    return batches


def embed(texts: List[str], model: str = None, batch_log: bool = False) -> np.ndarray:
    """Returns (n, dim) float32. Cached per (model, text).

    Only cache misses are sent to the server, and they go in batches. The cache
    stays keyed per individual text, so a batched run and a sequential run
    populate exactly the same cache entries and are interchangeable -- a cache
    built before batching existed stays valid.
    """
    model = model or EMBED_MODEL
    out, missing = [None] * len(texts), []
    for i, t in enumerate(texts):
        hit = _embed_cache.get(_key(model, t))
        if hit is not None:
            out[i] = hit
        else:
            missing.append(i)

    done = 0
    for group in _plan_batches(missing, texts):
        if batch_log:
            print(f"  embedding {done}/{len(missing)}...", flush=True)
        vecs = _embed_batch([texts[i] for i in group], model)
        for i, vec in zip(group, vecs):
            _embed_cache.put(_key(model, texts[i]), vec)
            out[i] = vec
        done += len(group)

    _embed_cache.flush()
    return np.array(out, dtype=np.float32)


def _embed_batch(texts: List[str], model: str) -> List[np.ndarray]:
    """Embed a list in one request; on failure SPLIT rather than retry.

    Splitting immediately is the whole point. A multi-item batch that fails has
    almost always failed for a reason that is about its size -- too many tokens
    to process inside the timeout, or an allocation that did not fit -- and that
    is deterministic. Retrying the identical oversized batch four times just
    pays the timeout four times before making progress; an early version of this
    did exactly that and wedged a run for fifteen minutes on its first batch.
    Halving converges in log2(n) steps instead.

    Transient failures are still retried, but at the level where retrying is
    actually the right answer: batch size 1, inside _embed_one(), which backs
    off and then RAISES. Nothing here invents a vector or drops a text -- a
    missing embedding must stop the run, because a silently dropped chunk is a
    corpus that no longer matches the one the answer key was built against.
    """
    if not texts:
        return []
    if len(texts) == 1:
        return [_embed_one(texts[0], model)]

    try:
        r = requests.post(f"{OLLAMA}/api/embed",
                          json={"model": model, "input": texts,
                                "keep_alive": KEEP_ALIVE},
                          timeout=EMBED_BATCH_TIMEOUT)
        if r.status_code == 200:
            vecs = r.json().get("embeddings")
            # A short list would silently misalign texts and vectors, so treat a
            # count mismatch as a hard failure rather than zipping whatever came
            # back against the wrong inputs.
            if vecs and len(vecs) == len(texts):
                return [np.array(v, dtype=np.float32) for v in vecs]
            raise ValueError(f"expected {len(texts)} embeddings, "
                             f"got {len(vecs) if vecs else 0}")
        r.raise_for_status()
        raise ValueError(f"unexpected status {r.status_code}")

    except (requests.RequestException, KeyError, ValueError) as e:
        mid = len(texts) // 2
        print(f"    embed batch({len(texts)}, {sum(map(len, texts))} chars) "
              f"failed ({type(e).__name__}); splitting {mid}+{len(texts) - mid}",
              flush=True)
        return (_embed_batch(texts[:mid], model)
                + _embed_batch(texts[mid:], model))


def _embed_one(text: str, model: str, attempts: int = 4) -> np.ndarray:
    """One embedding, with retry.

    Without retry a single transient timeout aborts an entire indexing run.
    On low-RAM hardware that happens for a mundane reason: Ollama evicts the
    embedding model to make room for a generation model, and the next call has
    to reload it from disk -- which can exceed a short timeout. Retrying costs
    nothing and the cache means no repeated work.

    keep_alive is passed here too, so the embedding model stays resident for the
    whole indexing pass instead of being swapped in and out per call.
    """
    import time

    last = None
    for attempt in range(attempts):
        try:
            # newer Ollama exposes /api/embed; older only /api/embeddings
            r = requests.post(f"{OLLAMA}/api/embed",
                              json={"model": model, "input": text,
                                    "keep_alive": KEEP_ALIVE},
                              timeout=EMBED_TIMEOUT)
            if r.status_code == 200:
                data = r.json()
                if "embeddings" in data:
                    return np.array(data["embeddings"][0], dtype=np.float32)
                if "embedding" in data:
                    return np.array(data["embedding"], dtype=np.float32)

            r = requests.post(f"{OLLAMA}/api/embeddings",
                              json={"model": model, "prompt": text,
                                    "keep_alive": KEEP_ALIVE},
                              timeout=EMBED_TIMEOUT)
            r.raise_for_status()
            return np.array(r.json()["embedding"], dtype=np.float32)

        except (requests.RequestException, KeyError, ValueError) as e:
            last = e
            wait = 3 * (attempt + 1)
            print(f"    embed retry {attempt + 1}/{attempts} in {wait}s "
                  f"({type(e).__name__})", flush=True)
            time.sleep(wait)

    raise RuntimeError(
        f"Embedding failed after {attempts} attempts: {last}\n"
        f"Check `ollama ps` -- if a generation model is resident it may be "
        f"evicting the embedding model. Free it with `ollama stop <model>`.")


def generate(prompt: str, model: str = None, temperature: float = 0.0,
             system: str = None) -> str:
    """temperature=0 by default. Non-deterministic generation makes your
    ablation table meaningless -- you would not know if a delta came from your
    change or from sampling noise."""
    model = model or GEN_MODEL
    k = _key(model, prompt, temperature, system)
    hit = _gen_cache.get(k)
    if hit is not None:
        return hit

    text = _chat(prompt, model, temperature, system)
    _gen_cache.put(k, text)
    _gen_cache.flush()
    return text


def _chat(prompt: str, model: str, temperature: float, system: str | None) -> str:
    """POST /api/chat. This is the endpoint on Ollama 0.2+; /api/generate was
    removed in later builds.

    NO SILENT FALLBACK. An earlier version caught timeouts and retried against
    /api/generate, which turned "the model is still loading" into a confusing
    404 from a different endpoint. Masking one failure with another failure is
    worse than failing loudly -- especially in a project whose whole output is
    numbers you have to trust.

    num_ctx is set explicitly: Ollama defaults Gemma 4 to a 4K window despite
    its 128K capacity. At the default, long re-ranking batches get silently
    truncated -- passages vanish with no error and your metrics quietly become
    wrong. That is the most dangerous kind of bug in an evaluation harness.

    keep_alive holds the model in memory between calls. With 7.2GB of weights
    and 2GB of VRAM, a cold load costs 1-3 minutes; without this you would pay
    it repeatedly across a single eval run.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    r = requests.post(
        f"{OLLAMA}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "think": THINK,
            "keep_alive": KEEP_ALIVE,
            "options": {"temperature": temperature, "num_ctx": NUM_CTX},
        },
        timeout=GEN_TIMEOUT,
    )
    if r.status_code == 404:
        raise RuntimeError(
            f"/api/chat returned 404. Check the model tag exists: `ollama list`.\n"
            f"Requested model: {model!r}")
    if r.status_code >= 400:
        # Surface Ollama's own message. A bare status code sends you hunting for
        # the wrong bug -- the server almost always says exactly what is wrong,
        # and on this hardware it is usually memory.
        try:
            detail = r.json().get("error", r.text)
        except ValueError:
            detail = r.text
        detail = str(detail)[:400]
        hint = ""
        low = detail.lower()
        if "memory" in low or "alloc" in low or "system" in low:
            hint = ("\nHINT: out of memory. Free other models with `ollama ps` "
                    "then `ollama stop <model>`, and/or lower the context:\n"
                    '      $env:NUM_CTX = "4096"')
        raise RuntimeError(
            f"Ollama {r.status_code} from /api/chat\n"
            f"  model={model!r}  num_ctx={NUM_CTX}\n"
            f"  error: {detail}{hint}")
    r.raise_for_status()
    return (r.json().get("message", {}).get("content") or "").strip()


def warmup(model: str = None, verbose: bool = True) -> float:
    """Load a model into memory and report how long it took.

    Call this once before a long run. On low-VRAM hardware the first call to a
    large model can take minutes; without a warmup that cost lands inside your
    first eval question and looks like a hang.
    """
    import time
    model = model or GEN_MODEL
    t0 = time.time()
    if verbose:
        print(f"warming up {model} (first load can take 1-3 min)...", flush=True)
    requests.post(f"{OLLAMA}/api/chat",
                  json={"model": model,
                        "messages": [{"role": "user", "content": "ok"}],
                        "stream": False, "keep_alive": KEEP_ALIVE},
                  timeout=GEN_TIMEOUT)
    dt = time.time() - t0
    if verbose:
        print(f"  loaded in {dt:.1f}s", flush=True)
    return dt


def generate_json(prompt: str, model: str = None, system: str = None,
                  retries: int = 2) -> dict:
    """Ask for JSON, strip code fences, parse. Retries on malformed output."""
    for attempt in range(retries + 1):
        raw = generate(prompt + ("" if attempt == 0 else "\n\nReturn ONLY valid JSON."),
                       model=model, temperature=0.0, system=system)
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                continue
    return {}


def installed_models() -> set:
    """Tags Ollama actually has locally."""
    try:
        r = requests.get(f"{OLLAMA}/api/tags", timeout=5)
        r.raise_for_status()
        return {m["name"] for m in r.json().get("models", [])}
    except (requests.RequestException, KeyError, ValueError):
        return set()


def require_models(*models: str):
    """Fail before a long run if a model is not pulled, not two hours into it.

    Ollama reports a missing tag as a 404 on the first call that needs it. For
    the re-ranking experiments that first call lands after the corpus has been
    indexed, so a typo in a config costs the whole indexing pass before it
    surfaces. Checking up front turns that into an instant, readable error.
    """
    have = installed_models()
    if not have:
        return                                   # server unreachable; health() reports it
    # Ollama lists an untagged pull as "name:latest", so a bare name must match
    # that. A name that CARRIES a tag has to match exactly: an earlier version
    # compared only the part before the colon, which meant "gemma4:e4b" was
    # happily accepted because "gemma4:e2b" was installed -- the check passed
    # while the run would still 404, which is worse than having no check.
    def present(m: str) -> bool:
        return m in have or (":" not in m and f"{m}:latest" in have)

    missing = [m for m in models if m and not present(m)]
    if missing:
        raise RuntimeError(
            f"model(s) not installed: {', '.join(missing)}\n"
            f"  available: {', '.join(sorted(have))}\n"
            f"  pull with: ollama pull {missing[0]}")


def health() -> bool:
    try:
        return requests.get(f"{OLLAMA}/api/tags", timeout=5).status_code == 200
    except requests.RequestException:
        return False
