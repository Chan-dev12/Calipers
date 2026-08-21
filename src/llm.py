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
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

# gemma3:1b (815MB) is the default because on 2GB VRAM the iteration loop
# matters more than raw quality -- you will run the eval set many times.
# Switch to the bigger model for the final run:
#     $env:GEN_MODEL = "gemma4:e2b"
GEN_MODEL = os.environ.get("GEN_MODEL", "gemma3:1b")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemma4:e2b")

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


def embed(texts: List[str], model: str = None, batch_log: bool = False) -> np.ndarray:
    """Returns (n, dim) float32. Cached per (model, text)."""
    model = model or EMBED_MODEL
    out, missing = [None] * len(texts), []
    for i, t in enumerate(texts):
        hit = _embed_cache.get(_key(model, t))
        if hit is not None:
            out[i] = hit
        else:
            missing.append(i)

    for n, i in enumerate(missing):
        if batch_log and n % 50 == 0:
            print(f"  embedding {n}/{len(missing)}...", flush=True)
        vec = _embed_one(texts[i], model)
        _embed_cache.put(_key(model, texts[i]), vec)
        out[i] = vec
    _embed_cache.flush()
    return np.array(out, dtype=np.float32)


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


def health() -> bool:
    try:
        return requests.get(f"{OLLAMA}/api/tags", timeout=5).status_code == 200
    except requests.RequestException:
        return False
