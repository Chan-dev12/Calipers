"""Probe which endpoints this Ollama build actually serves.

Run:  python tools\\probe_ollama.py
"""
import json
import sys

import requests

HOST = "http://localhost:11434"
_args = [a for a in sys.argv[1:] if not a.startswith("-")]
MODEL = _args[0] if _args else "gemma4:e2b"

print(f"host={HOST}  model={MODEL}\n")

# --- what version / what models ---
for path in ("/api/version", "/api/tags"):
    try:
        r = requests.get(HOST + path, timeout=10)
        body = r.json() if r.status_code == 200 else r.text[:200]
        if path == "/api/tags" and r.status_code == 200:
            body = [m["name"] for m in body.get("models", [])]
        print(f"GET  {path:28} {r.status_code}  {body}")
    except Exception as e:
        print(f"GET  {path:28} ERROR {e}")

print()

# --- candidate generation endpoints ---
#
# OPT-IN, because probing these LOADS THE GENERATION MODEL -- 7.2 GB for
# gemma4:e2b. On a 2 GB card that evicts the embedding model, and any eval run
# in progress starts timing out mid-index. That is not hypothetical: running
# this probe during a semantic-chunking run did exactly that, and the run only
# survived because the embed path splits and retries on failure.
#
# Pass --gen to probe them, and only when nothing else is running.
if "--gen" not in sys.argv:
    print("(skipping generation endpoints -- they load a multi-GB model and will")
    print(" evict the embedding model. Re-run with --gen when nothing else is")
    print(" using Ollama.)")
    CANDIDATES = []
else:
    CANDIDATES = None   # filled in below
_CANDIDATES = [
    ("/api/chat", {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say ok"}],
        "stream": False,
    }),
    ("/v1/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say ok"}],
        "stream": False,
    }),
    ("/api/generate", {
        "model": MODEL,
        "prompt": "Say ok",
        "stream": False,
    }),
]

if CANDIDATES is None:
    CANDIDATES = _CANDIDATES

for path, payload in CANDIDATES:
    try:
        r = requests.post(HOST + path, json=payload, timeout=180)
        snippet = r.text[:220].replace("\n", " ")
        print(f"POST {path:28} {r.status_code}  {snippet}")
    except Exception as e:
        print(f"POST {path:28} ERROR {e}")

print()

# --- embedding endpoints (these already work -- confirming which one) ---
for path, payload in (("/api/embed", {"model": "nomic-embed-text", "input": "hi"}),
                      ("/api/embeddings", {"model": "nomic-embed-text", "prompt": "hi"})):
    try:
        r = requests.post(HOST + path, json=payload, timeout=60)
        print(f"POST {path:28} {r.status_code}")
    except Exception as e:
        print(f"POST {path:28} ERROR {e}")


print()

# --- hardware state: the thing that silently invalidates every timing ---
#
# On this laptop the same embedding call measured 0.10s plugged in and 1.54s on
# battery, because the MX550 drops from 2100 MHz to 300 MHz on battery power.
# It is power policy, not thermal throttling -- the GPU sat at 57C throughout.
# Successive measurements of the same operation ranged over three orders of
# magnitude and supported opposite conclusions about whether batching helped.
import shutil
import subprocess
import time

print("--- hardware ---")
if shutil.which("nvidia-smi"):
    try:
        q = ("clocks.sm,clocks.max.sm,power.draw,temperature.gpu,"
             "memory.used,memory.total,utilization.gpu")
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15).stdout.strip()
        for line in out.splitlines():
            sm, mx, pw, temp, used, total, util = [x.strip() for x in line.split(",")]
            pct = 100.0 * float(sm) / max(float(mx), 1.0)
            print(f"  GPU clock   {sm} / {mx} MHz  ({pct:.0f}% of boost)")
            print(f"  power       {pw} W    temp {temp} C    util {util}%")
            print(f"  memory      {used} / {total} MiB")
            if pct < 60:
                print("  WARNING: the GPU is clocked well below boost. If the machine")
                print("           is on battery, plug it in before timing anything --")
                print("           expect roughly a 7x difference. If it is plugged in")
                print("           and the battery is low, the charger is prioritising")
                print("           charging and the clock will recover as it fills.")
    except (subprocess.SubprocessError, ValueError, OSError) as e:
        print(f"  nvidia-smi failed: {e}")
else:
    print("  no nvidia-smi (CPU-only machine, or drivers not on PATH)")

# Is the embedding model resident, and on which device?
try:
    ps = requests.get(HOST + "/api/ps", timeout=10).json().get("models", [])
    if not ps:
        print("  no model resident (first call will pay a cold load)")
    for m in ps:
        vram, total = m.get("size_vram", 0), m.get("size", 1)
        where = "GPU" if vram >= total * 0.9 else ("CPU" if vram == 0 else "split GPU/CPU")
        print(f"  resident: {m['name']} on {where} "
              f"({vram // 1048576}/{total // 1048576} MiB in VRAM)")
except (requests.RequestException, ValueError) as e:
    print(f"  /api/ps failed: {e}")

# Measured throughput, so the numbers above have a consequence attached.
print()
print("--- embedding throughput (best of 3, ~1000 chars) ---")
probe = "the quick brown fox jumps over the lazy dog. " * 23
try:
    times = []
    for _ in range(3):
        t0 = time.time()
        requests.post(HOST + "/api/embed",
                      json={"model": "nomic-embed-text", "input": probe,
                            "keep_alive": "5m"}, timeout=300)
        times.append(time.time() - t0)
    best = min(times)
    print(f"  {len(probe)} chars in {best:.2f}s = {len(probe) / best:.0f} chars/sec")
    if len(probe) / best < 500:
        print("  That is slow. Check the GPU clock above and make sure no other")
        print("  eval process is holding the server -- a stray run makes reads")
        print("  10-100x slower and looks identical to a performance bug.")
except requests.RequestException as e:
    print(f"  failed: {e}")
