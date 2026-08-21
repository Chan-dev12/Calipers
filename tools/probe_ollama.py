"""Probe which endpoints this Ollama build actually serves.

Run:  python tools\\probe_ollama.py
"""
import json
import sys

import requests

HOST = "http://localhost:11434"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "gemma4:e2b"

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
CANDIDATES = [
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
