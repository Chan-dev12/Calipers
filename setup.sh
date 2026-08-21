#!/usr/bin/env bash
# Calipers -- one-time setup. Run from the project root.
set -e
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "--- pulling models (a few GB, one time) ---"
ollama pull nomic-embed-text
ollama pull gemma4:e2b
echo
python -c "from src import llm; print('ollama:', 'OK' if llm.health() else 'NOT RUNNING -- start it with: ollama serve')"
echo "Setup done. Next: put documents in corpus/, then"
echo "  python tools/make_evalset.py draft --n 60"
