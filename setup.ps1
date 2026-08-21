# Calipers -- one-time setup (Windows PowerShell). Run from the project root.
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Write-Host "--- pulling models (a few GB, one time) ---"
ollama pull nomic-embed-text
ollama pull gemma4:e2b
python -c "from src import llm; print('ollama:', 'OK' if llm.health() else 'NOT RUNNING')"
Write-Host "Setup done. Put documents in corpus\, then run:"
Write-Host "  python tools\make_evalset.py draft --n 60"
