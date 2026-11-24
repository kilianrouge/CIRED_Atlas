"""Install a complete local copy of the embedding model into ATLAS/models/.

ATLAS loads the model from this folder with the Hugging Face Hub switched off, so a Hub
outage or API change can no longer silently push ranking onto the TF-IDF fallback.

Usage (from the ATLAS folder, with the app's Python):
    python tools/download_model.py
"""
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
REPO = "mixedbread-ai/mxbai-embed-large-v1"
TARGET = ROOT / "models" / "mxbai-embed-large-v1"

os.environ.pop("HF_HUB_OFFLINE", None)
snapshot_download(REPO, local_dir=str(TARGET),
                  allow_patterns=["*.json", "*.txt", "model.safetensors", "1_Pooling/*"])
shutil.rmtree(TARGET / ".cache", ignore_errors=True)
for p in [TARGET, *TARGET.rglob("*")]:  # shared install: readable by every user
    p.chmod(p.stat().st_mode | (0o755 if p.is_dir() else 0o644))
missing = [f for f in ("model.safetensors", "modules.json", "tokenizer.json",
                       "1_Pooling/config.json") if not (TARGET / f).exists()]
if missing:
    raise SystemExit(f"Download incomplete, missing: {missing}")
print(f"Model installed in {TARGET}")
