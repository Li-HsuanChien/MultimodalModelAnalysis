#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"



pip install --user --upgrade pip --quiet

echo "==> Installing dependencies …"
pip install --user \
    torch \
    transformers \
    deepface \
    tf-keras \
    speechbrain \
    opensmile \
    sentencepiece \
    tqdm
