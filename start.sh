#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"

echo "==> Checking for Python virtual environment …"
if [ ! -d "$VENV_DIR" ]; then
    echo "    Not found. Creating venv at $VENV_DIR …"
    python3 -m venv "$VENV_DIR"
    echo "    Venv created."
else
    echo "    Found existing venv at $VENV_DIR."
fi

echo "==> Activating venv …"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip …"
pip install  --user --upgrade pip --quiet

echo "==> Installing dependencies …"
pip install --user \
    # ── Deep learning / model inference ──────────────────────────────────
    torch \
    transformers \
    # ── Facial expression ─────────────────────────────────────────────────
    deepface \
    tf-keras \
    # ── Vocal tone ────────────────────────────────────────────────────────
    speechbrain \
    opensmile \
    # ── Text / stance ─────────────────────────────────────────────────────
    sentencepiece \
    # ── Database & utilities ──────────────────────────────────────────────
    tqdm

echo ""
echo "✓ Setup complete. To activate the venv in your shell, run:"
echo "      source $VENV_DIR/bin/activate"