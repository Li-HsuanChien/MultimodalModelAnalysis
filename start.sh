#!/usr/bin/env bash
set -euo pipefail

# ── Per-project package isolation using PYTHONUSERBASE ────────────────────────
# This satisfies the cluster requirement of --user while keeping packages
# out of your home directory and isolated to this project.

PROJECT_DIR="/storage/home/lpc5553/work/MultimodalModelAnalysis"
PKG_DIR="$PROJECT_DIR/.pkg"

echo "==> Configuring per-project package directory at $PKG_DIR …"
export PYTHONUSERBASE="$PKG_DIR"
export PATH="$PKG_DIR/bin:$PATH"

PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
export PYTHONPATH="$PKG_DIR/lib/python${PYVER}/site-packages${PYTHONPATH:+:$PYTHONPATH}"

echo "    PYTHONUSERBASE = $PYTHONUSERBASE"
echo "    PATH           = $PATH"
echo "    PYTHONPATH     = $PYTHONPATH"

echo "==> Upgrading pip …"
pip install --user --upgrade pip --quiet

echo "==> Installing dependencies …"

# Deep learning / model inference
pip install --user torch transformers

# Facial expression
pip install --user deepface tf-keras

# Vocal tone
pip install --user speechbrain opensmile

# Text / stance
pip install --user sentencepiece

# Database & utilities
pip install --user tqdm

echo ""
echo "✓ Setup complete."
echo ""
echo "To activate these packages in a new shell, run:"
echo ""
echo "    export PYTHONUSERBASE=$PKG_DIR"
echo "    export PATH=$PKG_DIR/bin:\$PATH"
echo "    export PYTHONPATH=$PKG_DIR/lib/python${PYVER}/site-packages\${PYTHONPATH:+:\$PYTHONPATH}"
echo ""
echo "Or add the above two lines to your ~/.bashrc or job submission script."