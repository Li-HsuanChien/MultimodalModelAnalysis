#!/usr/bin/env bash
set -euo pipefail

# ── Per-project package isolation using PYTHONUSERBASE ────────────────────────
# Satisfies cluster --user requirement while isolating packages per project.

PROJECT_DIR="/storage/home/lpc5553/work/MultimodalModelAnalysis"
PKG_DIR="$PROJECT_DIR/.pkg"

# ── 1. Check Python version ───────────────────────────────────────────────────
echo "==> Checking Python version …"
PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYVER_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PYVER_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")

if [ "$PYVER_MAJOR" -lt 3 ] || [ "$PYVER_MINOR" -lt 8 ]; then
    echo ""
    echo "ERROR: Python 3.8+ is required, but found Python ${PYVER}."
    echo "       Load a newer version first, for example:"
    echo ""
    echo "           module avail python        # see what is available"
    echo "           module load python/3.11    # load a supported version"
    echo ""
    exit 1
fi
echo "    Python ${PYVER} — OK"

# ── 2. Configure per-project package directory ────────────────────────────────
echo "==> Configuring package directory at $PKG_DIR …"
export PYTHONUSERBASE="$PKG_DIR"
export PATH="$PKG_DIR/bin:$PATH"
export PYTHONPATH="$PKG_DIR/lib/python${PYVER}/site-packages${PYTHONPATH:+:$PYTHONPATH}"

echo "    PYTHONUSERBASE = $PYTHONUSERBASE"
echo "    PATH           = $PATH"
echo "    PYTHONPATH     = $PYTHONPATH"

# ── 3. Upgrade pip ────────────────────────────────────────────────────────────
echo "==> Upgrading pip …"
python3 -m pip install --user --upgrade pip --quiet

# ── 4. Install dependencies ───────────────────────────────────────────────────
echo "==> Installing dependencies …"

# Deep learning / model inference
python3 -m pip install --user torch transformers

# Facial expression
python3 -m pip install --user deepface tf-keras

# Vocal tone
python3 -m pip install --user speechbrain opensmile

# Text / stance
python3 -m pip install --user sentencepiece

# Database & utilities
python3 -m pip install --user tqdm

# ── 5. Done ───────────────────────────────────────────────────────────────────
echo ""
echo "✓ Setup complete."
echo ""
echo "To activate these packages in a new shell or job script, add:"
echo ""
echo "    export PYTHONUSERBASE=$PKG_DIR"
echo "    export PATH=$PKG_DIR/bin:\$PATH"
echo "    export PYTHONPATH=$PKG_DIR/lib/python${PYVER}/site-packages\${PYTHONPATH:+:\$PYTHONPATH}"
echo ""