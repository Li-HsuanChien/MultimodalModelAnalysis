#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/storage/home/lpc5553/work/MultimodalModelAnalysis"
PKG_DIR="$PROJECT_DIR/.pkg"
PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

echo "==> Python ${PYVER} found at $(which python3)"

# ── 1. Bootstrap pip if missing ───────────────────────────────────────────────
if ! python3 -m pip --version &>/dev/null; then
    echo "==> pip not found — bootstrapping with ensurepip …"
    if python3 -m ensurepip --user 2>/dev/null; then
        echo "    ensurepip succeeded."
    else
        echo "    ensurepip failed — downloading get-pip.py …"
        curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
        python3 /tmp/get-pip.py --user
        rm /tmp/get-pip.py
        echo "    get-pip.py succeeded."
    fi
else
    echo "==> pip already available: $(python3 -m pip --version)"
fi

# ── 2. Point installs to project dir ─────────────────────────────────────────
export PYTHONUSERBASE="$PKG_DIR"
export PATH="$PKG_DIR/bin:$PATH"
export PYTHONPATH="$PKG_DIR/lib/python${PYVER}/site-packages${PYTHONPATH:+:$PYTHONPATH}"

echo "==> Installing to $PKG_DIR …"

# ── 3. Install dependencies ───────────────────────────────────────────────────
python3 -m pip install --user --upgrade pip --quiet

python3 -m pip install --user torch transformers
python3 -m pip install --user deepface tf-keras
python3 -m pip install --user speechbrain opensmile
python3 -m pip install --user sentencepiece
python3 -m pip install --user tqdm

# ── 4. Done ───────────────────────────────────────────────────────────────────
echo ""
echo "✓ Setup complete. Add these to your job scripts:"
echo ""
echo "    export PYTHONUSERBASE=$PKG_DIR"
echo "    export PATH=$PKG_DIR/bin:\$PATH"
echo "    export PYTHONPATH=$PKG_DIR/lib/python${PYVER}/site-packages\${PYTHONPATH:+:\$PYTHONPATH}"
echo ""