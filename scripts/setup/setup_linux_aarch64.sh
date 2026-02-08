#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PY_BIN="${PY_BIN:-python3.9}"
VENV_DIR="${VENV_DIR:-.venv}"
INSTALL_SYSTEM_PACKAGES="${INSTALL_SYSTEM_PACKAGES:-1}"

if [[ "$INSTALL_SYSTEM_PACKAGES" == "1" ]]; then
  sudo apt-get update
  sudo apt-get install -y \
    build-essential git git-lfs curl wget unzip pkg-config ccache \
    cmake ninja-build \
    python3.9 python3.9-venv python3.9-dev python3-pip \
    ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libeigen3-dev libboost-program-options-dev libboost-graph-dev \
    libboost-system-dev libboost-filesystem-dev \
    libsuitesparse-dev libfreeimage-dev libmetis-dev \
    libgoogle-glog-dev libgflags-dev libglew-dev \
    qtbase5-dev libqt5opengl5-dev libcgal-dev libceres-dev \
    libsqlite3-dev
fi

"$PY_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install -U pip setuptools wheel
python -m pip install -e . --no-deps

# aarch64 cp39 wheels verified for these versions
python -m pip install "torch==2.5.1" "torchvision==0.20.1"

# unavailable on linux aarch64 cp39: xformers, pycolmap
python -m pip install $(grep -Ev '^(xformers|pycolmap)$' requirements.txt | tr '\n' ' ')
python -m pip install --force-reinstall "moviepy==1.0.3" "numpy<2"
python -m pip install pytest httpx

python - <<'PY'
import torch, fastapi, typer, moviepy.editor
print('[ok] torch:', torch.__version__)
print('[ok] cuda_available:', torch.cuda.is_available())
print('[ok] fastapi:', fastapi.__version__)
print('[ok] typer:', typer.__version__)
PY

echo
if python - <<'PY'
import pycolmap  # noqa: F401
print('present')
PY
then
  echo "[ok] pycolmap already installed"
else
  echo "[note] pycolmap not installed (expected on linux aarch64 cp39 wheels)"
  echo "[note] if you need COLMAP export, run: bash scripts/setup/build_pycolmap_aarch64.sh"
fi

echo
cat <<'TXT'
Setup complete.

Run backend:
  source .venv/bin/activate
  export KMP_DUPLICATE_LIB_OK=TRUE
  da3 backend --model-dir depth-anything/DA3NESTED-GIANT-LARGE-1.1 --device cuda --host 0.0.0.0 --port 8008
TXT
