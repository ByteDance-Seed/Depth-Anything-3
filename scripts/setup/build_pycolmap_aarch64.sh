#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="${VENV_DIR:-.venv}"
COLMAP_SRC_DIR="${COLMAP_SRC_DIR:-$REPO_ROOT/third_party/colmap-src}"
COLMAP_REF="${COLMAP_REF:-}"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[error] venv not found: $VENV_DIR"
  echo "Run: bash scripts/setup/setup_linux_aarch64.sh"
  exit 1
fi

source "$REPO_ROOT/$VENV_DIR/bin/activate"
read -r PY_MAJOR PY_MINOR <<EOF
$(python - <<'PY'
import sys
print(sys.version_info.major, sys.version_info.minor)
PY
)
EOF

if [[ -z "$COLMAP_REF" ]]; then
  if [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 10 ]]; then
    COLMAP_REF="3.13.0"
    echo "[warn] Python ${PY_MAJOR}.${PY_MINOR} detected; defaulting COLMAP_REF=${COLMAP_REF} (main requires >=3.10 for pycolmap)."
  else
    COLMAP_REF="main"
  fi
fi
echo "[info] Using COLMAP_REF=${COLMAP_REF}"

sudo apt-get update
sudo apt-get install -y \
  build-essential git pkg-config ccache \
  cmake ninja-build \
  libeigen3-dev libboost-program-options-dev libboost-graph-dev \
  libboost-system-dev libboost-filesystem-dev \
  libsuitesparse-dev libfreeimage-dev libmetis-dev libopenimageio-dev openimageio-tools \
  libopencv-dev \
  libgoogle-glog-dev libgflags-dev libglew-dev \
  qtbase5-dev libqt5opengl5-dev libcgal-dev libceres-dev

mkdir -p "$(dirname "$COLMAP_SRC_DIR")"
if [[ ! -d "$COLMAP_SRC_DIR/.git" ]]; then
  git clone --recursive https://github.com/colmap/colmap.git "$COLMAP_SRC_DIR"
fi

cd "$COLMAP_SRC_DIR"
git fetch --all --tags
git checkout "$COLMAP_REF"
git submodule update --init --recursive

cmake -S . -B build -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DGUI_ENABLED=OFF

cmake --build build -j"$(nproc)"
sudo cmake --install build

python -m pip install -U pip

# Build/install Python bindings from COLMAP source tree.
# COLMAP changed package layout across versions, so detect install root.
PYCOLMAP_PKG_PATH=""
if [[ -f "$COLMAP_SRC_DIR/pyproject.toml" ]]; then
  PYCOLMAP_PKG_PATH="$COLMAP_SRC_DIR"
elif [[ -f "$COLMAP_SRC_DIR/python/pyproject.toml" || -f "$COLMAP_SRC_DIR/python/setup.py" ]]; then
  PYCOLMAP_PKG_PATH="$COLMAP_SRC_DIR/python"
elif [[ -f "$COLMAP_SRC_DIR/pycolmap/pyproject.toml" || -f "$COLMAP_SRC_DIR/pycolmap/setup.py" ]]; then
  PYCOLMAP_PKG_PATH="$COLMAP_SRC_DIR/pycolmap"
else
  echo "[error] Could not find installable pycolmap package in $COLMAP_SRC_DIR"
  exit 1
fi
echo "[info] Installing pycolmap from: $PYCOLMAP_PKG_PATH"
python -m pip install "$PYCOLMAP_PKG_PATH"

python - <<'PY'
import pycolmap
print('[ok] pycolmap version:', pycolmap.__version__)
PY

echo "PyCOLMAP build/install completed."
