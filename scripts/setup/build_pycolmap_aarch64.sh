#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="${VENV_DIR:-.venv}"
COLMAP_SRC_DIR="${COLMAP_SRC_DIR:-$REPO_ROOT/third_party/colmap-src}"
COLMAP_REF="${COLMAP_REF:-main}"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[error] venv not found: $VENV_DIR"
  echo "Run: bash scripts/setup/setup_linux_aarch64.sh"
  exit 1
fi

sudo apt-get update
sudo apt-get install -y \
  build-essential git pkg-config ccache \
  cmake ninja-build \
  libeigen3-dev libboost-program-options-dev libboost-graph-dev \
  libboost-system-dev libboost-filesystem-dev \
  libsuitesparse-dev libfreeimage-dev libmetis-dev \
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

source "$REPO_ROOT/$VENV_DIR/bin/activate"
python -m pip install -U pip

# Build/install Python bindings from COLMAP source tree
python -m pip install "$COLMAP_SRC_DIR/python"

python - <<'PY'
import pycolmap
print('[ok] pycolmap version:', pycolmap.__version__)
PY

echo "PyCOLMAP build/install completed."
