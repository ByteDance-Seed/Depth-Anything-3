# Linux aarch64 Setup Guide (Python 3.9)

This guide targets running Depth Anything 3 on Linux `aarch64` with Python `3.9`.
It includes two paths:
- Path A: run backend without COLMAP export (`pycolmap` optional)
- Path B: build COLMAP + PyCOLMAP from source

## 0. What we verified

- `torch` and `torchvision` wheels for Linux `aarch64` + `cp39` exist.
- `xformers` wheel for Linux `aarch64` + `cp39` is not available.
- `pycolmap` wheel for Linux `aarch64` + `cp39` is not available.

Because of this, default setup installs all required runtime dependencies except `xformers` and `pycolmap`.
`colmap` export can be enabled later by source build.

## 1. System packages (Ubuntu/Debian)

```bash
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
```

## 2. Clone and checkout

```bash
git clone --recursive https://github.com/forgottencow77/Depth-Anything-3.git
cd Depth-Anything-3
git checkout feat/visual-slam-mvp
```

## 3. Virtual environment and Python deps

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel

# Install package itself without forcing unavailable deps
python -m pip install -e . --no-deps

# Install PyTorch stack first (aarch64 cp39 wheels exist)
python -m pip install "torch==2.5.1" "torchvision==0.20.1"

# Install project requirements excluding unavailable wheels
python -m pip install $(grep -Ev '^(xformers|pycolmap)$' requirements.txt | tr '\n' ' ')

# Align moviepy and numpy for current code path
python -m pip install --force-reinstall "moviepy==1.0.3" "numpy<2"

# Useful test tooling
python -m pip install pytest httpx
```

## 4. Validation

```bash
python - <<'PY'
import torch, fastapi, typer, moviepy.editor
print('torch:', torch.__version__)
print('cuda_available:', torch.cuda.is_available())
print('imports-ok')
PY

# If your environment has OpenMP duplicate runtime issue:
export KMP_DUPLICATE_LIB_OK=TRUE

da3 backend --help
pytest --version
```

## 5. Start backend

```bash
source .venv/bin/activate
export KMP_DUPLICATE_LIB_OK=TRUE

da3 backend \
  --model-dir depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
  --device cuda \
  --host 0.0.0.0 \
  --port 8008
```

## 6. Optional: Build COLMAP + PyCOLMAP from source

Run this when you need `--export-format colmap` support:

```bash
bash scripts/setup/build_pycolmap_aarch64.sh
```

Then verify:

```bash
source .venv/bin/activate
python - <<'PY'
import pycolmap
print('pycolmap:', pycolmap.__version__)
PY
```

## 7. Notes for GPU servers on aarch64

- Some aarch64 environments use vendor-specific PyTorch/CUDA wheels (for example Jetson-style stacks).
- If `torch.cuda.is_available()` is `False`, install vendor-provided torch/torchvision builds and re-run the validation step.
- `xformers` may require source build and can fail depending on toolchain and torch version; this repository can run without it.

## 8. One-command setup script

If you want to automate Path A setup:

```bash
bash scripts/setup/setup_linux_aarch64.sh
```

The script is idempotent and can be re-run safely.
