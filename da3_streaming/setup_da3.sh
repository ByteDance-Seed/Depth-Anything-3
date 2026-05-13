#!/bin/bash
set -e

REPO_DIR="Depth-Anything-3"

echo "[*] Verificando repositório..."

if [ ! -d "$REPO_DIR" ]; then
    echo "[*] Clonando repositório..."
    git clone --recursive https://github.com/ByteDance-Seed/Depth-Anything-3.git
else
    echo "[✓] Repositório já existe, pulando clone."
fi

cd "$REPO_DIR"

# Atualiza submodules caso necessário
echo "[*] Verificando submodules..."
git submodule update --init --recursive

echo "[*] Instalando NumPy/Numba..."
pip show numpy >/dev/null 2>&1 && pip show numba >/dev/null 2>&1 || {
    pip install "numpy<2.0" "numba>=0.59.0"
}

echo "[*] Verificando gsplat..."
python -c "import gsplat" >/dev/null 2>&1 || {
    pip install git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70
}

echo "[*] Verificando pacote principal..."
pip show depth-anything-3 >/dev/null 2>&1 || {
    pip install -e .
}

echo "[*] Verificando Open3D / OpenCV / PyVirtualDisplay..."
python -c "import open3d, cv2, pyvirtualdisplay" >/dev/null 2>&1 || {
    pip install open3d opencv-python pyvirtualdisplay
}

echo "[*] Verificando da3_streaming..."
cd da3_streaming

if [ -f "requirements.txt" ]; then
    echo "[*] Instalando requirements..."
    pip install -r requirements.txt
fi

echo "[*] Verificando faiss..."
python -c "import faiss" >/dev/null 2>&1 || {
    pip install -U faiss-cpu
}

echo "[*] Verificando weights..."
if [ ! -d "weights" ]; then
    bash ./scripts/download_weights.sh
else
    echo "[✓] Weights já baixados, pulando."
fi

cd ..

echo "[*] Verificando dependências do sistema..."
if command -v apt-get >/dev/null 2>&1; then
    if ! dpkg -s libgl1-mesa-glx xvfb >/dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get install -y libgl1-mesa-glx xvfb
    else
        echo "[✓] Dependências do sistema já instaladas."
    fi
else
    echo "[!] apt-get não encontrado, pulando dependências do sistema."
fi

echo "[✓] Setup completo. Agora você pode rodar o reconstruct.py na raiz."