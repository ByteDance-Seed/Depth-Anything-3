FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    git-lfs \
    ffmpeg \
    cmake \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir xformers --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir hatchling hatch-vcs && \
    pip install --no-cache-dir -e ".[all]" --no-build-isolation

EXPOSE 7860

ENTRYPOINT ["da3"]
CMD ["gradio"]
