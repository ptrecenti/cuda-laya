FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Instalação do PyTorch Nightly compilado com CUDA 13.0 e suporte direto a sm_120 (Blackwell)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

# Instalação do runtime Laya
RUN pip install --no-cache-dir "laya[serve]"

EXPOSE 8090

ENTRYPOINT ["laya-serve"]
