# Neural Forge - single-container Hugging Face Space (Docker SDK, GPU hardware).
# Model weights are never baked into the image; they download on first start into HF_HOME.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl git \
        libgl1 libglib2.0-0 libgomp1 libsm6 libxext6 \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

RUN useradd -m -u 1000 user

WORKDIR /app

COPY --chown=user:user requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r /app/requirements.txt
RUN pip uninstall -y onnxruntime opencv-python opencv-contrib-python opencv-contrib-python-headless || true
RUN pip install --force-reinstall --no-deps "onnxruntime-gpu>=1.19" "opencv-python-headless>=4.9"

ENV HOME=/home/user \
    APP_DIR=/app \
    HF_HOME=/home/user/.cache/huggingface \
    NF_MODEL_CACHE_DIR=/home/user/.cache/huggingface/neural_forge \
    NF_CROPS_DIR=/home/user/recordings/crops \
    XDG_CACHE_HOME=/home/user/.cache \
    TORCH_HOME=/home/user/.cache/torch \
    YOLO_CONFIG_DIR=/home/user/.config/Ultralytics \
    MPLCONFIGDIR=/home/user/.cache/matplotlib \
    VLLM_CACHE_ROOT=/home/user/.cache/vllm \
    TRITON_CACHE_DIR=/home/user/.cache/triton \
    NUMBA_CACHE_DIR=/home/user/.cache/numba \
    NF_PORT=7860 \
    NF_HOST=0.0.0.0

COPY --chown=user:user . /app

RUN mkdir -p /home/user/.cache /home/user/.config /home/user/recordings \
    && chown -R user:user /home/user /app \
    && chmod +x /app/deploy/*.sh

USER user

EXPOSE 7860

CMD ["bash", "/app/deploy/entrypoint.sh"]
