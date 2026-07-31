# =============================================================
#  MuseTalk - image GPU
#  CUDA 11.8 / PyTorch 2.0.1 / Python 3.10 (configuration officielle)
# =============================================================
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# USE_TF=0 : transformers importe TensorFlow dès qu'il le trouve. MuseTalk n'en a
# aucun besoin, et cet import bloque l'initialisation (mesuré sur le pod : import
# indéfini avec TF, 17 s sans).
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Etc/UTC \
    HF_HOME=/root/.cache/huggingface \
    FFMPEG_PATH=/usr/bin \
    TORCH_HOME=/opt/torch_cache \
    USE_TF=0 \
    USE_TORCH=1

# ---------- 1. Système ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-dev python3-pip \
      git git-lfs curl wget ca-certificates xz-utils unzip \
      ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
      build-essential ninja-build \
 && rm -rf /var/lib/apt/lists/* \
 && ln -sf /usr/bin/python3.10 /usr/bin/python \
 && ln -sf /usr/bin/pip3 /usr/bin/pip

# ---------- 2. PyTorch CUDA 11.8 ----------
RUN pip install --upgrade pip setuptools wheel \
 && pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
      --index-url https://download.pytorch.org/whl/cu118

# ---------- 3. Code MuseTalk ----------
ARG MUSETALK_REPO=https://github.com/TMElyralab/MuseTalk.git
ARG MUSETALK_REF=main
RUN git clone "${MUSETALK_REPO}" /opt/MuseTalk \
 && cd /opt/MuseTalk && git checkout "${MUSETALK_REF}" && rm -rf .git

WORKDIR /opt/MuseTalk
RUN pip install -r requirements.txt \
 && pip install openmim "huggingface_hub[cli]" gdown

# ---------- 4. Stack OpenMMLab (roues précompilées) ----------
RUN mim install mmengine \
 && pip install mmcv==2.0.1 \
      -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html \
 && mim install "mmdet==3.1.0" \
 && mim install "mmpose==1.1.0"

COPY entrypoint.sh download_weights.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/download_weights.sh

# ---------- 5. Poids (dernière couche : la plus lourde) ----------
# true  -> image autonome (~20 Go), idéale pour un template RunPod
# false -> image légère, poids à monter sur /opt/MuseTalk/models
ARG DOWNLOAD_WEIGHTS=true
RUN if [ "$DOWNLOAD_WEIGHTS" = "true" ]; then \
      /usr/local/bin/download_weights.sh /opt/MuseTalk/models ; \
    else mkdir -p /opt/MuseTalk/models ; fi

EXPOSE 7860
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["sleep", "infinity"]
