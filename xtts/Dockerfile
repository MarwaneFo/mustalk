# =============================================================
#  XTTS-v2 - synthese vocale avec clonage de voix, RunPod Serverless
#
#  Entree : du texte + une voix de reference (6 s suffisent)
#  Sortie : un WAV, directement consommable par l'endpoint MuseTalk.
#
#  LICENCE : le code (coqui-tts) est sous MPL-2.0, mais les POIDS
#  XTTS-v2 relevent de la Coqui Public Model License, qui interdit
#  l'usage commercial. Convient aux tests et a l'usage personnel.
# =============================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Etc/UTC \
    COQUI_TOS_AGREED=1 \
    XDG_DATA_HOME=/opt/tts-data \
    NUMBA_CACHE_DIR=/tmp/numba

# espeak-ng sert au phonemiseur, libsndfile a la lecture/ecriture audio.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-dev python3-pip \
      ffmpeg espeak-ng libsndfile1 curl ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && ln -sf /usr/bin/python3.10 /usr/bin/python \
 && ln -sf /usr/bin/pip3 /usr/bin/pip

RUN pip install --upgrade pip setuptools wheel \
 && pip install torch==2.4.1 torchaudio==2.4.1 \
      --index-url https://download.pytorch.org/whl/cu124

# Version choisie avec soin : la 0.27.5 tire transformers>=4.57, qui
# importe torchcodec -- lequel exige un torch bien plus recent que le
# 2.4.1 installe ci-dessus. La 0.25.3 se contente de torch>=2.1 et
# epingle numpy<2.0, ce qui ecarte au passage le conflit binaire
# numpy 2 / opencv. XTTS-v2 y est pleinement supporte.
RUN pip install "coqui-tts==0.25.3" runpod requests 
 && python -c "import TTS, numpy, transformers; print('coqui-tts', TTS.__version__, 'numpy', numpy.__version__, 'transformers', transformers.__version__)"

# Poids XTTS-v2 (~2 Go), telecharges au build : les tirer au demarrage
# ajouterait plusieurs minutes a chaque demarrage a froid.
RUN python -c "\
from TTS.utils.manage import ModelManager; \
p = ModelManager().download_model('tts_models/multilingual/multi-dataset/xtts_v2'); \
print('poids installes :', p)"

# Voix de reference : echantillon d'Inna, pour que la voix synthetisee
# corresponde a l'avatar anime par l'endpoint MuseTalk.
COPY xtts/speaker_inna.wav /opt/voices/inna.wav
COPY xtts/handler.py /opt/handler.py

# Echoue le build plutot que de livrer une image qui plantera au demarrage.
RUN python -c "\
from TTS.api import TTS; \
t = TTS('tts_models/multilingual/multi-dataset/xtts_v2', progress_bar=False); \
print('XTTS-v2 charge -- verification OK')"

ENV DEFAULT_SPEAKER=/opt/voices/inna.wav \
    DEFAULT_LANGUAGE=fr

WORKDIR /opt
ENTRYPOINT []
CMD ["python", "-u", "/opt/handler.py"]
