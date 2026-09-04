"""
Handler RunPod Serverless pour XTTS-v2 — synthese vocale avec clonage de voix.

Le modele est charge une seule fois au demarrage du worker : le recharger a
chaque requete couterait une trentaine de secondes pour rien.

Entree :
{
  "input": {
    "text":     "Bonjour, voici mon message.",   # requis
    "language": "fr",                            # defaut : fr
    "speaker_wav_base64": "...",                 # voix de reference ; a defaut
                                                 # celle embarquee dans l'image
    "speed":    1.0                              # 0.5 a 2.0
  }
}

Sortie : {"audio_base64": "<wav>", "sample_rate": 24000, "duration_s": 4.2}

Le WAV renvoye alimente directement l'endpoint MuseTalk via audio_base64.
"""

import base64
import io
import os
import tempfile
import time
import traceback
import wave

import runpod

# A poser avant l'import de TTS : sinon la bibliotheque s'arrete pour
# demander l'acceptation de la licence, et bloque le demarrage du worker.
os.environ.setdefault("COQUI_TOS_AGREED", "1")

import torch  # noqa: E402
from TTS.api import TTS  # noqa: E402

DEFAULT_SPEAKER = os.environ.get("DEFAULT_SPEAKER", "/opt/voices/inna.wav")
DEFAULT_LANGUAGE = os.environ.get("DEFAULT_LANGUAGE", "fr")
MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

# XTTS-v2 accepte ces 17 langues ; on refuse les autres tot, avec un message
# clair, plutot que de laisser le modele echouer de facon obscure.
LANGUES = {"en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
           "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"}

_TTS = None


def _load():
    global _TTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[init] device={device} torch={torch.__version__}", flush=True)
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        print(f"[init] gpu={torch.cuda.get_device_name(0)} capability=sm_{cap[0]}{cap[1]}", flush=True)

    t0 = time.time()
    _TTS = TTS(MODEL, progress_bar=False).to(device)
    print(f"[init] XTTS-v2 charge en {time.time()-t0:.0f}s", flush=True)

    if not os.path.exists(DEFAULT_SPEAKER):
        print(f"[init] !! voix de reference absente : {DEFAULT_SPEAKER}", flush=True)


_load()


def _wav_duration(path):
    with wave.open(path) as w:
        return w.getnframes() / w.getframerate()


def handler(job):
    started = time.time()
    inp = job.get("input") or {}
    workdir = tempfile.mkdtemp(prefix="xtts_")

    try:
        text = (inp.get("text") or "").strip()
        if not text:
            raise ValueError("Le champ 'text' est requis.")
        if len(text) > 5000:
            raise ValueError(f"Texte trop long ({len(text)} caracteres, maximum 5000).")

        lang = (inp.get("language") or DEFAULT_LANGUAGE).lower()
        if lang not in LANGUES:
            raise ValueError(f"Langue '{lang}' non supportee. Choix : {sorted(LANGUES)}")

        speed = float(inp.get("speed", 1.0))
        if not 0.5 <= speed <= 2.0:
            raise ValueError("'speed' doit etre compris entre 0.5 et 2.0.")

        # Voix de reference : celle fournie, sinon celle embarquee.
        if inp.get("speaker_wav_base64"):
            speaker = os.path.join(workdir, "speaker.wav")
            with open(speaker, "wb") as f:
                f.write(base64.b64decode(inp["speaker_wav_base64"]))
        else:
            speaker = DEFAULT_SPEAKER
        if not os.path.exists(speaker):
            raise FileNotFoundError(f"Voix de reference introuvable : {speaker}")

        out = os.path.join(workdir, "sortie.wav")
        print(f"[tts] {len(text)} caracteres, langue={lang}, vitesse={speed}", flush=True)
        _TTS.tts_to_file(text=text, speaker_wav=speaker, language=lang,
                         file_path=out, speed=speed)

        if not os.path.exists(out):
            raise RuntimeError("Aucun audio produit.")

        with open(out, "rb") as f:
            audio = f.read()
        dur = _wav_duration(out)
        print(f"[tts] {dur:.1f}s d'audio en {time.time()-started:.1f}s", flush=True)

        return {
            "audio_base64": base64.b64encode(audio).decode(),
            "sample_rate": 24000,
            "duration_s": round(dur, 2),
            "language": lang,
            "elapsed_s": round(time.time() - started, 2),
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)


runpod.serverless.start({"handler": handler})
