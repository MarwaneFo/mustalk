"""
Handler RunPod Serverless pour Chatterbox Multilingual v3 (Resemble AI, MIT).

Pourquoi ce troisieme moteur vocal a cote de XTTS-v2 : les poids de XTTS sont
sous Coqui Public Model License, donc **non commerciaux**, et Coqui a ferme en
janvier 2024 -- il n'y a plus personne a qui acheter une licence. Chatterbox
est en MIT, couvre 23 langues au lieu de 17, et est encore maintenu.

Requete :
{
  "text": "Bonjour...",
  "language": "fr",                # 23 langues ; voir LANGUES
  "voice": "inna",                 # voix embarquee, ou :
  "speaker_wav_base64": "...",     # echantillon de reference fourni
  "exaggeration": 0.5,             # 0.25 a 2.0 ; expressivite
  "temperature": 0.8,              # 0.05 a 5.0 ; variabilite
  "cfg_weight": 0.5                # adherence au texte et au rythme
}

Reponse : {"audio_base64" (WAV), "duration_s", "sample_rate", "voice", ...}

Chatterbox incorpore un filigrane inaudible dans tout ce qu'il produit --
choix delibere de Resemble, qui ne se desactive pas. A savoir avant de
l'adopter, meme si c'est sans consequence pour la plupart des usages.
"""
import base64
import io
import os

print("[boot] handler chatterbox demarre", flush=True)

import sys
import tempfile
import time
import traceback

import requests
import runpod

VOICES_DIR = os.environ.get("VOICES_DIR", "/opt/voices")
DEFAULT_VOICE = os.environ.get("DEFAULT_VOICE", "inna")
T3_MODEL = os.environ.get("CHATTERBOX_T3", "v3")

_MODELE = None
LANGUES = set()


def _charger():
    """Chargement au PREMIER JOB, jamais a l'import.

    Lecon tiree de LatentSync : charger avant runpod.serverless.start() rend
    toute erreur invisible. Le processus meurt sans que RunPod ait un job a
    marquer en echec, et le worker s'annonce pret alors qu'il ne peut rien
    accepter -- les jobs s'empilent en file sans explication. Ici, la moindre
    erreur revient dans la reponse du job.
    """
    global _MODELE, LANGUES
    t = time.time()

    import torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

    LANGUES = set(SUPPORTED_LANGUAGES)

    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        archs = torch.cuda.get_arch_list()
        print(f"[init] torch={torch.__version__} gpu={torch.cuda.get_device_name(0)} "
              f"capability=sm_{cap[0]}{cap[1]}", flush=True)
        if f"sm_{cap[0]}{cap[1]}" not in archs:
            print(f"[init] !! sm_{cap[0]}{cap[1]} absent des noyaux compiles dans "
                  "torch. GPU trop recent : choisis Ampere, Ada ou Hopper, "
                  "jamais Blackwell.", flush=True)
        device = "cuda"
    else:
        print("[init] !! aucun GPU detecte, repli sur le processeur", flush=True)
        device = "cpu"

    _MODELE = ChatterboxMultilingualTTS.from_pretrained(device, t3_model=T3_MODEL)
    print(f"[init] Chatterbox {T3_MODEL} pret en {time.time() - t:.0f}s, "
          f"{len(LANGUES)} langues, sortie a {_MODELE.sr} Hz", flush=True)


def _voix_disponibles():
    if not os.path.isdir(VOICES_DIR):
        return {}
    return {os.path.splitext(f)[0].lower(): os.path.join(VOICES_DIR, f)
            for f in sorted(os.listdir(VOICES_DIR)) if f.lower().endswith(".wav")}


def _reference(job_input, workdir):
    """Trois facons de designer la voix, par ordre de priorite.

    Un echantillon fourni l'emporte sur un nom : il permet d'essayer une voix
    sans reconstruire l'image, ce qui nous a deja servi.
    """
    if job_input.get("speaker_wav_base64"):
        chemin = os.path.join(workdir, "reference.wav")
        with open(chemin, "wb") as f:
            f.write(base64.b64decode(job_input["speaker_wav_base64"]))
        return chemin, "fournie"

    dispo = _voix_disponibles()
    nom = (job_input.get("voice") or DEFAULT_VOICE).lower()
    if nom in dispo:
        return dispo[nom], nom
    if dispo:
        raise ValueError(f"Voix '{nom}' inconnue. Disponibles : {sorted(dispo)}")
    return None, "par defaut du modele"


def handler(job):
    started = time.time()
    job_input = job.get("input") or {}
    workdir = tempfile.mkdtemp(prefix="chatterbox_")
    try:
        etapes = {}

        if _MODELE is None:
            t = time.time()
            _charger()
            etapes["chargement_modele"] = round(time.time() - t, 1)

        texte = (job_input.get("text") or "").strip()
        if not texte:
            raise ValueError("Fournir 'text'.")

        langue = (job_input.get("language") or "fr").lower()
        if langue not in LANGUES:
            raise ValueError(f"Langue '{langue}' non supportee. "
                             f"Choix : {sorted(LANGUES)}")

        reference, nom_voix = _reference(job_input, workdir)

        import torch
        import numpy as np
        import soundfile as sf

        t = time.time()
        kwargs = {
            "language_id": langue,
            # Valeurs par defaut de Resemble. exaggeration au-dela de 1.0
            # devient instable, ils le signalent eux-memes.
            "exaggeration": float(job_input.get("exaggeration", 0.5)),
            "temperature": float(job_input.get("temperature", 0.8)),
            "cfg_weight": float(job_input.get("cfg_weight", 0.5)),
        }
        if reference:
            kwargs["audio_prompt_path"] = reference

        wav = _MODELE.generate(texte, **kwargs)
        etapes["synthese"] = round(time.time() - t, 1)

        echantillons = wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
        tampon = io.BytesIO()
        sf.write(tampon, echantillons, _MODELE.sr, format="WAV", subtype="PCM_16")
        donnees = tampon.getvalue()

        return {
            "audio_base64": base64.b64encode(donnees).decode(),
            "duration_s": round(len(echantillons) / _MODELE.sr, 2),
            "sample_rate": _MODELE.sr,
            "voice": nom_voix,
            "language": langue,
            "model": f"chatterbox-multilingual-{T3_MODEL}",
            "etapes": etapes,
            "total_s": round(time.time() - started, 2),
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)


runpod.serverless.start({"handler": handler})
