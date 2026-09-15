"""
Handler RunPod Serverless pour LatentSync (ByteDance, Apache 2.0).

Pourquoi ce second modèle à côté de MuseTalk : mesuré sur nos propres rendus,
MuseTalk fait bouger la bouche à 80-105 % de son amplitude de parole pendant
les silences, et n'articule pas les visèmes des consonnes. La cause est son
entraînement, qui ne comporte quasiment pas de supervision de synchronisation
labiale. LatentSync, lui, est entraîné sous contrainte d'un SyncNet.

Requête :
{
  "audio_base64": "...",        # ou "audio_url"
  "video_url":     "https://…", # facultatif ; sinon la vidéo embarquée
  "inference_steps": 20,        # 10 a 50 ; le temps y est proportionnel
  "resolution":      512,       # 256 ou 512 ; 256 va 3 a 4 fois plus vite
  "guidance_scale":  1.5,
  "seed": 1247                  # -1 pour aléatoire
}

Réponse : {"video_base64", "duration_s", "etapes", "inference_steps", ...}

Différence d'architecture avec MuseTalk, qui change l'usage : LatentSync ne
pré-calcule pas d'avatar. Il retraite la vidéo source à chaque appel, ce qui
supprime toute la mécanique d'avatars, de masques et de latents — au prix
d'un coût déplacé du démarrage vers chaque génération.
"""
import base64
import os

print("[boot] handler demarre", flush=True)

import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import requests
import runpod

ROOT = os.environ.get("LATENTSYNC_ROOT", "/opt/LatentSync")
os.chdir(ROOT)          # les chemins du projet sont relatifs : configs/, checkpoints/
sys.path.insert(0, ROOT)

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from diffusers import AutoencoderKL, DDIMScheduler  # noqa: E402
from accelerate.utils import set_seed  # noqa: E402
from latentsync.models.unet import UNet3DConditionModel  # noqa: E402
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline  # noqa: E402
from latentsync.whisper.audio2feature import Audio2Feature  # noqa: E402
from DeepCache import DeepCacheSDHelper  # noqa: E402

CONFIG = os.environ.get("LATENTSYNC_CONFIG", "configs/unet/stage2_512.yaml")
CKPT = os.environ.get("LATENTSYNC_CKPT", "checkpoints/latentsync_unet.pt")
VIDEO_DEFAUT = os.environ.get("LATENTSYNC_VIDEO", "/opt/LatentSync/data/Inna.mp4")

_PIPELINE = None
_CONFIG = None


def _charger():
    """Chargement unique, au PREMIER JOB et non a l'import.

    Reprend scripts/inference.py de LatentSync, en gardant le pipeline vivant
    entre les requetes : le recharger a chaque job couterait plus cher que la
    generation elle-meme.

    Appele depuis le handler, deliberement. Charger avant
    runpod.serverless.start() rendait toute erreur invisible : le processus
    mourait sans que RunPod ait un job a marquer en echec, et le worker
    s'annoncait pret alors qu'il ne pouvait rien accepter.
    """
    global _PIPELINE, _CONFIG
    t = time.time()

    config = OmegaConf.load(CONFIG)
    _CONFIG = config

    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        archs = torch.cuda.get_arch_list()
        print(f"[init] torch={torch.__version__} gpu={torch.cuda.get_device_name(0)} "
              f"capability=sm_{cap[0]}{cap[1]}", flush=True)
        print(f"[init] architectures compilées : {archs}", flush=True)
        if f"sm_{cap[0]}{cap[1]}" not in archs:
            print(f"[init] !! ATTENTION : sm_{cap[0]}{cap[1]} absent des noyaux compilés. "
                  "Ce GPU est trop récent pour cette version de torch — choisis "
                  "Ampere, Ada ou Hopper, jamais Blackwell.", flush=True)
    else:
        print("[init] !! aucun GPU détecté", flush=True)

    # Le float16 n'est utilisé qu'à partir de sm_80 ; en dessous, LatentSync
    # retombe en float32 et la génération devient beaucoup plus lente.
    fp16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if fp16 else torch.float32
    print(f"[init] précision {'float16' if fp16 else 'float32'}", flush=True)

    scheduler = DDIMScheduler.from_pretrained("configs")

    dim = config.model.cross_attention_dim
    whisper_path = "checkpoints/whisper/small.pt" if dim == 768 else "checkpoints/whisper/tiny.pt"
    audio_encoder = Audio2Feature(
        model_path=whisper_path,
        device="cuda",
        num_frames=config.data.num_frames,
        audio_feat_length=config.data.audio_feat_length,
    )

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model), CKPT, device="cpu",
    )
    unet = unet.to(dtype=dtype)

    pipeline = LipsyncPipeline(
        vae=vae, audio_encoder=audio_encoder, unet=unet, scheduler=scheduler,
    ).to("cuda")

    # DeepCache réutilise une partie des calculs entre pas de diffusion.
    # C'est le réglage que LatentSync retient dans son propre inference.sh.
    helper = DeepCacheSDHelper(pipe=pipeline)
    helper.set_params(cache_interval=3, cache_branch_id=0)
    helper.enable()

    _PIPELINE = pipeline
    print(f"[init] pipeline prêt en {time.time() - t:.0f}s "
          f"(résolution {config.data.resolution})", flush=True)


def _fetch(job_input, cle_b64, cle_url, dest):
    if job_input.get(cle_b64):
        with open(dest, "wb") as f:
            f.write(base64.b64decode(job_input[cle_b64]))
        return dest
    if job_input.get(cle_url):
        r = requests.get(job_input[cle_url], timeout=180)
        r.raise_for_status()
        with open(dest, "wb") as f:
            f.write(r.content)
        return dest
    return None


def handler(job):
    started = time.time()
    job_input = job.get("input") or {}
    workdir = tempfile.mkdtemp(prefix="latentsync_")
    try:
        etapes = {}

        # Premier job du worker : c'est ici qu'on paye le chargement. Plus
        # long pour lui, mais toute erreur remonte dans la reponse au lieu
        # de tuer un processus muet.
        if _PIPELINE is None:
            t = time.time()
            _charger()
            etapes["chargement_modeles"] = round(time.time() - t, 1)

        t = time.time()
        audio = _fetch(job_input, "audio_base64", "audio_url",
                       os.path.join(workdir, "audio.wav"))
        if not audio:
            raise ValueError("Fournir 'audio_base64' ou 'audio_url'.")
        video = _fetch(job_input, "video_base64", "video_url",
                       os.path.join(workdir, "source.mp4")) or VIDEO_DEFAUT
        if not os.path.exists(video):
            raise FileNotFoundError(f"Vidéo source absente : {video}")
        etapes["entrees"] = round(time.time() - t, 1)

        # Les deux leviers qui font le compromis vitesse/qualite. Le temps de
        # diffusion est proportionnel au nombre de pas, et croit avec le carre
        # de la resolution : 256 px va environ quatre fois plus vite que 512.
        pas = max(1, min(50, int(job_input.get("inference_steps", 20))))
        resolution = int(job_input.get("resolution", _CONFIG.data.resolution))
        if resolution not in (256, 512):
            raise ValueError("resolution doit valoir 256 ou 512, pas %r" % resolution)
        guidage = float(job_input.get("guidance_scale", 1.5))
        graine = int(job_input.get("seed", 1247))
        if graine != -1:
            set_seed(graine)          # rendu reproductible d'un appel à l'autre
        else:
            torch.seed()

        sortie = os.path.join(workdir, "sortie.mp4")
        temp_dir = os.path.join(workdir, "temp")

        t = time.time()
        _PIPELINE(
            video_path=video,
            audio_path=audio,
            video_out_path=sortie,
            num_frames=_CONFIG.data.num_frames,
            num_inference_steps=pas,
            guidance_scale=guidage,
            weight_dtype=torch.float16 if torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] > 7 else torch.float32,
            width=resolution,
            height=resolution,
            mask_image_path=_CONFIG.data.mask_image_path,
            temp_dir=temp_dir,
        )
        etapes["diffusion"] = round(time.time() - t, 1)

        if not os.path.exists(sortie):
            raise RuntimeError(f"Vidéo non produite : {sortie}")

        with open(sortie, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()

        return {
            "duration_s": round(time.time() - started, 2),
            "etapes": etapes,
            "inference_steps": pas,
            "guidance_scale": guidage,
            "seed": graine,
            "resolution": resolution,
            "video_base64": video_b64,
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


runpod.serverless.start({"handler": handler})
