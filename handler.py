"""
Handler RunPod Serverless pour MuseTalk v1.5.

Les modèles sont chargés une seule fois au démarrage du worker (hors du handler),
sinon on paierait ~8,7 Go de chargement à chaque requête.

Subtilité : realtime_inference.py définit `args`, `vae`, `unet`, `whisper`, `fp`…
à l'intérieur de `if __name__ == "__main__"`. Un import ne les crée donc pas ;
il faut les injecter dans le module avant d'instancier `Avatar`.

Entrée attendue :
{
  "input": {
    "avatar_id":    "Inna",           # avatar pré-calculé présent dans results/
    "audio_url":    "https://...",    # ou "audio_base64"
    "audio_base64": "...",
    "fps":          25,
    "batch_size":   20
  }
}

Sortie : {"video_base64": "...", "duration_s": 12.4, "avatar_id": "Inna"}
"""

import os
import sys
import base64
import shutil
import tempfile
import subprocess
import time
import argparse
import traceback

import requests
import runpod

MUSETALK_ROOT = os.environ.get("MUSETALK_ROOT", "/opt/MuseTalk")
DEFAULT_AVATAR = os.environ.get("AVATAR_ID", "Inna")

# Les avatars pré-calculés pèsent plusieurs Go (full_imgs) : trop pour une image.
# RunPod Serverless monte le volume réseau sur /runpod-volume ; on l'utilise en
# priorité, avec repli sur l'image pour un avatar éventuellement embarqué.
AVATAR_ROOTS = [
    p for p in (
        os.environ.get("AVATARS_DIR"),
        "/runpod-volume/musetalk/results/v15/avatars",
        "/workspace/musetalk/results/v15/avatars",
        os.path.join(MUSETALK_ROOT, "results/v15/avatars"),
    ) if p
]

# À poser AVANT tout import de transformers : sinon il charge TensorFlow, ce qui
# bloque l'initialisation du worker (constaté sur le pod).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TORCH_HOME", "/opt/torch_cache")

# Les chemins du projet sont relatifs (./models, ./results) : il faut s'y placer.
os.chdir(MUSETALK_ROOT)
sys.path.insert(0, MUSETALK_ROOT)

import torch  # noqa: E402
from transformers import WhisperModel  # noqa: E402
from musetalk.utils.utils import load_all_model  # noqa: E402
from musetalk.utils.audio_processor import AudioProcessor  # noqa: E402
from musetalk.utils.face_parsing import FaceParsing  # noqa: E402
import scripts.realtime_inference as ri  # noqa: E402


def _build_args():
    """Reconstruit le Namespace que realtime_inference attend en global."""
    return argparse.Namespace(
        version="v15",
        ffmpeg_path="/usr/bin",
        gpu_id=0,
        vae_type="sd-vae",
        unet_config="./models/musetalkV15/musetalk.json",
        unet_model_path="./models/musetalkV15/unet.pth",
        whisper_dir="./models/whisper",
        bbox_shift=0,
        result_dir="./results",
        extra_margin=10,
        fps=25,
        audio_padding_length_left=2,
        audio_padding_length_right=2,
        batch_size=20,
        output_vid_name=None,
        use_saved_coord=False,
        saved_coord=False,
        parsing_mode="jaw",
        left_cheek_width=90,
        right_cheek_width=90,
        skip_save_images=False,
        inference_config="configs/inference/realtime.yaml",
    )


def _load_models():
    """Chargement unique, au démarrage du worker."""
    args = _build_args()
    ri.args = args

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device} cuda={torch.cuda.is_available()}", flush=True)

    # Sans ces lignes, une incompatibilité entre l'architecture du GPU et les
    # noyaux compilés dans torch ne se manifeste que par un « CUDA error: no
    # kernel image is available » au premier .half(), sans dire quel GPU ni
    # quelles architectures sont supportées. Ici on le voit avant le crash.
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        archs = torch.cuda.get_arch_list()
        print(f"[init] torch={torch.__version__} gpu={torch.cuda.get_device_name(0)} "
              f"capability=sm_{cap[0]}{cap[1]}", flush=True)
        print(f"[init] architectures compilées dans torch : {archs}", flush=True)
        if f"sm_{cap[0]}{cap[1]}" not in archs:
            print(f"[init] !! ATTENTION : sm_{cap[0]}{cap[1]} absent des noyaux compilés. "
                  "Ce GPU est trop récent pour cette version de torch — le chargement "
                  "des modèles va échouer. Choisis un autre GPU ou une image plus récente.",
                  flush=True)

    vae, unet, pe = load_all_model(
        unet_model_path=args.unet_model_path,
        vae_type=args.vae_type,
        unet_config=args.unet_config,
        device=device,
    )
    pe = pe.half().to(device)
    vae.vae = vae.vae.half().to(device)
    unet.model = unet.model.half().to(device)

    audio_processor = AudioProcessor(feature_extractor_path=args.whisper_dir)
    weight_dtype = unet.model.dtype
    whisper = WhisperModel.from_pretrained(args.whisper_dir)
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)

    fp = FaceParsing(
        left_cheek_width=args.left_cheek_width,
        right_cheek_width=args.right_cheek_width,
    )

    # Injection des globales dont Avatar.inference() a besoin.
    ri.device = device
    ri.vae, ri.unet, ri.pe = vae, unet, pe
    ri.timesteps = torch.tensor([0], device=device)
    ri.audio_processor = audio_processor
    ri.whisper = whisper
    ri.weight_dtype = weight_dtype
    ri.fp = fp
    print("[init] modèles chargés", flush=True)


_load_models()
_AVATARS = {}


def _resolve_avatars():
    """
    La classe Avatar construit ses chemins en dur : ./results/v15/avatars/<id>,
    relatifs au répertoire courant. On fait donc pointer MUSETALK_ROOT/results
    vers le premier emplacement qui contient réellement des avatars.
    """
    link = os.path.join(MUSETALK_ROOT, "results")
    for root in AVATAR_ROOTS:
        if not os.path.isdir(root):
            continue
        if os.path.isdir(os.path.join(MUSETALK_ROOT, "results/v15/avatars")) \
                and os.path.samefile(root, os.path.join(MUSETALK_ROOT, "results/v15/avatars")):
            return root
        target = os.path.abspath(os.path.join(root, "..", ".."))  # .../results
        if os.path.islink(link) or not os.path.exists(link):
            if os.path.islink(link):
                os.unlink(link)
            os.symlink(target, link)
            print(f"[avatars] results -> {target}", flush=True)
        return root
    print(f"[avatars] AUCUN emplacement trouvé parmi {AVATAR_ROOTS}", flush=True)
    return None


_AVATAR_ROOT = _resolve_avatars()


def _ensure_frames(avatar_id):
    """
    full_imgs pèse 3,2 Go : trop lourd pour une image Docker. On ne l'embarque
    donc pas, on le régénère au démarrage depuis la vidéo source — MuseTalk
    l'avait lui-même produit par extraction de frames, en 1920x1080 nommées
    00000000.png…, ce que ffmpeg reproduit à l'identique.

    Coût : environ 1 à 2 min, une seule fois par worker.
    """
    base = f"./results/v15/avatars/{avatar_id}"
    full = os.path.join(base, "full_imgs")
    mask = os.path.join(base, "mask")
    if not os.path.isdir(mask):
        return  # avatar monté depuis un volume : rien à régénérer
    n_mask = len(os.listdir(mask))
    n_full = len(os.listdir(full)) if os.path.isdir(full) else 0
    if n_full >= n_mask > 0:
        return

    video = os.environ.get("AVATAR_VIDEO", f"/opt/MuseTalk/data/video/{avatar_id}.mp4")
    if not os.path.exists(video):
        raise FileNotFoundError(
            f"Vidéo source absente : {video}. Elle est nécessaire pour "
            f"régénérer full_imgs ({n_mask} frames attendues)."
        )

    print(f"[frames] extraction de {n_mask} frames depuis {video}", flush=True)
    t = time.time()
    os.makedirs(full, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", video,
         "-start_number", "0", os.path.join(full, "%08d.png")],
        check=True,
    )

    # ffmpeg peut produire plus de frames que l'avatar n'en référence
    # (coords.pkl et mask/ font foi) : on élague le surplus pour garder
    # l'alignement entre les listes.
    names = sorted(os.listdir(full))
    for extra in names[n_mask:]:
        os.remove(os.path.join(full, extra))
    print(f"[frames] {len(os.listdir(full))} frames prêtes en {time.time()-t:.0f}s", flush=True)


def _get_avatar(avatar_id, batch_size):
    """Les avatars pré-calculés sont réutilisés entre les requêtes."""
    if avatar_id not in _AVATARS:
        path = f"./results/v15/avatars/{avatar_id}"
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Avatar '{avatar_id}' introuvable. Cherché dans : {AVATAR_ROOTS}. "
                "Il doit être pré-calculé (latents.pt, coords.pkl, mask/, full_imgs/) "
                "et accessible via un volume réseau monté sur /runpod-volume."
            )
        _ensure_frames(avatar_id)
        print(f"[avatar] chargement de {avatar_id}", flush=True)
        _AVATARS[avatar_id] = ri.Avatar(
            avatar_id=avatar_id,
            video_path="",
            bbox_shift=0,
            batch_size=batch_size,
            preparation=False,   # réutilise le pré-calcul, n'écrase rien
        )
    return _AVATARS[avatar_id]


def _fetch_audio(job_input, workdir):
    dest = os.path.join(workdir, "audio.wav")
    if job_input.get("audio_base64"):
        with open(dest, "wb") as f:
            f.write(base64.b64decode(job_input["audio_base64"]))
    elif job_input.get("audio_url"):
        r = requests.get(job_input["audio_url"], timeout=120)
        r.raise_for_status()
        with open(dest, "wb") as f:
            f.write(r.content)
    else:
        raise ValueError("Fournir 'audio_url' ou 'audio_base64'.")
    return dest


def handler(job):
    started = time.time()
    job_input = job.get("input") or {}
    workdir = tempfile.mkdtemp(prefix="musetalk_")
    try:
        avatar_id = job_input.get("avatar_id", DEFAULT_AVATAR)
        fps = int(job_input.get("fps", 25))
        batch_size = int(job_input.get("batch_size", 20))

        audio_path = _fetch_audio(job_input, workdir)
        avatar = _get_avatar(avatar_id, batch_size)

        out_name = f"job_{int(started)}"
        avatar.inference(audio_path, out_name, fps, False)

        out_path = os.path.join(avatar.video_out_path, f"{out_name}.mp4")
        if not os.path.exists(out_path):
            raise RuntimeError(f"Vidéo non produite : {out_path}")

        with open(out_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()
        os.remove(out_path)

        return {
            "avatar_id": avatar_id,
            "duration_s": round(time.time() - started, 2),
            "video_base64": video_b64,
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


runpod.serverless.start({"handler": handler})
