#!/usr/bin/env bash
# Télécharge les poids de MuseTalk dans le dossier passé en $1.
#
# URLs directes plutôt que `huggingface-cli` : ce dernier est déprécié et sort
# désormais "huggingface-cli is deprecated and no longer works" sans rien
# télécharger — constaté en conditions réelles, 49 Mo obtenus au lieu de 9 Go.
#
# Chaque fichier est vérifié à l'octet près : un téléchargement tronqué puis
# repris produit un fichier plus GROS que l'original, donc invalide, et l'erreur
# ne se révèle qu'au chargement du modèle. Les tailles ci-dessous ont été
# relevées sur une installation qui génère effectivement des vidéos.
set -euo pipefail

M="${1:-/opt/MuseTalk/models}"
HF=https://huggingface.co
CK="${TORCH_HOME:-/opt/torch_cache}/hub/checkpoints"
mkdir -p "$M" "$CK"

# url | destination | taille attendue en octets
FILES="
$HF/TMElyralab/MuseTalk/resolve/main/musetalk/musetalk.json|$M/musetalk/musetalk.json|748
$HF/TMElyralab/MuseTalk/resolve/main/musetalk/pytorch_model.bin|$M/musetalk/pytorch_model.bin|3400076549
$HF/TMElyralab/MuseTalk/resolve/main/musetalkV15/musetalk.json|$M/musetalkV15/musetalk.json|748
$HF/TMElyralab/MuseTalk/resolve/main/musetalkV15/unet.pth|$M/musetalkV15/unet.pth|3400074924
$HF/stabilityai/sd-vae-ft-mse/resolve/main/config.json|$M/sd-vae/config.json|547
$HF/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin|$M/sd-vae/diffusion_pytorch_model.bin|334707217
$HF/openai/whisper-tiny/resolve/main/config.json|$M/whisper/config.json|1983
$HF/openai/whisper-tiny/resolve/main/pytorch_model.bin|$M/whisper/pytorch_model.bin|151095027
$HF/openai/whisper-tiny/resolve/main/preprocessor_config.json|$M/whisper/preprocessor_config.json|184990
$HF/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth|$M/dwpose/dw-ll_ucoco_384.pth|406878486
$HF/ManyOtherFunctions/face-parse-bisent/resolve/main/79999_iter.pth|$M/face-parse-bisent/79999_iter.pth|53289463
https://download.pytorch.org/models/resnet18-5c106cde.pth|$M/face-parse-bisent/resnet18-5c106cde.pth|46827520
https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth|$CK/s3fd-619a316812.pth|89843225
"

# Herestring et non pipe : un `while` derrière un pipe tourne dans un sous-shell,
# où un `exit 1` ne ferait pas échouer le build.
while IFS='|' read -r url dest size; do
    [ -z "${url:-}" ] && continue
    mkdir -p "$(dirname "$dest")"
    ok=0
    for try in 1 2 3 4 5 6 7 8; do
        rm -f "$dest"
        # Pas de -c : reprendre sur un fichier corrompu aggrave le problème.
        wget -q --tries=3 --timeout=60 -O "$dest" "$url" || true
        got=$(stat -c%s "$dest" 2>/dev/null || echo 0)
        if [ "$got" = "$size" ]; then
            echo "  OK   $(basename "$dest")  ($got octets)"
            ok=1
            break
        fi
        echo "  ...tentative $try : $got / $size attendu  ($(basename "$dest"))"
        sleep 5
    done
    if [ "$ok" != 1 ]; then
        echo "  ECHEC definitif : $url"
        exit 1
    fi
done <<< "$FILES"

# syncnet (1,5 Go) ne sert qu'à l'entraînement : volontairement omis.

echo ">>> Poids installés : $(du -sh "$M" | cut -f1)"
find "$M" -type f -printf '%12s  %p\n' | sort -rn
