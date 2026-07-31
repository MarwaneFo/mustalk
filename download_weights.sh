#!/usr/bin/env bash
# Télécharge tous les poids nécessaires à MuseTalk dans le dossier passé en $1
set -euo pipefail

M="${1:-/opt/MuseTalk/models}"
mkdir -p "$M"; cd "$M"

echo ">>> MuseTalk v1.0 + v1.5 + syncnet"
huggingface-cli download TMElyralab/MuseTalk --local-dir "$M"

echo ">>> SD-VAE ft-mse"
huggingface-cli download stabilityai/sd-vae-ft-mse \
  config.json diffusion_pytorch_model.bin --local-dir "$M/sd-vae"

echo ">>> Whisper tiny"
huggingface-cli download openai/whisper-tiny \
  config.json pytorch_model.bin preprocessor_config.json --local-dir "$M/whisper"

echo ">>> DWPose"
huggingface-cli download yzd-v/DWPose dw-ll_ucoco_384.pth --local-dir "$M/dwpose"

echo ">>> Face parsing BiSeNet"
mkdir -p "$M/face-parse-bisent"
wget -q -O "$M/face-parse-bisent/resnet18-5c106cde.pth" \
  https://download.pytorch.org/models/resnet18-5c106cde.pth
huggingface-cli download ManyOtherFunctions/face-parse-bisent 79999_iter.pth \
  --local-dir "$M/face-parse-bisent"

# S3FD : musetalk/utils/face_detection le télécharge depuis un site tiers
# à la PREMIÈRE inférence. Sur Serverless, ça veut dire un démarrage à froid
# qui part chercher 86 Mo chez adrianbulat.com — et qui échoue si le site est
# indisponible. On l'embarque donc dans l'image.
CK="${TORCH_HOME:-/opt/torch_cache}/hub/checkpoints"
mkdir -p "$CK"
wget -c -q --tries=10 --timeout=45 -O "$CK/s3fd-619a316812.pth" \
  https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth
echo ">>> S3FD : $(du -h "$CK/s3fd-619a316812.pth" | cut -f1)"

# Le cache HF double la place occupée : on le purge après copie.
rm -rf "$M"/.cache /root/.cache/huggingface

echo ">>> Poids installés : $(du -sh "$M" | cut -f1)"
find "$M" -maxdepth 2 -type f | head -40
