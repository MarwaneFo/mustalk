#!/usr/bin/env bash
set -e
cd /opt/MuseTalk

echo "=== MuseTalk container ==="
python -c "import torch;print('torch',torch.__version__,'| cuda:',torch.cuda.is_available())" || true

if [ ! -e models/musetalkV15 ] && [ ! -e models/musetalk ]; then
  echo "!! Aucun poids dans /opt/MuseTalk/models"
  echo "!! Monter un volume, ou lancer : download_weights.sh /opt/MuseTalk/models"
fi

exec "$@"
