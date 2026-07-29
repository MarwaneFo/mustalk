#!/usr/bin/env bash
# =============================================================
#  Installe Docker DANS un pod RunPod
#  Un pod = déjà un conteneur -> dockerd a besoin d'astuces.
#  Fallback automatique sur buildah (rootless, sans démon)
#  qui produit exactement les mêmes images OCI.
# =============================================================
set -uo pipefail

log() { echo -e "\n\033[1;36m>>> $*\033[0m"; }
ok()  { echo -e "\033[1;32m  OK  $*\033[0m"; }
ko()  { echo -e "\033[1;31m  KO  $*\033[0m"; }

log "Environnement"
echo "  kernel : $(uname -r)"
echo "  distro : $(. /etc/os-release; echo "$PRETTY_NAME")"
echo "  root   : $(id -u)"
if [ -f /.dockerenv ] || grep -qa 'docker\|kubepods\|containerd' /proc/1/cgroup 2>/dev/null; then
  echo "  contexte : conteneur (pod RunPod)"
fi

# ---------- 1. Docker CE ----------
log "Installation de Docker CE"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq ca-certificates curl gnupg lsb-release uidmap iptables >/dev/null

install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | gpg --dearmor -o /etc/apt/keyrings/docker.gpg --yes
chmod a+r /etc/apt/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  > /etc/apt/sources.list.d/docker.list

apt-get update -qq
apt-get install -y -qq docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin >/dev/null \
  && ok "binaires Docker installés ($(docker --version))" \
  || { ko "échec apt docker-ce"; exit 1; }

# ---------- 2. Démarrage du démon (mode conteneur) ----------
log "Démarrage de dockerd (storage-driver=vfs, sans iptables)"
mkdir -p /etc/docker /var/log
cat > /etc/docker/daemon.json <<'JSON'
{
  "storage-driver": "vfs",
  "iptables": false,
  "bridge": "none",
  "ip-forward": false,
  "ip6tables": false
}
JSON

pkill dockerd 2>/dev/null
nohup dockerd > /var/log/dockerd.log 2>&1 &

for i in $(seq 1 20); do
  sleep 2
  if docker info >/dev/null 2>&1; then
    ok "dockerd opérationnel"
    docker info --format '  storage-driver : {{.Driver}}'
    echo
    echo "Docker est prêt. Build :  docker build -t musetalk:latest ."
    exit 0
  fi
done

ko "dockerd n'a pas démarré (pod non privilégié) — voir /var/log/dockerd.log"
tail -n 15 /var/log/dockerd.log

# ---------- 3. Fallback : buildah ----------
log "Bascule sur buildah (build d'images sans démon)"
apt-get install -y -qq buildah podman fuse-overlayfs >/dev/null \
  && ok "buildah installé ($(buildah --version))" \
  || { ko "échec install buildah"; exit 1; }

mkdir -p /etc/containers
cat > /etc/containers/storage.conf <<'CONF'
[storage]
driver = "vfs"
runroot = "/var/run/containers/storage"
graphroot = "/var/lib/containers/storage"
CONF

cat <<'EOF'

============================================================
 dockerd est indisponible dans ce pod (pas de privilèges),
 mais buildah fait le même travail :

   buildah bud -t musetalk:latest -f Dockerfile .
   buildah push musetalk:latest docker://docker.io/<user>/musetalk:latest

 Pour tester l'image sur place :
   podman run --rm -it musetalk:latest bash
============================================================
EOF
