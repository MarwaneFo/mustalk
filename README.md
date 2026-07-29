# MuseTalk — image Docker

Image GPU prête à l'emploi pour [MuseTalk](https://github.com/TMElyralab/MuseTalk)
(CUDA 11.8 / PyTorch 2.0.1 / Python 3.10 — la configuration officielle du projet).

## Pourquoi GitHub Actions et pas le pod RunPod ?

Un pod RunPod est lui-même un conteneur non privilégié : pas de `CAP_SYS_ADMIN`,
`mount --bind` et `unshare` refusés. Docker s'y installe et le démon démarre, mais
**aucune étape `RUN` ne peut s'exécuter** — donc aucun build possible (buildah et
podman butent sur la même limite). Le build se fait donc sur un runner GitHub, et
RunPod récupère l'image finie depuis le registre.

## Build

Le workflow [`.github/workflows/build.yml`](.github/workflows/build.yml) se déclenche
à chaque push sur `main`, ou manuellement via l'onglet **Actions**.

Il publie sur GHCR :

```
ghcr.io/<owner>/musetalk:latest
ghcr.io/<owner>/musetalk:v1.5
```

Entrée `download_weights` :

| Valeur | Taille | Usage |
|---|---|---|
| `true` (défaut) | ~20 Go | image autonome — template RunPod, Serverless |
| `false` | ~9 Go | poids montés au runtime sur `/opt/MuseTalk/models` |

## Utilisation sur RunPod

1. **Templates** → *New Template*
2. *Container Image* : `ghcr.io/<owner>/musetalk:v1.5`
3. *Container Disk* : 30 Go minimum
4. *Volume Mount Path* : `/workspace` (pour retrouver tes données et avatars)
5. Si le package GHCR est privé : renseigner les *Container Registry Credentials*
   (utilisateur GitHub + un PAT avec le scope `read:packages`), ou rendre le package
   public depuis la page du package sur GitHub.

## Contenu de l'image

| Chemin | Contenu |
|---|---|
| `/opt/MuseTalk` | code source |
| `/opt/MuseTalk/models` | poids (musetalk v1.0 + v1.5, syncnet, sd-vae, whisper, dwpose, face-parse) |
| `/usr/local/bin/download_weights.sh` | (re)télécharge les poids |
| `/usr/local/bin/entrypoint.sh` | vérifie CUDA et la présence des poids |

## Test local

```bash
docker run --rm --gpus all -it \
  -v /chemin/vers/tes/donnees:/workspace \
  ghcr.io/<owner>/musetalk:v1.5 bash
```
