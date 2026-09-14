"""
Corrige, si besoin, l'import obsolete de basicsr.

basicsr importe torchvision.transforms.functional_tensor, supprime depuis
torchvision 0.17. On corrige l'import plutot que de retrograder torchvision,
dont dependent MuseTalk et la pile OpenMMLab.

Ce script ne doit JAMAIS echouer : si basicsr a deja corrige l'import dans
sa version courante, il n'y a rien a faire et le build doit continuer. Une
version precedente utilisait grep, qui renvoie 1 quand il ne trouve rien --
et cassait la chaine && du Dockerfile.
"""
import os
import site
import sys

ANCIEN = "torchvision.transforms.functional_tensor"
NOUVEAU = "torchvision.transforms.functional"

racines = []
for d in site.getsitepackages():
    c = os.path.join(d, "basicsr")
    if os.path.isdir(c):
        racines.append(c)

if not racines:
    print("basicsr introuvable -- rien a corriger")
    sys.exit(0)

corriges = 0
for racine in racines:
    for dossier, _, fichiers in os.walk(racine):
        for f in fichiers:
            if not f.endswith(".py"):
                continue
            p = os.path.join(dossier, f)
            try:
                s = open(p, encoding="utf-8").read()
            except (UnicodeDecodeError, OSError):
                continue
            if ANCIEN in s:
                open(p, "w", encoding="utf-8").write(s.replace(ANCIEN, NOUVEAU))
                corriges += 1
                print("corrige :", os.path.relpath(p, racine))

print(f"{corriges} fichier(s) corrige(s)")
