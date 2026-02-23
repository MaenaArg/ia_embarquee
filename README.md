# ia_embarquee
# IA Embarquée — Reconnaissance de chiffres manuscrits sur Raspberry Pi 5

> Projet JUNIA AP5 · 2025-2026

Application d'intelligence artificielle embarquée capable de détecter et reconnaître des chiffres manuscrits (0-9) en temps réel à partir d'une caméra, déployée sur Raspberry Pi 5 avec un moteur d'inférence écrit en C.

---

## Architecture du projet

```
ia_embarquee/
|-- docker/           # Dockerfile, run.sh -> environnement OpenCV embarqué
|-- training/         # Scripts d'entraînement MNIST, MLP et CNN + data augmentée (MLP et CNN)
|-- models/           # Poids exportés (.txt) utilisés par le moteur C
|-- inference_c/      # Moteur d'inférence C + application caméra (main.cpp)
|-- utils/            # Benchmarks Python (numpy et pur) + script de convertion dataset perso en type MNIST (Python) + résultats de prédictions, etc..
|-- data/             # Dataset personnel + version convertie en MNIST
|-- docs/             # Rapport PDF + Jeu de test de fin de projet avec caméra (MLP)
```

---

## Résultats

| Implémentation | Temps / image | Précision dataset perso | Précision MNIST |
|---|---|---|---|
| Python pur | 19,3 ms | 90,0% | 96,7% |
| Python numpy | 0,05 ms | 90,0% | 96,7% |
| **C optimisé** | **0,23 ms** | **90,0%** | **96,7%** |
| **Caméra temps réel** | **3–5 ms** | — | — |

**FPS caméra : 30 images/s · Seuil de binarisation : 20**

---

## Pipeline complet

1. Le script `run.sh` démarre la capture vidéo et lance le conteneur Docker
2. Le conteneur Docker reçoit le flux, applique la chaîne de prétraitement OpenCV
3. Le moteur d'inférence C effectue le forward pass MLP en 0,23 ms
4. Le résultat (chiffre + confiance + FPS) est incrustré sur la frame et renvoyé

---

## Démarrage rapide

### Prérequis

- Raspberry Pi 5 avec Raspberry Pi OS 64-bit
- Docker installé
- Caméra Module V3 connectée

### Lancement

```bash
cd docker
./run.sh all       # Build Docker + démarrage caméra + lancement conteneur
./run.sh logs      # Vérifier les logs
./run.sh stop      # Arrêt
```

### Visualisation distante

```bash
# Depuis un PC sur le même réseau
vlc "tcp://<IP_RASPBERRY>:8554"
```
