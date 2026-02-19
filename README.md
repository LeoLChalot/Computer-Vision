# 🎥 Détection d'Objets en Temps Réel

Application Python de détection et labellisation d'objets en temps réel via webcam, utilisant **OpenCV** et le modèle **YOLOv4-tiny**.

## Stack Technique

| Composant        | Technologie         |
|------------------|---------------------|
| Vision           | OpenCV (DNN module) |
| Modèle           | YOLOv4-tiny         |
| Dataset          | COCO (80 classes)   |
| Langage          | Python 3.10+        |

## Installation

```bash
# 1. Créer et activer l'environnement virtuel
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Télécharger le modèle (~23 MB)
python download_model.py
```

## Lancer l'application

```bash
python object_detection.py
```

## Contrôles

| Touche    | Action                    |
|-----------|---------------------------|
| `Q`       | Quitter                   |
| `ESC`     | Quitter                   |

## Objets détectables (80 classes COCO)

Personnes, véhicules (voiture, bus, camion, vélo, moto), animaux (chien, chat, oiseau, cheval),
objets du quotidien (téléphone, ordinateur portable, tasse, bouteille, clavier, souris, livre, ciseaux),
meubles (chaise, canapé, lit, table), et bien d'autres.
