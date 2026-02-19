"""
Télécharge les fichiers du modèle YOLOv4-tiny nécessaires pour la détection d'objets.

Fichiers téléchargés dans le dossier 'model/' :
  - yolov4-tiny.weights  (~23 MB)
  - yolov4-tiny.cfg
  - coco.names (80 classes d'objets)
"""

import os
import urllib.request
import sys

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

FILES = {
    "yolov4-tiny.weights": "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights",
    "yolov4-tiny.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
    "coco.names": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names",
}


def download_file(url: str, dest_path: str) -> None:
    """Télécharge un fichier avec une barre de progression."""
    filename = os.path.basename(dest_path)

    if os.path.exists(dest_path):
        print(f"  ✓ {filename} existe déjà, téléchargement ignoré.")
        return

    print(f"  ⬇ Téléchargement de {filename}...")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r    {percent:5.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=progress_hook)
        print(f"\n  ✓ {filename} téléchargé avec succès.")
    except Exception as e:
        print(f"\n  ✗ Erreur lors du téléchargement de {filename}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise


def main():
    print("=" * 50)
    print("  Téléchargement du modèle YOLOv4-tiny")
    print("=" * 50)

    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"\nDossier de destination : {MODEL_DIR}\n")

    for filename, url in FILES.items():
        dest_path = os.path.join(MODEL_DIR, filename)
        download_file(url, dest_path)

    print("\n" + "=" * 50)
    print("  ✓ Tous les fichiers sont prêts !")
    print("=" * 50)


if __name__ == "__main__":
    main()
