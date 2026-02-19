"""
Application de détection d'objets en temps réel via webcam.

Utilise le modèle YOLOv4-tiny avec OpenCV DNN pour détecter et labelliser
les objets visibles par la caméra. Affiche les bounding boxes colorées,
le nom de l'objet, le score de confiance et le FPS.

Contrôles :
  - Q ou ESC : quitter l'application
"""

import os
import sys
import time

import cv2
import numpy as np

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "yolov4-tiny.weights")
CONFIG_PATH = os.path.join(MODEL_DIR, "yolov4-tiny.cfg")
NAMES_PATH = os.path.join(MODEL_DIR, "coco.names")

CONFIDENCE_THRESHOLD = 0.6   # Seuil minimum de confiance pour afficher une détection
NMS_THRESHOLD = 0.4          # Seuil de Non-Maximum Suppression
INPUT_SIZE = (416, 416)      # Taille d'entrée du réseau YOLO

WINDOW_NAME = "Detection d'objets en temps reel - YOLOv4-tiny"


def generate_colors(num_classes: int) -> list:
    """Génère une palette de couleurs distinctes pour chaque classe."""
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    return [tuple(int(c) for c in color) for color in colors]


def load_model():
    """Charge le modèle YOLOv4-tiny et les noms de classes."""
    # Vérifier que les fichiers du modèle existent
    for path, name in [(WEIGHTS_PATH, "weights"), (CONFIG_PATH, "cfg"), (NAMES_PATH, "names")]:
        if not os.path.exists(path):
            print(f"Erreur : fichier '{name}' introuvable : {path}")
            print("Exécutez d'abord : python download_model.py")
            sys.exit(1)

    # Charger les noms de classes
    with open(NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Classes chargées : {len(class_names)} catégories")

    # Charger le réseau
    print("Chargement du modèle YOLOv4-tiny...")
    net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)

    # Utiliser le backend et la cible optimaux
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    print("Modèle chargé avec succès !")
    return net, class_names


def get_output_layers(net) -> list:
    """Récupère les noms des couches de sortie du réseau."""
    layer_names = net.getLayerNames()
    unconnected = net.getUnconnectedOutLayers()
    return [layer_names[i - 1] for i in unconnected.flatten()]


def detect_objects(net, frame, output_layers):
    """Effectue la détection d'objets sur une frame."""
    blob = cv2.dnn.blobFromImage(
        frame,
        scalefactor=1.0 / 255.0,
        size=INPUT_SIZE,
        swapRB=True,
        crop=False
    )
    net.setInput(blob)
    outputs = net.forward(output_layers)
    return outputs


def process_detections(outputs, frame_width, frame_height, class_names):
    """Traite les sorties du réseau et retourne les détections filtrées."""
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = float(scores[class_id])

            if confidence > CONFIDENCE_THRESHOLD:
                # YOLO retourne les coordonnées normalisées du centre + dimensions
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                w = int(detection[2] * frame_width)
                h = int(detection[3] * frame_height)

                # Coordonnées du coin supérieur gauche
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(confidence)
                class_ids.append(class_id)

    # Appliquer Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            results.append({
                "box": boxes[i],
                "confidence": confidences[i],
                "class_id": class_ids[i],
                "label": class_names[class_ids[i]],
            })

    return results


def draw_detections(frame, detections, colors):
    """Dessine les bounding boxes et les labels sur la frame."""
    for det in detections:
        x, y, w, h = det["box"]
        color = colors[det["class_id"] % len(colors)]
        confidence = det["confidence"]
        label = det["label"]

        # Dessiner la bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Préparer le texte du label
        text = f"{label} {confidence:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Fond du label
        label_y = max(y, text_h + 10)
        cv2.rectangle(
            frame,
            (x, label_y - text_h - 8),
            (x + text_w + 8, label_y + 4),
            color,
            cv2.FILLED
        )

        # Texte du label (blanc sur fond coloré)
        cv2.putText(
            frame,
            text,
            (x + 4, label_y - 2),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    return frame


def draw_hud(frame, fps, num_detections):
    """Affiche le HUD (FPS et nombre d'objets détectés)."""
    h, w = frame.shape[:2]

    # Fond semi-transparent pour le HUD
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (280, 80), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Texte FPS
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    # Nombre d'objets détectés
    cv2.putText(
        frame,
        f"Objets detectes: {num_detections}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 200, 255),
        2,
        cv2.LINE_AA
    )

    # Instructions en bas
    cv2.putText(
        frame,
        "Appuyez sur Q ou ESC pour quitter",
        (w // 2 - 180, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA
    )

    return frame


def main():
    """Boucle principale de l'application."""
    # Charger le modèle
    net, class_names = load_model()
    output_layers = get_output_layers(net)
    colors = generate_colors(len(class_names))

    # Ouvrir la webcam
    print("\nOuverture de la webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erreur : impossible d'accéder à la webcam.")
        print("Vérifiez que votre webcam est branchée et non utilisée par une autre application.")
        sys.exit(1)

    # Configurer la résolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Résolution webcam : {actual_w}x{actual_h}")
    print(f"\nDétection en cours... (appuyez sur Q ou ESC pour quitter)\n")

    # Créer la fenêtre
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1024, 576)

    # Variables pour le calcul du FPS
    fps = 0.0
    frame_count = 0
    fps_start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Erreur : impossible de lire la frame de la webcam.")
                break

            frame_height, frame_width = frame.shape[:2]

            # Détection d'objets
            outputs = detect_objects(net, frame, output_layers)
            detections = process_detections(outputs, frame_width, frame_height, class_names)

            # Dessiner les détections
            frame = draw_detections(frame, detections, colors)

            # Calculer le FPS (moyenne glissante)
            frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = time.time()

            # Afficher le HUD
            frame = draw_hud(frame, fps, len(detections))

            # Afficher la frame
            cv2.imshow(WINDOW_NAME, frame)

            # Vérifier les touches
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # Q ou ESC
                print("Fermeture de l'application...")
                break

    except KeyboardInterrupt:
        print("\nInterruption par l'utilisateur.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Application fermée.")


if __name__ == "__main__":
    main()
