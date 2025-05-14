# app/services/visualizer.py (NEUE DATEI oder Ergänzung in analyzer.py)
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models
import pickle
from io import BytesIO
from http.client import HTTPException

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from app.config import get_settings  # Wiederverwenden der Settings für Pfade

settings = get_settings()

# --- Konfiguration und Modell laden (ähnlich deinem Skript) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
NUM_CLASSES = None  # Wird aus dem Pickle geladen

# Pfad zum Modell anpassen, falls er sich vom Analyse-Modell unterscheidet
# Hier nehmen wir an, es ist das gleiche Modell wie für die Analyse
VIS_MODEL_PATH = settings.MODEL_PATH  # Oder dein 'advanced_ml_project_model_vF.pkl' Pfad


def load_visualization_model():
    global MODEL, NUM_CLASSES
    if MODEL is not None:
        return

    try:
        # Lade das Modell für die Visualisierung
        # ANNAHME: Das .pkl enthält ein Dictionary mit 'state_dict' und 'num_classes'
        with open(VIS_MODEL_PATH, 'rb') as f:
            model_info = pickle.load(f)

        NUM_CLASSES = model_info['num_classes']

        # Stelle sicher, dass dein 'advanced_ml_project_model_vF.pkl' so geladen wird, wie es gespeichert wurde.
        # Wenn es das gesamte Modell ist:
        # MODEL = pickle.load(f)
        # Wenn es state_dict ist:
        model_instance = models.resnet18(weights=None)  # Oder pretrained=False für ältere torchvision Versionen
        model_instance.fc = torch.nn.Linear(model_instance.fc.in_features, NUM_CLASSES)
        model_instance.load_state_dict(model_info['state_dict'])
        MODEL = model_instance
        MODEL.to(DEVICE).eval()
        print(f"Visualisierungsmodell '{VIS_MODEL_PATH}' geladen mit {NUM_CLASSES} Klassen.")

    except FileNotFoundError:
        print(f"Fehler: Modelldatei '{VIS_MODEL_PATH}' nicht gefunden.")
        # Hier könntest du eine spezifischere Exception werfen oder None zurückgeben
        raise
    except Exception as e:
        print(f"Fehler beim Laden des Visualisierungsmodells: {e}")
        raise


# Lade das Modell beim Start des Moduls (oder lazy in der Funktion)
load_visualization_model()


def generate_grad_cam_overlay(image_bytes: bytes) -> bytes:
    """
    Erzeugt ein Bild mit Grad-CAM++ Overlay und Bounding Boxes.
    Gibt das Bild als Bytes (PNG-Format) zurück.
    """
    if MODEL is None:
        raise RuntimeError("Visualisierungsmodell nicht geladen.")

    try:
        # ========== BILDVORVERARBEITUNG ==========
        pil_image = Image.open(BytesIO(image_bytes)).convert('RGB')

        transform_pipeline = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # Ggf. Normalisierung hinzufügen, falls beim Training verwendet
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform_pipeline(pil_image).unsqueeze(0).to(DEVICE)
        # Konvertiere das für die Überlagerung genutzte Bild zu float im Bereich [0, 1]
        rgb_image_for_overlay = np.array(pil_image.resize((224, 224))) / 255.0

        # ========== Grad-CAM++ ==========
        target_layers = [MODEL.layer4[-1]]  # Letzter Conv-Block von ResNet18
        cam_algorithm = GradCAMPlusPlus(model=MODEL, target_layers=target_layers)

        # Klasse mit höchster Wahrscheinlichkeit bestimmen
        with torch.no_grad():
            output = MODEL(input_tensor)
            # Wenn dein Modell Logits ausgibt, ist argmax korrekt.
            # Wenn es bereits Wahrscheinlichkeiten (nach Softmax) sind, ist es auch ok.
            class_idx = output.argmax(dim=1).item()
            # Optional: Hole den Klassennamen, wenn du einen LabelEncoder hast
            # class_name = label_encoder.inverse_transform([class_idx])[0]

        targets_for_cam = [ClassifierOutputTarget(class_idx)]
        grayscale_cam = cam_algorithm(input_tensor=input_tensor, targets=targets_for_cam)[0]  # (H, W), Werte 0-1

        # ========== HEATMAP ÜBERLAGERN ==========
        # Wichtig: use_rgb=True, wenn rgb_image_for_overlay im RGB-Format ist
        heatmap_on_image = show_cam_on_image(rgb_image_for_overlay, grayscale_cam, use_rgb=True)

        # ========== BOUNDING BOXES ==========
        threshold_value = 0.7  # Fester Threshold
        binary_map = (grayscale_cam > threshold_value).astype(np.uint8) * 255

        kernel = np.ones((3, 3), np.uint8)  # Für Erosion
        cleaned_map = cv2.erode(binary_map, kernel, iterations=1)

        contours, _ = cv2.findContours(cleaned_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Kopiere das Bild mit Heatmap, um darauf die Boxen zu zeichnen
        # Stelle sicher, dass es ein 8-bit Bild ist (0-255)
        overlay_with_boxes = np.uint8(
            heatmap_on_image * 255).copy() if heatmap_on_image.max() <= 1.0 else heatmap_on_image.copy()

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            padding = 2
            x_pad = max(x + padding, 0)
            y_pad = max(y + padding, 0)
            w_pad = max(w - 2 * padding, 1)
            h_pad = max(h - 2 * padding, 1)
            cv2.rectangle(overlay_with_boxes, (x_pad, y_pad), (x_pad + w_pad, y_pad + h_pad), (0, 255, 0),
                          2)  # Grüne Box

        # Konvertiere das finale Bild (numpy array) in Bytes
        final_image_pil = Image.fromarray(overlay_with_boxes)
        img_byte_arr = BytesIO()
        final_image_pil.save(img_byte_arr, format='PNG')  # Speichere als PNG
        img_byte_arr = img_byte_arr.getvalue()

        return img_byte_arr

    except Exception as e:
        print(f"Fehler bei der Grad-CAM Erzeugung: {e}")
        # Hier könntest du eine spezifische Exception für die API werfen
        raise HTTPException(status_code=500, detail=f"Fehler bei der Bildvisualisierung: {str(e)}")