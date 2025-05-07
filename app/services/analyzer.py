# app/services/analyzer.py
import os
import time
import pickle  # Für LabelEncoder UND jetzt auch für das Modell-state_dict
import logging
from io import BytesIO  # Um Bytes als Datei zu behandeln
from typing import Dict, Tuple, List, Optional  # Optional hinzugefügt

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
# import numpy as np # Nicht direkt in diesem Ausschnitt verwendet, kann ggf. entfernt werden, wenn nirgends anders genutzt
from sklearn.preprocessing import LabelEncoder  # Import LabelEncoder

from app.models.plant_analysis import PlantAnalysisResponse, PlantStatus, AnalysisResult
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Oder settings.LOG_LEVEL, falls du das konfigurierbar machen möchtest

# --- Modell- und Vorverarbeitungslogik (adaptiert aus Trainingsskript) ---

# Definiere die Transformationen genau wie im Training
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    # Fügen Sie hier ggf. Normalisierung hinzu, falls im Training verwendet
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Stelle sicher, dass dies konsistent mit dem Training ist
])


# Mapping von vorhergesagten Klassen zu PlantStatus (Beispiel)
def map_label_to_status(label: str) -> PlantStatus:
    label_lower = label.lower()
    if "healthy" in label_lower:
        return PlantStatus.HEALTHY
    else:
        # Alle anderen Klassen werden als UNHEALTHY eingestuft
        return PlantStatus.UNHEALTHY


# example Empfehlungeen
def get_details_recommendations(label: str) -> tuple[str, Optional[str]]:
    label_lower = label.lower()
    if "bacterial_spot" in label_lower:
        return "Bacterial spot detected.", "Apply copper-based fungicides. Ensure proper plant spacing."
    elif "early_blight" in label_lower:
        return "Signs of early blight.", "Remove affected leaves. Apply fungicide if severe. Improve air circulation."
    elif "late_blight" in label_lower:
        return "Late blight detected.", "Apply fungicide immediately. Destroy severely infected plants."
    elif "leaf_mold" in label_lower:
        return "Leaf mold detected.", "Improve ventilation. Apply appropriate fungicide."
    elif "target_spot" in label_lower:
        return "Target spot detected.", "Remove infected leaves. Ensure proper watering practices."
    elif "healthy" in label_lower:
        return "Plant appears healthy. No significant issues detected.", "Continue regular care and monitoring."
    elif "tomato__yellow_leaf_curl_virus" in label_lower:
        return "Tomato Yellow Leaf Curl Virus detected.", "Remove and destroy infected plants. Control whitefly populations."
    elif "tomato__mosaic_virus" in label_lower:
        return "Tomato Mosaic Virus detected.", "Remove infected plants. Practice good sanitation. Control aphids."
    elif "tomato__spider_mites" in label_lower:
        return "Spider mites (Two-spotted spider mite) detected.", "Use miticides or insecticidal soap. Increase humidity."
    elif "strawberry___leaf_scorch" in label_lower:
        return "Strawberry leaf scorch detected.", "Apply appropriate fungicides. Ensure good air circulation and remove infected debris."
    elif "pepper_bell__bacterial_spot" in label_lower:
        return "Pepper bell bacterial spot detected.", "Similar to tomato bacterial spot, use copper fungicides and ensure spacing."
    elif "chili__leaf_curl" in label_lower:
        return "Chili leaf curl detected.", "Often viral; control insect vectors like aphids and whiteflies. Remove severely affected plants."
    elif "chili__yellowish" in label_lower:
        return "Chili plant shows yellowish discoloration.", "Check for nutrient deficiencies (e.g., nitrogen, iron) or potential diseases. Soil testing might be needed."
    elif "chili__leaf_spot" in label_lower:
        return "Chili leaf spot detected.", "Apply fungicides. Improve air circulation. Avoid overhead watering."
    elif "chili__whitefly" in label_lower:
        return "Whiteflies detected on chili plant.", "Use sticky traps, insecticidal soaps, or appropriate insecticides. Encourage natural predators."
    elif "cucumber__diseased" in label_lower:
        return "Cucumber plant appears diseased.", "Further diagnosis needed. Check for common cucumber diseases like downy mildew, powdery mildew, or angular leaf spot."
    else: # Fallback
        return f"Analysis result: {label}", "Consult specific plant care resources for potential issues. Ensure the label mapping is comprehensive."


class PlantAnalysisModel:
    """Lädt das trainierte PyTorch-Modell und den LabelEncoder."""

    def __init__(self, model_path: str, label_encoder_path: str):
        self.model_path = model_path
        self.label_encoder_path = label_encoder_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.label_encoder = None
        self.num_classes = None  # Wird vom LabelEncoder abgeleitet
        self._load_artifacts()

    def _load_artifacts(self):
        logger.info(
            f"Versuche Artefakte zu laden. Modellpfad: {self.model_path}, LabelEncoder-Pfad: {self.label_encoder_path}")
        try:
            # Lade LabelEncoder
            if not os.path.exists(self.label_encoder_path):
                raise FileNotFoundError(f"LabelEncoder-Datei nicht gefunden unter {self.label_encoder_path}")
            with open(self.label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            self.num_classes = len(self.label_encoder.classes_)
            logger.info(
                f"LabelEncoder erfolgreich geladen. Anzahl Klassen: {self.num_classes}. Klassen: {self.label_encoder.classes_}")

            # Initialisiere Modellarchitektur (muss der im Training entsprechen!)
            # Verwende resnet18 als Basis, wie im Trainingsskript
            self.model = models.resnet18(weights=None)  # Moderner Weg für pretrained=False
            # Passe die letzte Schicht an die Anzahl der Klassen an
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.num_classes)
            logger.info(f"Modellarchitektur (ResNet18) initialisiert für {self.num_classes} Klassen.")

            # Lade den trainierten Zustand (state_dict) MIT PICKLE ZUERST
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Modelldatei nicht gefunden unter {self.model_path}")

            logger.info(f"Versuche, Modelldatei '{self.model_path}' mit pickle.load() zu laden...")
            loaded_object_from_pickle = None
            with open(self.model_path, 'rb') as f:
                # ACHTUNG: Wenn das Modell mit torch.save(model, filepath) gespeichert wurde (ganzes Modell, nicht state_dict),
                # dann wäre torch.load() hier besser. Aber da wir pickle.dump(model.state_dict(), ..) hatten:
                loaded_object_from_pickle = pickle.load(f)

            logger.info(
                f"Modelldatei erfolgreich mit pickle.load() geladen. Typ des geladenen Objekts: {type(loaded_object_from_pickle)}")

            actual_state_dict = None
            if isinstance(loaded_object_from_pickle, dict):
                # Fall 1: Das geladene Objekt ist bereits der state_dict (flache Struktur von Tensoren)
                # Überprüfe, ob die Keys typische Layer-Namen sind und nicht "state_dict" selbst ein Key ist.
                if "state_dict" not in loaded_object_from_pickle and all(
                        isinstance(v, torch.Tensor) for v in loaded_object_from_pickle.values()):
                    logger.info("Geladenes Objekt scheint bereits ein flacher state_dict zu sein.")
                    actual_state_dict = loaded_object_from_pickle
                # Fall 2: Das geladene Objekt ist ein Dictionary, das den state_dict unter dem Key "state_dict" enthält
                elif "state_dict" in loaded_object_from_pickle and isinstance(loaded_object_from_pickle["state_dict"],
                                                                              dict):
                    logger.info(
                        "Gefunden: 'state_dict'-Key im geladenen Objekt. Verwende diesen verschachtelten state_dict.")
                    actual_state_dict = loaded_object_from_pickle["state_dict"]
                    # Optional: Überprüfe num_classes, falls im Pickle gespeichert
                    if "num_classes" in loaded_object_from_pickle:
                        pickled_num_classes = loaded_object_from_pickle['num_classes']
                        logger.info(f"Gefunden: 'num_classes'-Key im geladenen Objekt: {pickled_num_classes}")
                        if pickled_num_classes != self.num_classes:
                            logger.warning(
                                f"Anzahl Klassen im Pickle ({pickled_num_classes}) stimmt nicht mit LabelEncoder ({self.num_classes}) überein!")
                            # Hier könntest du entscheiden, ob du abbrichst oder mit den Klassen vom LabelEncoder weitermachst
                else:
                    logger.error(
                        f"Geladenes Dictionary hat eine unerwartete Struktur. Keys: {list(loaded_object_from_pickle.keys())[:10]}...")
                    raise ValueError(
                        "Konnte keinen gültigen state_dict aus der Struktur des geladenen Dictionarys extrahieren.")
            else:
                # Fall 3: Das geladene Objekt ist kein Dictionary. Vielleicht wurde das gesamte Modell gepickelt?
                # Dies ist für `model.load_state_dict()` nicht direkt verwendbar, es sei denn, es ist ein PyTorch-Modul.
                # Wenn es ein komplettes Modell ist, das mit `pickle.dump(model, ...)` gespeichert wurde:
                if isinstance(loaded_object_from_pickle, nn.Module):
                    logger.info(
                        "Geladenes Objekt scheint ein komplettes nn.Module zu sein. Versuche, es direkt zu verwenden.")
                    self.model = loaded_object_from_pickle  # Hier würde das gesamte Modell ersetzt
                    # Stelle sicher, dass die Anzahl der Klassen des geladenen Modells passt, wenn es ein fc-Layer hat
                    if hasattr(self.model, 'fc') and isinstance(self.model.fc, nn.Linear):
                        if self.model.fc.out_features != self.num_classes:
                            logger.warning(
                                f"Anzahl Output-Features des geladenen Modells ({self.model.fc.out_features}) "
                                f"stimmt nicht mit LabelEncoder ({self.num_classes}) überein! Dies wird Probleme verursachen.")
                    # In diesem Fall ist load_state_dict nicht mehr nötig, da das Modell bereits geladen ist.
                    actual_state_dict = "MODEL_LOADED_DIRECTLY"  # Platzhalter, um den nächsten Schritt zu überspringen
                else:
                    raise TypeError(
                        f"Geladene Modelldaten sind weder ein Dictionary (state_dict) noch ein nn.Module, sondern {type(loaded_object_from_pickle)}")

            if actual_state_dict is None:
                raise ValueError(
                    "Konnte keinen gültigen state_dict aus der geladenen Datei extrahieren oder das Modell direkt laden.")

            if actual_state_dict != "MODEL_LOADED_DIRECTLY":  # Nur wenn wir einen state_dict haben und nicht das ganze Modell
                logger.info(
                    f"Versuche nun, das extrahierte state_dict (Typ: {type(actual_state_dict)}) in das Modell zu laden...")
                self.model.load_state_dict(actual_state_dict)

            self.model.to(self.device)
            self.model.eval()
            logger.info(f"PyTorch-Modell erfolgreich aus '{self.model_path}' geladen und auf {self.device} verschoben.")

        except FileNotFoundError as e:
            logger.error(f"Fehler beim Laden der Artefakte: Datei nicht gefunden. {e}", exc_info=True)
            raise RuntimeError(f"Datei nicht gefunden: {e}")
        except pickle.UnpicklingError as e:
            logger.error(
                f"Fehler beim Entpickeln der Datei '{self.model_path}': {e}. Datei könnte korrupt sein oder nicht im Pickle-Format.",
                exc_info=True)
            raise RuntimeError(f"Fehler beim Entpickeln der Modelldatei: {e}")
        except TypeError as e:
            logger.error(f"Typfehler beim Laden des state_dict oder Modells: {e}", exc_info=True)
            raise RuntimeError(f"Typfehler beim Laden des state_dict oder Modells: {e}")
        except Exception as e:  # Fängt auch RuntimeError von load_state_dict ab (z.B. Missing keys)
            logger.error(f"Allgemeiner Fehler beim Laden der Modell-Artefakte: {str(e)}", exc_info=True)
            raise RuntimeError(f"Fehler beim Laden der Modell-Artefakte: {str(e)}")

    def predict(self, image_bytes: bytes) -> Tuple[Dict, float]:
        """Führt die Vorhersage basierend auf den Bild-Bytes durch."""
        if not self.model or not self.label_encoder:
            # Diese Prüfung sollte idealerweise schon in _load_artifacts fehlschlagen, falls etwas nicht stimmt
            raise RuntimeError("Modell oder LabelEncoder nicht korrekt initialisiert.")

        try:
            # Öffne Bild aus Bytes
            image = Image.open(BytesIO(image_bytes)).convert("RGB")

            # Wende Transformationen an
            image_tensor = TRANSFORM(image).unsqueeze(0)  # Füge Batch-Dimension hinzu
            image_tensor = image_tensor.to(self.device)

            # Inferenz
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                class_index = predicted_idx.item()
                confidence_score = confidence.item()

            # Konvertiere Index zurück zu Label
            predicted_label = self.label_encoder.inverse_transform([class_index])[0]

            # Erhalte Status, Details und Empfehlungen
            status = map_label_to_status(predicted_label)
            details, recommendations = get_details_recommendations(predicted_label)

            prediction_result = {
                "status": status,
                "details": details,
                "recommendations": recommendations
            }

            return prediction_result, confidence_score

        except Exception as e:
            logger.error(f"Fehler während der Modellvorhersage: {str(e)}", exc_info=True)
            # Hier könntest du einen spezifischeren Fehler für die API zurückgeben
            raise RuntimeError(f"Vorhersage fehlgeschlagen: {str(e)}")


class PlantAnalyzerService:
    """Service für die Analyse von Pflanzenbildern."""

    def __init__(self):
        # Das Laden der Artefakte geschieht jetzt im Konstruktor von PlantAnalysisModel
        self.model_service = PlantAnalysisModel(  # Benenne die Instanzvariable um, um Verwechslung zu vermeiden
            model_path=settings.MODEL_PATH,
            label_encoder_path=settings.LABEL_ENCODER_PATH
        )
        # Stelle sicher, dass das Upload-Verzeichnis existiert (obwohl es hier nicht mehr primär genutzt wird)
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    async def analyze_image(self, image_bytes: bytes) -> AnalysisResult:
        """
        Analysiert Pflanzenbild-Bytes und gibt das Ergebnis zurück.
        """
        start_time = time.time()

        try:
            # Metadaten aus Bytes extrahieren (Breite/Höhe)
            temp_img = Image.open(BytesIO(image_bytes))
            width, height = temp_img.size
            temp_img.close()  # Schließe das temporäre Bild

            # Führe Analyse mit dem Modell durch
            # Greife auf die predict-Methode der PlantAnalysisModel-Instanz zu
            prediction_dict, confidence = self.model_service.predict(image_bytes)

            # Erstelle das Response-Objekt
            plant_analysis = PlantAnalysisResponse(
                status=prediction_dict["status"],
                details=prediction_dict["details"],
                recommendations=prediction_dict.get("recommendations")
            )

            processing_time = int((time.time() - start_time) * 1000)

            return AnalysisResult(
                plant_analysis=plant_analysis,
                confidence_score=confidence,
                processing_time_ms=processing_time,
                image_width=width,
                image_height=height
            )

        except Exception as e:
            logger.error(f"Fehler im Image Analysis Service: {str(e)}", exc_info=True)
            # Es ist oft besser, hier eine spezifischere Exception oder einen Fehlercode zurückzugeben,
            # statt einer generischen RuntimeError, die die API-Antwort beeinflusst.
            # Für den Moment belassen wir es bei RuntimeError, um die Kette nicht zu unterbrechen.
            raise RuntimeError(f"Servicefehler während der Bildanalyse: {str(e)}")


# Singleton Instanz des Service erstellen
# Diese Zeile führt dazu, dass _load_artifacts beim Import des Moduls aufgerufen wird.
analyzer_service = PlantAnalyzerService()