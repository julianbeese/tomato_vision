# /Users/julianbeese/Developer/NOVA/tomato_vision/test/maintest.py
import pytest
import sys
from pathlib import Path
from PIL import Image
import io
import pickle
import torch  # Hinzugefügt für isinstance(..., torch.Tensor)

# Füge das App-Verzeichnis zum Python-Pfad hinzu
APP_DIR = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(APP_DIR.parent))  # Fügt 'tomato_vision' hinzu

# Importiere die notwendigen Klassen/Funktionen
try:
    from app.services.analyzer import PlantAnalyzerService
    from app.config import get_settings
    from app.models.plant_analysis import AnalysisResult, PlantAnalysisResponse
except ImportError as e:
    print(f"Fehler beim Importieren von App-Modulen: {e}")
    print("Stelle sicher, dass die Testdatei im richtigen Verzeichnis liegt und sys.path korrekt gesetzt ist.")
    pytest.skip("App-Module konnten nicht importiert werden.", allow_module_level=True)

# Markiere alle Tests in dieser Datei als async für pytest-asyncio
pytestmark = pytest.mark.asyncio

# --- Test Setup ---
settings = None
MODEL_PATH_FOR_TEST = None
IMAGES_TO_TEST = []  # Liste für die zu testenden Bilder

try:
    settings = get_settings()
    # KORREKTER PFAD ZUM VERZEICHNIS MIT DEN TESTBILDERN
    # Annahme: Bilder sind in app/data/test/images/
    ACTUAL_TEST_IMAGE_DIR = APP_DIR / "data" / "test" / "images"
    MODEL_PATH_FOR_TEST = Path(settings.MODEL_PATH)

    if not ACTUAL_TEST_IMAGE_DIR.is_dir():
        pytest.skip(f"Testbild-Verzeichnis nicht gefunden unter {ACTUAL_TEST_IMAGE_DIR}", allow_module_level=True)
    else:
        # Sammle alle JPG, JPEG, PNG Bilder im Verzeichnis
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        for ext in image_extensions:
            IMAGES_TO_TEST.extend(list(ACTUAL_TEST_IMAGE_DIR.glob(ext)))

        if not IMAGES_TO_TEST:
            pytest.skip(f"Keine Testbilder (jpg, png) in {ACTUAL_TEST_IMAGE_DIR} gefunden.", allow_module_level=True)
        else:
            print(f"Gefundene Testbilder ({len(IMAGES_TO_TEST)}):")
            for img_path in IMAGES_TO_TEST[:5]:  # Zeige die ersten 5
                print(f"  - {img_path.name}")
            if len(IMAGES_TO_TEST) > 5:
                print(f"  ... und {len(IMAGES_TO_TEST) - 5} weitere.")

    if not MODEL_PATH_FOR_TEST.is_file():
        pytest.skip(f"Modelldatei nicht gefunden unter {MODEL_PATH_FOR_TEST}", allow_module_level=True)

except Exception as e:
    pytest.skip(f"Setup fehlgeschlagen: {e}", allow_module_level=True)


# --- Testfunktion für direktes Laden des Modells mit Pickle ---
def test_load_model_directly_with_pickle():
    """
    Testet, ob die Modelldatei direkt mit pickle.load() geladen werden kann.
    """
    print(f"\nTeste direktes Laden der Modelldatei: {MODEL_PATH_FOR_TEST}")
    assert MODEL_PATH_FOR_TEST is not None, "Modellpfad wurde nicht im Setup initialisiert."
    assert MODEL_PATH_FOR_TEST.is_file(), f"Modelldatei {MODEL_PATH_FOR_TEST} existiert nicht."

    try:
        with open(MODEL_PATH_FOR_TEST, "rb") as f:
            model_data = pickle.load(f)
        assert model_data is not None, "Pickle.load() hat None zurückgegeben."
        print(f"Modelldatei erfolgreich mit pickle.load() geladen.")
        print(f"Typ der geladenen Daten: {type(model_data)}")
        assert isinstance(model_data, dict), "Geladene Modelldaten sind kein Dictionary (erwartet für state_dict)."
        print(f"Anzahl der Keys im (möglicherweise äußeren) Dictionary: {len(model_data.keys())}")
        if "state_dict" in model_data and isinstance(model_data["state_dict"], dict):
            print(f"Anzahl der Keys im inneren 'state_dict': {len(model_data['state_dict'].keys())}")
        elif all(isinstance(v, torch.Tensor) for v in model_data.values()):
            print(
                f"Das geladene Dictionary scheint bereits der flache state_dict zu sein. Anzahl Keys: {len(model_data.keys())}")

    except pickle.UnpicklingError as e:
        pytest.fail(
            f"Fehler beim Entpickeln der Modelldatei (pickle.UnpicklingError): {e} - Datei könnte korrupt sein oder kein Pickle-Format haben.")
    except FileNotFoundError:
        pytest.fail(f"Modelldatei nicht gefunden unter: {MODEL_PATH_FOR_TEST}")
    except Exception as e:
        pytest.fail(f"Ein unerwarteter Fehler ist beim direkten Laden der Modelldatei aufgetreten: {e}")


# --- Parametrisierte asynchrone Testfunktion für die vollständige Analyse ---
@pytest.mark.dependency(depends=["test_load_model_directly_with_pickle"])
@pytest.mark.parametrize("image_path_to_test", IMAGES_TO_TEST, ids=[img.name for img in IMAGES_TO_TEST])
async def test_analyze_image_from_data_folder_async(image_path_to_test: Path):
    """
    Testet das Laden eines Bildes, die Analyse und gibt das Ergebnis sowie LabelEncoder-Klassen aus.
    Wird für jedes Bild im Testverzeichnis ausgeführt.
    """
    print(f"\n--- Teste Bild: {image_path_to_test.name} ---")
    assert image_path_to_test.is_file(), f"Testbild {image_path_to_test} nicht gefunden oder ist keine Datei."

    try:
        with open(image_path_to_test, "rb") as f:
            image_bytes = f.read()
        assert image_bytes is not None, "Bild konnte nicht als Bytes geladen werden."
        print(f"Bild erfolgreich als Bytes geladen (Größe: {len(image_bytes)} Bytes).")

        # Service wird hier für jeden Test (jedes Bild) neu initialisiert.
        # Für viele Bilder könnte man überlegen, den Service in einem Fixture mit passendem Scope zu erstellen.
        print("Initialisiere PlantAnalyzerService...")
        analyzer = None
        try:
            analyzer = PlantAnalyzerService()
            assert analyzer.model_service is not None, "Model-Service (PlantAnalysisModel) im AnalyzerService wurde nicht initialisiert."
            assert analyzer.model_service.model is not None, "PyTorch-Modell im PlantAnalysisModel wurde nicht geladen."
            assert analyzer.model_service.label_encoder is not None, "LabelEncoder im PlantAnalysisModel wurde nicht geladen."
            print("PlantAnalyzerService erfolgreich initialisiert.")

            # Ausgabe der LabelEncoder-Klassen (könnte in ein Fixture, wenn es nicht für jedes Bild wiederholt werden soll)
            # Für diesen Test lassen wir es pro Bild, um sicherzustellen, dass der Service jedes Mal korrekt initialisiert.
            print("\n--- LabelEncoder Klassen (vom aktuellen Analyzer) ---")
            if hasattr(analyzer.model_service.label_encoder, 'classes_'):
                le_classes = analyzer.model_service.label_encoder.classes_
                print(f"Vom LabelEncoder geladene Klassen ({len(le_classes)}):")
                for i, cls_name in enumerate(le_classes):
                    print(f"  Index {i}: '{cls_name}'")
            else:
                print("Konnte 'classes_' Attribut im LabelEncoder nicht finden.")
            print("--- Ende LabelEncoder Klassen ---\n")

        except Exception as e:
            pytest.fail(
                f"Fehler bei der Initialisierung des PlantAnalyzerService oder beim Ausgeben der LabelEncoder-Klassen: {e}")

        print("Führe Bildanalyse durch...")
        analysis_result: AnalysisResult = await analyzer.analyze_image(image_bytes)
        try:
            assert analysis_result is not None, "Analyse hat kein Ergebnis zurückgegeben."
            assert isinstance(analysis_result,
                              AnalysisResult), "Analyseergebnis hat nicht den erwarteten Typ AnalysisResult."
            assert isinstance(analysis_result.plant_analysis,
                              PlantAnalysisResponse), "Analyseergebnis enthält nicht das erwartete PlantAnalysisResponse Objekt."
            print("Bildanalyse erfolgreich abgeschlossen.")
        except Exception as e:
            pytest.fail(f"Fehler während der Bildanalyse: {e}")

        print("\n--- Analyseergebnis für {image_path_to_test.name} ---")
        print(f"Status: {analysis_result.plant_analysis.status}")
        print(f"Vorhergesagtes Label (Details): {analysis_result.plant_analysis.details}")
        if analysis_result.plant_analysis.recommendations:
            print(f"Empfehlungen: {analysis_result.plant_analysis.recommendations}")
        print(f"Konfidenz: {analysis_result.confidence_score:.4f}")
        print(f"Verarbeitungszeit: {analysis_result.processing_time_ms} ms")
        print(f"Bildgröße: {analysis_result.image_width}x{analysis_result.image_height}")
        print(f"--- Ende Analyseergebnis für {image_path_to_test.name} ---")

        assert analysis_result.plant_analysis.status is not None
        assert analysis_result.plant_analysis.details is not None
        assert 0.0 <= analysis_result.confidence_score <= 1.0

    except FileNotFoundError:  # Sollte durch die assert-Zeile am Anfang abgedeckt sein
        pytest.fail(f"Testbild nicht gefunden: {image_path_to_test}")
    except Exception as e:
        pytest.fail(f"Ein unerwarteter Fehler ist im Test für Bild {image_path_to_test.name} aufgetreten: {e}")