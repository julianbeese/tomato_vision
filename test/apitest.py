# /Users/julianbeese/Developer/NOVA/tomato_vision/test/test_integration_analyzer.py
import pytest
import sys
from pathlib import Path
import pickle
import torch  # For isinstance(..., torch.Tensor) in model load test
from typing import List  # For type hinting

# Add the project root directory to the Python path to allow importing app modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import FastAPI TestClient and the main app
from fastapi.testclient import TestClient
from app.main import app  # Your FastAPI application instance
from app.config import get_settings, Settings
from app.models.plant_analysis import AnalysisResult  # Assuming API returns this as discussed

# --- Test Setup ---
settings: Settings = get_settings()
MODEL_PATH_FOR_TEST: Path = Path(settings.MODEL_PATH)

# Assuming test images are in app/data/test/images/ relative to project root
APP_DIR = PROJECT_ROOT / "app"
ACTUAL_TEST_IMAGE_DIR: Path = APP_DIR / "data" / "test" / "images"
IMAGES_TO_TEST: List[Path] = []

try:
    if not ACTUAL_TEST_IMAGE_DIR.is_dir():
        pytest.skip(f"Test image directory not found: {ACTUAL_TEST_IMAGE_DIR}", allow_module_level=True)
    else:
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        for ext in image_extensions:
            IMAGES_TO_TEST.extend(list(ACTUAL_TEST_IMAGE_DIR.glob(ext)))

        if not IMAGES_TO_TEST:
            pytest.skip(f"No test images (jpg, png) found in {ACTUAL_TEST_IMAGE_DIR}", allow_module_level=True)
        else:
            print(f"Found test images ({len(IMAGES_TO_TEST)}):")
            for img_path in IMAGES_TO_TEST[:5]:  # Show the first 5
                print(f"  - {img_path.name}")
            if len(IMAGES_TO_TEST) > 5:
                print(f"  ... and {len(IMAGES_TO_TEST) - 5} more.")

    if not MODEL_PATH_FOR_TEST.is_file():
        pytest.skip(f"Model file not found: {MODEL_PATH_FOR_TEST}", allow_module_level=True)

except Exception as e:
    pytest.skip(f"Test setup failed: {e}", allow_module_level=True)

# Initialize the TestClient
# This will also initialize your FastAPI app, including loading the analyzer_service
client = TestClient(app)


# --- Test function for direct loading of the model with Pickle ---
def test_load_model_directly_with_pickle():
    """
    Tests if the model file can be loaded directly using pickle.load().
    """
    print(f"\nTesting direct loading of model file: {MODEL_PATH_FOR_TEST}")
    assert MODEL_PATH_FOR_TEST is not None, "Model path was not initialized in setup."
    assert MODEL_PATH_FOR_TEST.is_file(), f"Model file {MODEL_PATH_FOR_TEST} does not exist."

    try:
        with open(MODEL_PATH_FOR_TEST, "rb") as f:
            model_data = pickle.load(f)
        assert model_data is not None, "pickle.load() returned None."
        print("Model file successfully loaded with pickle.load().")
        print(f"Type of loaded data: {type(model_data)}")

        # Assuming the pickle file contains a state_dict (a dictionary)
        assert isinstance(model_data, dict), "Loaded model data is not a dictionary (expected for state_dict)."
        print(f"Number of keys in the (potentially outer) dictionary: {len(model_data.keys())}")

        # Check if it's a nested state_dict or a flat one
        if "state_dict" in model_data and isinstance(model_data["state_dict"], dict):
            print(f"Number of keys in inner 'state_dict': {len(model_data['state_dict'].keys())}")
        elif all(isinstance(v, torch.Tensor) for v in model_data.values()):  # Check if all values are tensors
            print(f"The loaded dictionary appears to be a flat state_dict. Number of keys: {len(model_data.keys())}")
        else:
            # This case might indicate an unexpected structure if it's not a flat dict of tensors
            # or a dict containing 'state_dict'
            print("Loaded dictionary structure is not a recognized state_dict format (flat or nested).")

    except pickle.UnpicklingError as e:
        pytest.fail(
            f"Error unpickling the model file (pickle.UnpicklingError): {e} - File might be corrupt or not a pickle format.")
    except FileNotFoundError:
        pytest.fail(f"Model file not found at: {MODEL_PATH_FOR_TEST}")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred during direct loading of the model file: {e}")


# --- Parametrized test function for the API endpoint ---
@pytest.mark.parametrize("image_path_to_test", IMAGES_TO_TEST, ids=[img.name for img in IMAGES_TO_TEST])
def test_analyze_image_api_endpoint(image_path_to_test: Path):
    """
    Tests the /api/analyze endpoint by uploading an image and checking the response.
    It specifically verifies that recommendations are present.
    """
    print(f"\n--- Testing API with image: {image_path_to_test.name} ---")
    assert image_path_to_test.is_file(), f"Test image {image_path_to_test} not found or is not a file."

    try:
        with open(image_path_to_test, "rb") as image_file:
            # The 'file' key matches the parameter name in your API route: file: UploadFile = File(...)
            files = {"file": (image_path_to_test.name, image_file, "image/jpeg")}  # Adjust content type if needed

            # Make a POST request to the API endpoint
            # The client handles the async nature of the endpoint
            response = client.post("/api/analyze", files=files)

        # Check HTTP status code
        assert response.status_code == 200, f"API request failed with status {response.status_code}: {response.text}"

        # Parse the JSON response
        try:
            response_data = response.json()
        except ValueError:  # or JSONDecodeError depending on Python/library version
            pytest.fail(f"Failed to decode JSON response: {response.text}")

        print(f"API Response for {image_path_to_test.name}: {response_data}")

        # Validate the response structure against AnalysisResult Pydantic model
        # This implicitly checks if all fields are present and have the correct types
        try:
            analysis_result = AnalysisResult(**response_data)
        except Exception as e:  # Catches Pydantic validation errors
            pytest.fail(f"API response does not match AnalysisResult model: {e}\nResponse data: {response_data}")

        # Specifically check for recommendations
        # recommendations field is Optional[str], so it can be a string or None
        assert "recommendations" in analysis_result.plant_analysis.model_dump(), \
            "Field 'recommendations' is missing in plant_analysis."
        # If recommendations are vital, you might assert they are not None,
        # or even match expected content for certain images if you have ground truth.
        # For now, we just check its presence as per the model.
        if analysis_result.plant_analysis.recommendations is not None:
            assert isinstance(analysis_result.plant_analysis.recommendations, str), \
                "Recommendations should be a string if not None."
            print(f"Recommendations received: '{analysis_result.plant_analysis.recommendations}'")
        else:
            print("Recommendations received: None (which is valid for Optional[str])")

        # General checks for other important fields
        assert analysis_result.plant_analysis.status is not None, "Status should not be None."
        assert analysis_result.plant_analysis.details is not None, "Details should not be None."
        assert 0.0 <= analysis_result.confidence_score <= 1.0, "Confidence score out of range."
        assert analysis_result.processing_time_ms >= 0, "Processing time should be non-negative."
        assert analysis_result.image_width > 0, "Image width should be positive."
        assert analysis_result.image_height > 0, "Image height should be positive."

        print(f"Successfully validated API response for {image_path_to_test.name}")

    except FileNotFoundError:  # Should be caught by the initial assert
        pytest.fail(f"Test image not found: {image_path_to_test}")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred in API test for image {image_path_to_test.name}: {e}")