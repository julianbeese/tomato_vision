import os
import uuid
from pathlib import Path
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from app.config import get_settings, Settings
from app.models.plant_analysis import PlantAnalysisResponse
from app.services.analyzer import analyzer_service

router = APIRouter()
settings = get_settings()


def get_file_extension(filename: str) -> str:
    """Returns the file extension"""
    return Path(filename).suffix.lower()


async def save_upload_file(upload_file: UploadFile) -> str:
    """
    Saves the uploaded file and returns the storage path

    Args:
        upload_file: The uploaded file

    Returns:
        The path where the file was saved
    """
    # Ensure upload directory exists
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    # Check file extension and generate secure filename
    file_extension = get_file_extension(upload_file.filename)
    if file_extension not in ['.jpg', '.jpeg', '.png', '.webp']:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, PNG or WEBP images are allowed."
        )

    # Generate unique filename
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(settings.UPLOAD_DIR, unique_filename)

    # Save file
    try:
        contents = await upload_file.read()
        with open(file_path, 'wb') as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

    return file_path


async def remove_file(file_path: str) -> None:
    """
    Removes a file after processing

    Args:
        file_path: Path to the file to be removed
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        # Just log, don't fail
        print(f"Warning: Error removing temporary file {file_path}: {e}")


@router.post(
    "/analyze",
    response_model=PlantAnalysisResponse,
    summary="Analyze an image of a tomato plant",
    description="Upload an image and analyze the health status of the tomato plant",
)
async def analyze_plant_image(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        settings: Settings = Depends(get_settings),
) -> PlantAnalysisResponse:
    """
    API endpoint for plant image analysis

    Args:
        background_tasks: Tasks for background processing
        file: The uploaded image
        settings: Application settings

    Returns:
        The analysis result
    """
    # Save file
    file_path = await save_upload_file(file)

    try:
        # Analyze image
        result = await analyzer_service.analyze_image(file_path)

        # Remove file after analysis
        background_tasks.add_task(remove_file, file_path)

        # Return only the PlantAnalysisResponse object,
        # not the internal metadata
        return result.plant_analysis

    except Exception as e:
        # Clean up in case of error
        background_tasks.add_task(remove_file, file_path)
        raise HTTPException(status_code=500, detail=f"Error during image analysis: {str(e)}")


@router.get(
    "/health",
    summary="API health status",
    description="Check if the analysis service is available"
)
async def health_check() -> Dict[str, Any]:
    """
    Simple health check for the API

    Returns:
        Status and version of the API
    """
    return {
        "status": "healthy",
        "service": "tomato-vision-analyzer",
        "version": "0.1.0",
    }