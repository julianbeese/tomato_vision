# app/api/routes.py
import os
import uuid
from pathlib import Path
from typing import Dict, Any
import logging
from io import BytesIO

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse

from app.config import get_settings, Settings
from app.models.plant_analysis import PlantAnalysisResponse, AnalysisResult
from app.services.analyzer import analyzer_service
from app.services.visualizer import generate_grad_cam_overlay

router = APIRouter()
settings = get_settings()
logger = logging.getLogger(__name__)

SUPPORTED_CONTENT_TYPES = ["image/jpeg", "image/png", "image/webp"]

@router.post(
    "/analyze",
    response_model=AnalysisResult,  # <--- MODIFIED: Change to AnalysisResult
    summary="Analyze an image of a tomato plant",
    description="Upload an image and analyze the health status of the tomato plant",
)
async def analyze_plant_image(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        settings: Settings = Depends(get_settings),
) -> AnalysisResult: # <--- MODIFIED: Change return type hint
    """
    API endpoint for plant image analysis. Reads image bytes directly.

    Args:
        background_tasks: Tasks for background processing.
        file: The uploaded image file.
        settings: Application settings.

    Returns:
        The full analysis result including metadata. # <--- MODIFIED: Updated docstring
    """
    if file.content_type not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Only {', '.join(SUPPORTED_CONTENT_TYPES)} allowed."
        )

    if file.size > settings.MAX_UPLOAD_SIZE:
         raise HTTPException(
             status_code=413,
             detail=f"File too large. Maximum size is {settings.MAX_UPLOAD_SIZE // 1024 // 1024} MB."
         )

    try:
        image_bytes = await file.read()

        if not image_bytes:
             raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        result_internal = await analyzer_service.analyze_image(image_bytes) # This returns AnalysisResult

        return result_internal  # <--- MODIFIED: Return the full AnalysisResult object

    except HTTPException as http_exc:
         raise http_exc
    except Exception as e:
        logger.error(f"Error during image analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during image analysis: {str(e)}")

# NEUE ROUTE für Grad-CAM Visualisierung
@router.post(
    "/visualize_grad_cam",
    summary="Generate Grad-CAM++ visualization for a tomato plant image",
    description="Upload an image and get back the image with Grad-CAM++ overlay and bounding boxes.",
    response_class=StreamingResponse # Wichtig für Bild-Rückgabe
)
async def visualize_plant_image_grad_cam(
    file: UploadFile = File(...),
    # settings: Settings = Depends(get_settings) # Falls benötigt
):
    """
    API endpoint for generating Grad-CAM++ visualization.
    """
    if file.content_type not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Only {', '.join(SUPPORTED_CONTENT_TYPES)} allowed."
        )

    # Optional: Dateigrößenprüfung, ähnlich wie in /analyze
    # if file.size > settings.MAX_UPLOAD_SIZE:
    #     raise HTTPException(
    #         status_code=413,
    #         detail=f"File too large. Maximum size is {settings.MAX_UPLOAD_SIZE // 1024 // 1024} MB."
    #     )

    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Rufe die neue Funktion zur Erzeugung des Overlays auf
        processed_image_bytes = generate_grad_cam_overlay(image_bytes)

        # Gib das Bild als StreamingResponse zurück
        # media_type "image/png" oder "image/jpeg" je nachdem, wie du es in generate_grad_cam_overlay speicherst
        return StreamingResponse(BytesIO(processed_image_bytes), media_type="image/png")

    except HTTPException as http_exc:
        # Diese Exceptions werden von generate_grad_cam_overlay oder den Prüfungen hier geworfen
        raise http_exc
    except RuntimeError as r_err: # Fängt Fehler vom Modellladen etc. ab
        logger.error(f"Runtime error during visualization: {str(r_err)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during visualization: {str(r_err)}")
    except Exception as e:
        logger.error(f"Unexpected error during Grad-CAM visualization: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error during Grad-CAM visualization: {str(e)}")


# Health Check remains unchanged
@router.get(
    "/health",
    summary="API health status",
    description="Check if the analysis service is available"
)
async def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "service": "tomato-vision-analyzer",
        "version": "0.1.0",
    }
