# app/api/routes.py
import os
import uuid
from pathlib import Path
from typing import Dict, Any
import logging

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, BackgroundTasks
# No JSONResponse needed if returning Pydantic model directly
# from fastapi.responses import JSONResponse # Remove if not used elsewhere

from app.config import get_settings, Settings
# Import AnalysisResult along with PlantAnalysisResponse
from app.models.plant_analysis import PlantAnalysisResponse, AnalysisResult # <--- MODIFIED
from app.services.analyzer import analyzer_service

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