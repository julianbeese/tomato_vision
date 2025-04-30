from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class PlantStatus(str, Enum):
    """Possible statuses of an analyzed plant"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"

    # Additional specific conditions could be added here, e.g.:
    # NUTRIENT_DEFICIENCY = "nutrient_deficiency"
    # PEST_DAMAGE = "pest_damage"
    # DISEASE = "disease"
    # WATER_STRESS = "water_stress"


class PlantAnalysisResponse(BaseModel):
    """Model for the plant analysis response"""
    status: PlantStatus = Field(
        ...,
        description="Health status of the plant"
    )
    details: str = Field(
        ...,
        description="Detailed information about the analysis results"
    )
    recommendations: Optional[str] = Field(
        None,
        description="Recommended actions based on the analysis"
    )


class AnalysisResult(BaseModel):
    """Internal representation of the analysis result with additional metadata"""
    plant_analysis: PlantAnalysisResponse
    confidence_score: float = Field(
        ...,
        description="Confidence score of the analysis (0-1)",
        ge=0.0,
        le=1.0
    )
    processing_time_ms: int = Field(
        ...,
        description="Processing time in milliseconds"
    )
    image_width: int = Field(..., description="Width of the analyzed image")
    image_height: int = Field(..., description="Height of the analyzed image")