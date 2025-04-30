import os
from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()


class Settings(BaseSettings):
    """Application configuration based on environment variables"""

    # API configuration
    API_PREFIX: str = "/api"

    # CORS configuration (Frontend URLs)
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",  # Next.js development server
        "http://localhost:9002",  # Next.js with Turbopack
        "http://localhost:8080",
        "https://tomato-vision.example.com",  # Production URL (adjust as needed)
    ]

    # Directory for uploaded images
    UPLOAD_DIR: str = "uploads"

    # Path to ML model
    MODEL_PATH: str = "model/plant_analysis_model.pkl"

    # Debug mode
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # Maximum file size for uploads (in bytes)
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10 MB

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Return settings with caching"""
    return Settings()