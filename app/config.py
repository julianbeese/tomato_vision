# app/config.py
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
        "http://localhost:3000",  # Standard Next.js dev server
        "http://localhost:9002",  # *** Hinzugefügt: Next.js mit Turbopack (aus frontend.txt) ***
        "http://localhost:8080", # Ggf. weitere
        "https://tomato-vision.example.com",  # Beispiel Produktions-URL (anpassen)
        # Fügen Sie hier ggf. weitere URLs hinzu, von denen Anfragen erlaubt sein sollen
    ]

    # Directory for uploaded images (wird nur kurzzeitig genutzt, wenn Bytes direkt verarbeitet werden)
    UPLOAD_DIR: str = "uploads"

    # Path to ML model and LabelEncoder
    MODEL_PATH: str = "app/models/advanced_ml_project_model_test.pkl"
    LABEL_ENCODER_PATH: str = "app/models/label_encoder_test.pkl"

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