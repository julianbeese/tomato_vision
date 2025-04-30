from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from app.api.routes import router as api_router
from app.config import get_settings

settings = get_settings()

app = FastAPI(
    title="Tomato Vision API",
    description="API for tomato plant health analysis",
    version="0.1.0",
)

# CORS-Middleware für Frontend-Integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routen einbinden
app.include_router(api_router, prefix="/api")


# Root-Endpunkt
@app.get("/", tags=["Health"])
async def root():
    return {"message": "Welcome to Tomato Vision API", "status": "active"}


# Angepasste OpenAPI-Dokumentation
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Tomato Vision API",
        version="0.1.0",
        description="API zur Analyse der Gesundheit von Tomatenpflanzen",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)