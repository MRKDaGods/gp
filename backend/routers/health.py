from datetime import datetime
from pathlib import Path

from fastapi import APIRouter

router = APIRouter()


@router.get("/api/health")
async def health_check():
    """Health check endpoint"""
    models_dir = Path("models")
    models_loaded = models_dir.exists() and any(models_dir.glob("**/*.pt"))

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": models_loaded,
        "mode": "demo" if not models_loaded else "production",
        "version": "1.0.0",
    }


@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "MTMC Tracker API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }
