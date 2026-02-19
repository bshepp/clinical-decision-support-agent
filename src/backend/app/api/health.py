"""Health check endpoint."""
import logging

from fastapi import APIRouter

from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "ok", "service": "CDS Agent"}


@router.get("/api/health/config")
async def config_check():
    """Diagnostic endpoint: shows whether critical env vars are configured (no secrets)."""
    return {
        "medgemma_base_url_set": bool(settings.medgemma_base_url),
        "medgemma_api_key_set": bool(settings.medgemma_api_key),
        "medgemma_model_id": settings.medgemma_model_id,
        "hf_token_set": bool(settings.hf_token),
        "medgemma_max_tokens": settings.medgemma_max_tokens,
    }


@router.get("/api/health/model")
async def model_readiness():
    """Check if the MedGemma endpoint is warm and accepting requests."""
    from app.services.medgemma import MedGemmaService

    service = MedGemmaService()
    ready = await service.check_readiness()
    return {
        "ready": ready,
        "model_id": settings.medgemma_model_id,
        "base_url_set": bool(settings.medgemma_base_url),
    }