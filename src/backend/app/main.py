"""
Clinical Decision Support Agent — FastAPI Backend
"""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import cases, health, ws
from app.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Clinical Decision Support Agent",
    description="Agentic clinical decision support powered by MedGemma (HAI-DEF)",
    version="0.1.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(health.router, tags=["health"])
app.include_router(cases.router, prefix="/api/cases", tags=["cases"])
app.include_router(ws.router, prefix="/ws", tags=["websocket"])


@app.on_event("startup")
async def startup():
    """Initialize services on startup."""
    # Log configuration (mask secrets)
    def _mask(val: str) -> str:
        if not val:
            return "(empty)"
        if len(val) <= 8:
            return "***"
        return val[:4] + "..." + val[-4:]

    logger.info("=== CDS Agent Backend Starting ===")
    logger.info(f"  medgemma_base_url : {settings.medgemma_base_url or '(empty)'}")
    logger.info(f"  medgemma_model_id : {settings.medgemma_model_id}")
    logger.info(f"  medgemma_api_key  : {_mask(settings.medgemma_api_key)}")
    logger.info(f"  hf_token          : {_mask(settings.hf_token)}")
    logger.info(f"  medgemma_max_tokens: {settings.medgemma_max_tokens}")
    logger.info(f"  cors_origins      : {settings.cors_origins}")
    logger.info(f"  chroma_persist_dir: {settings.chroma_persist_dir}")

    if not settings.medgemma_base_url:
        logger.warning("MEDGEMMA_BASE_URL is empty -- MedGemma API calls will fail!")
    if not settings.medgemma_api_key:
        logger.warning("MEDGEMMA_API_KEY is empty -- MedGemma API calls will fail!")
