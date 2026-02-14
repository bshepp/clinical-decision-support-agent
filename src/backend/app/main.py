"""
Clinical Decision Support Agent — FastAPI Backend
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import cases, health, ws
from app.config import settings

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
    # TODO: Initialize MedGemma model / connection
    # TODO: Initialize RAG vector store
    pass
