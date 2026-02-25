"""
Feedback endpoint — stores user feedback from the demo.
Simple JSON Lines file storage. No database needed.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()

FEEDBACK_FILE = Path("/tmp/cds_feedback.jsonl")


class FeedbackSubmission(BaseModel):
    message: str = Field(..., max_length=1000)
    contact: str | None = Field(None, max_length=200)
    page_url: str | None = None
    user_agent: str | None = None


@router.post("/api/feedback")
async def submit_feedback(feedback: FeedbackSubmission, request: Request):
    """Save user feedback to a JSONL file."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message": feedback.message,
        "contact": feedback.contact,
        "page_url": feedback.page_url,
        "user_agent": feedback.user_agent,
        "client_ip": request.client.host if request.client else None,
    }

    try:
        with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info(f"Feedback saved: {feedback.message[:50]}...")
    except Exception as e:
        logger.error(f"Failed to save feedback: {e}")
        return {"status": "error", "message": "Failed to save feedback"}

    return {"status": "ok", "message": "Thank you for your feedback!"}


@router.get("/api/feedback")
async def list_feedback():
    """List all feedback (for the developer). No auth — it's a demo."""
    if not FEEDBACK_FILE.exists():
        return {"feedback": [], "count": 0}

    entries = []
    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except Exception as e:
        logger.error(f"Failed to read feedback: {e}")
        return {"feedback": [], "count": 0, "error": str(e)}

    return {"feedback": entries, "count": len(entries)}
