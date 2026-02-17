# [Shared: Track Utilities]
"""
Endpoint warm-up check — verifies the MedGemma endpoint is online
before running experiments. Handles scale-to-zero cold starts.

Usage:
    from tracks.shared.endpoint_check import wait_for_endpoint
    await wait_for_endpoint()  # blocks until endpoint responds or gives up
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings

logger = logging.getLogger(__name__)


async def check_endpoint_health() -> tuple[bool, str]:
    """
    Send a minimal request to the endpoint.
    Returns (is_healthy, message).
    """
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=settings.medgemma_api_key or "not-needed",
            base_url=settings.medgemma_base_url or "http://localhost:8000/v1",
            timeout=30.0,
        )
        resp = await client.chat.completions.create(
            model=settings.medgemma_model_id or "tgi",
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=4,
            temperature=0.0,
        )
        text = resp.choices[0].message.content or ""
        return True, f"Endpoint alive: {text.strip()[:30]}"
    except Exception as e:
        msg = str(e)
        if "503" in msg:
            return False, "Endpoint returned 503 — model loading or scaled to zero"
        if "Connection" in msg or "connect" in msg.lower():
            return False, f"Connection error: {msg[:120]}"
        return False, f"Endpoint error: {msg[:120]}"


async def wait_for_endpoint(
    max_wait_sec: int = 600,
    poll_interval_sec: int = 30,
    quiet: bool = False,
) -> bool:
    """
    Wait for the MedGemma endpoint to become healthy.

    Polls every poll_interval_sec seconds, up to max_wait_sec total.
    Returns True if endpoint is online, False if timed out.

    Prints status messages to stdout unless quiet=True.
    """
    t0 = time.monotonic()

    ok, msg = await check_endpoint_health()
    if ok:
        if not quiet:
            print(f"[endpoint] {msg}")
        return True

    if not quiet:
        print(f"[endpoint] {msg}")
        print(f"[endpoint] Waiting up to {max_wait_sec}s for endpoint to come online...")
        print(f"[endpoint] If endpoint is paused, resume it at: https://ui.endpoints.huggingface.co/")

    attempt = 1
    while (time.monotonic() - t0) < max_wait_sec:
        await asyncio.sleep(poll_interval_sec)
        elapsed = int(time.monotonic() - t0)
        ok, msg = await check_endpoint_health()
        if ok:
            if not quiet:
                print(f"[endpoint] Online after {elapsed}s - {msg}")
            return True
        attempt += 1
        if not quiet:
            print(f"[endpoint] Attempt {attempt} ({elapsed}s elapsed): {msg}")

    if not quiet:
        print(f"[endpoint] TIMEOUT after {max_wait_sec}s — endpoint never came online")
        print(f"[endpoint] Check https://ui.endpoints.huggingface.co/ and resume the endpoint")
    return False
