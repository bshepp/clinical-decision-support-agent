"""
MedGemma Service — handles all communication with the MedGemma model.

Supports two modes:
  1. API mode — calls MedGemma via an OpenAI-compatible API endpoint
  2. Local mode — loads the model locally via transformers (for edge/offline)

All tools that need MedGemma go through this service.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel

from app.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class MedGemmaService:
    """
    Unified interface for MedGemma inference.

    Usage:
        service = MedGemmaService()
        result = await service.generate("Analyze this patient case...", max_tokens=2048)
        structured = await service.generate_structured("...", ResponseModel)
    """

    def __init__(self):
        self._client = None
        self._local_model = None
        self._mode = "api" if settings.medgemma_base_url else "local"

    async def _get_client(self):
        """Lazy-initialize the API client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    api_key=settings.medgemma_api_key or "not-needed",
                    base_url=settings.medgemma_base_url or "http://localhost:8000/v1",
                )
            except ImportError:
                raise RuntimeError(
                    "openai package required for API mode. Install with: pip install openai"
                )
        return self._client

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 0,
        temperature: float = 0.3,
    ) -> str:
        """
        Generate text from MedGemma.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt for context setting
            max_tokens: Max tokens to generate (0 = use default from config)
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        max_tokens = max_tokens or settings.medgemma_max_tokens

        if self._mode == "api":
            return await self._generate_api(prompt, system_prompt, max_tokens, temperature)
        else:
            return await self._generate_local(prompt, system_prompt, max_tokens, temperature)

    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
        max_tokens: int = 0,
        temperature: float = 0.2,
    ) -> T:
        """
        Generate a structured (Pydantic model) response from MedGemma.

        Appends JSON schema instructions to the prompt and parses the response.

        Args:
            prompt: The user prompt
            response_model: Pydantic model class to parse the response into
            system_prompt: Optional system prompt
            max_tokens: Max tokens
            temperature: Sampling temperature

        Returns:
            Parsed Pydantic model instance
        """
        schema = response_model.model_json_schema()
        structured_prompt = (
            f"{prompt}\n\n"
            f"Respond ONLY with valid JSON matching this schema:\n"
            f"```json\n{json.dumps(schema, indent=2)}\n```\n"
            f"Do not include any text outside the JSON."
        )

        raw = await self.generate(structured_prompt, system_prompt, max_tokens, temperature)

        # Extract JSON from response (handle markdown code blocks)
        json_str = self._extract_json(raw)

        try:
            data = json.loads(json_str)
            return response_model.model_validate(data)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse structured response: {e}. Raw: {raw[:200]}")
            # Retry with stricter prompt as fallback
            raise ValueError(f"MedGemma returned invalid JSON for {response_model.__name__}: {e}")

    async def _generate_api(
        self, prompt: str, system_prompt: Optional[str], max_tokens: int, temperature: float
    ) -> str:
        """Generate via OpenAI-compatible API.

        MedGemma (served by TGI on HuggingFace Endpoints) natively supports the
        system role, so we send system/user messages properly.  If the backend
        happens to be plain Gemma on Google AI Studio (which rejects the system
        role), we automatically fall back to folding the system prompt into the
        user message.
        """
        client = await self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await client.chat.completions.create(
                model=settings.medgemma_model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            # Fallback: fold system prompt into user message (Google AI Studio compat)
            if system_prompt and "system" in str(e).lower():
                logger.warning("Backend rejected system role — folding into user message.")
                fallback_messages = [
                    {"role": "user", "content": f"{system_prompt}\n\n{prompt}"}
                ]
                response = await client.chat.completions.create(
                    model=settings.medgemma_model_id,
                    messages=fallback_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content
            raise

    async def _generate_local(
        self, prompt: str, system_prompt: Optional[str], max_tokens: int, temperature: float
    ) -> str:
        """Generate via locally loaded model (transformers)."""
        # TODO: Implement local inference with transformers
        # This is the path for edge deployment or offline development
        raise NotImplementedError(
            "Local inference not yet implemented. "
            "Set MEDGEMMA_BASE_URL to use API mode, or implement local loading."
        )

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON from a response that might include markdown code blocks."""
        # Try to find JSON in ```json ... ``` blocks
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            return text[start:end].strip()
        if "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            return text[start:end].strip()
        # Try to find raw JSON
        for i, char in enumerate(text):
            if char in "{[":
                # Find matching closing bracket
                depth = 0
                for j in range(i, len(text)):
                    if text[j] in "{[":
                        depth += 1
                    elif text[j] in "}]":
                        depth -= 1
                    if depth == 0:
                        return text[i : j + 1]
        return text.strip()
