# [Track D: Arbitrated Parallel]
"""
Track D — Domain-specialist reasoning agents.

Each specialist receives the same patient profile but a different
system prompt that biases it toward its domain. Specialists run
in parallel via asyncio.gather().
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Tuple

from app.models.schemas import ClinicalReasoningResult, PatientProfile
from app.services.medgemma import MedGemmaService
from app.tools.clinical_reasoning import SYSTEM_PROMPT as BASE_SYSTEM_PROMPT, REASONING_PROMPT
from tracks.arbitrated.config import ArbitratedConfig, SpecialistDef
from tracks.shared.cost_tracker import (
    CostLedger,
    LLMCallRecord,
    estimate_cost,
    estimate_tokens,
)

logger = logging.getLogger(__name__)


class SpecialistAgent:
    """
    A single specialist reasoning agent.

    Uses the base clinical reasoning prompt augmented with a
    domain-specific system prompt addendum.
    """

    def __init__(
        self,
        spec: SpecialistDef,
        temperature: float = 0.3,
        max_tokens: int = 3072,
    ):
        self.spec = spec
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.medgemma = MedGemmaService()

    async def reason(
        self,
        profile: PatientProfile,
        ledger: CostLedger,
        iteration: int = 0,
        additional_context: str = "",
    ) -> ClinicalReasoningResult:
        """
        Run specialist reasoning on a patient profile.

        Args:
            profile: Structured patient data
            ledger: Cost tracker
            iteration: Which arbiter round (0 = initial)
            additional_context: Arbiter feedback for resubmission rounds
        """
        system_prompt = BASE_SYSTEM_PROMPT + "\n\n" + self.spec.system_prompt_addendum

        prompt = REASONING_PROMPT.format(
            age=profile.age or "Unknown",
            gender=profile.gender.value,
            chief_complaint=profile.chief_complaint,
            hpi=profile.history_of_present_illness or "Not provided",
            pmh=", ".join(profile.past_medical_history) if profile.past_medical_history else "None reported",
            medications=_format_medications(profile),
            allergies=", ".join(profile.allergies) if profile.allergies else "NKDA",
            labs=_format_labs(profile),
            vitals=_format_vitals(profile),
            social_hx=profile.social_history or "Not provided",
            family_hx=profile.family_history or "Not provided",
        )

        if additional_context:
            prompt += f"\n\nARBITER FEEDBACK FROM PREVIOUS ROUND:\n{additional_context}"

        input_tokens = estimate_tokens(system_prompt + prompt)
        t0 = time.monotonic()

        result = await self.medgemma.generate_structured(
            prompt=prompt,
            response_model=ClinicalReasoningResult,
            system_prompt=system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        output_tokens = estimate_tokens(str(result.model_dump_json()))

        ledger.calls.append(LLMCallRecord(
            call_id=str(uuid.uuid4())[:8],
            track_id=ledger.track_id,
            step_name=f"specialist_{self.spec.specialist_id}",
            iteration=iteration,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            latency_ms=elapsed_ms,
            temperature=self.temperature,
            max_tokens_requested=self.max_tokens,
            estimated_cost_usd=estimate_cost(input_tokens, output_tokens),
            timestamp=time.time(),
        ))

        logger.info(
            f"  [{self.spec.name}] round {iteration}: "
            f"{len(result.differential_diagnosis)} diagnoses, "
            f"top: {result.differential_diagnosis[0].diagnosis if result.differential_diagnosis else '?'}"
        )
        return result


async def run_specialists_parallel(
    specialists: List[SpecialistAgent],
    profile: PatientProfile,
    ledger: CostLedger,
    iteration: int = 0,
    arbiter_feedback: Optional[Dict[str, str]] = None,
) -> Dict[str, ClinicalReasoningResult]:
    """
    Run all specialists in parallel, returning specialist_id → result.

    Args:
        arbiter_feedback: Optional dict of specialist_id → tailored prompt for resubmission.
    """
    feedback = arbiter_feedback or {}

    tasks = [
        agent.reason(
            profile=profile,
            ledger=ledger,
            iteration=iteration,
            additional_context=feedback.get(agent.spec.specialist_id, ""),
        )
        for agent in specialists
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    output: Dict[str, ClinicalReasoningResult] = {}
    for agent, result in zip(specialists, results):
        if isinstance(result, Exception):
            logger.error(f"Specialist {agent.spec.name} failed: {result}")
            continue
        output[agent.spec.specialist_id] = result

    return output


# ──────────────────────────────────────────────
# Formatting helpers (avoid circular import from clinical_reasoning)
# ──────────────────────────────────────────────

def _format_medications(profile: PatientProfile) -> str:
    if not profile.current_medications:
        return "None reported"
    return "; ".join(f"{m.name} {m.dose or ''}" for m in profile.current_medications)


def _format_labs(profile: PatientProfile) -> str:
    if not profile.lab_results:
        return "None available"
    return "; ".join(
        f"{l.test_name}: {l.value}{' [ABNORMAL]' if l.is_abnormal else ''}"
        for l in profile.lab_results
    )


def _format_vitals(profile: PatientProfile) -> str:
    if not profile.vital_signs:
        return "Not available"
    v = profile.vital_signs
    parts = []
    if v.blood_pressure:
        parts.append(f"BP {v.blood_pressure}")
    if v.heart_rate:
        parts.append(f"HR {v.heart_rate}")
    if v.temperature:
        parts.append(f"Temp {v.temperature}")
    if v.respiratory_rate:
        parts.append(f"RR {v.respiratory_rate}")
    if v.oxygen_saturation:
        parts.append(f"SpO2 {v.oxygen_saturation}")
    return ", ".join(parts) if parts else "Not available"
