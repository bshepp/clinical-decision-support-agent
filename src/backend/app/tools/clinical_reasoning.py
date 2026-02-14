"""
Tool: Clinical Reasoning Agent

Uses MedGemma to perform clinical reasoning over a structured patient profile.
Generates differential diagnosis, risk assessment, and recommended workup.
"""
from __future__ import annotations

import logging

from app.models.schemas import (
    ClinicalReasoningResult,
    PatientProfile,
)
from app.services.medgemma import MedGemmaService

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert clinical reasoning assistant. Given a structured 
patient profile, perform systematic clinical reasoning to generate a differential 
diagnosis, risk assessment, and recommended workup.

IMPORTANT GUIDELINES:
- Think step-by-step through the clinical reasoning process
- Consider the most likely diagnoses first, then less common but important ones
- Always consider dangerous "can't miss" diagnoses
- Base your reasoning on the available evidence (symptoms, labs, history)
- Be explicit about your reasoning chain
- Rate likelihood as "low", "moderate", or "high"
- Rate priority of actions as "low", "moderate", "high", or "critical"
- This is a decision SUPPORT tool — always recommend clinician judgment"""

REASONING_PROMPT = """Perform clinical reasoning on the following patient case.

PATIENT PROFILE:
- Age: {age}, Gender: {gender}
- Chief Complaint: {chief_complaint}
- HPI: {hpi}
- Past Medical History: {pmh}
- Current Medications: {medications}
- Allergies: {allergies}
- Lab Results: {labs}
- Vital Signs: {vitals}
- Social History: {social_hx}
- Family History: {family_hx}

Provide:
1. A ranked differential diagnosis (most likely first) with supporting evidence and reasoning
2. An overall risk assessment
3. Recommended workup (tests, referrals, treatments) with priority levels and rationale
4. Your full chain-of-thought reasoning"""


class ClinicalReasoningTool:
    """Uses MedGemma for clinical reasoning over patient data."""

    def __init__(self):
        self.medgemma = MedGemmaService()

    async def run(self, profile: PatientProfile) -> ClinicalReasoningResult:
        """
        Perform clinical reasoning on a patient profile.

        Args:
            profile: Structured patient profile from the parser

        Returns:
            ClinicalReasoningResult with differential diagnosis, risk, and recommendations
        """
        prompt = REASONING_PROMPT.format(
            age=profile.age or "Unknown",
            gender=profile.gender.value,
            chief_complaint=profile.chief_complaint,
            hpi=profile.history_of_present_illness or "Not provided",
            pmh=", ".join(profile.past_medical_history) if profile.past_medical_history else "None reported",
            medications=self._format_medications(profile),
            allergies=", ".join(profile.allergies) if profile.allergies else "NKDA",
            labs=self._format_labs(profile),
            vitals=self._format_vitals(profile),
            social_hx=profile.social_history or "Not provided",
            family_hx=profile.family_history or "Not provided",
        )

        result = await self.medgemma.generate_structured(
            prompt=prompt,
            response_model=ClinicalReasoningResult,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=4096,
        )

        logger.info(
            f"Clinical reasoning complete: {len(result.differential_diagnosis)} diagnoses, "
            f"{len(result.recommended_workup)} recommendations"
        )
        return result

    @staticmethod
    def _format_medications(profile: PatientProfile) -> str:
        if not profile.current_medications:
            return "None reported"
        return "; ".join(
            f"{m.name} {m.dose or ''}" for m in profile.current_medications
        )

    @staticmethod
    def _format_labs(profile: PatientProfile) -> str:
        if not profile.lab_results:
            return "None available"
        lines = []
        for lab in profile.lab_results:
            abnormal = " [ABNORMAL]" if lab.is_abnormal else ""
            ref = f" (ref: {lab.reference_range})" if lab.reference_range else ""
            lines.append(f"{lab.test_name}: {lab.value}{ref}{abnormal}")
        return "; ".join(lines)

    @staticmethod
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
