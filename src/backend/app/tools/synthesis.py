"""
Tool: Synthesis Agent

Uses MedGemma to synthesize all tool outputs into a final Clinical Decision Support report.
This is the capstone of the pipeline — it takes structured data from every tool
and produces a cohesive, clinician-ready report.
"""
from __future__ import annotations

import logging
from typing import Optional

from app.models.schemas import (
    CDSReport,
    ClinicalReasoningResult,
    ConflictDetectionResult,
    DrugInteractionResult,
    GuidelineRetrievalResult,
    PatientProfile,
)
from app.services.medgemma import MedGemmaService

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a clinical decision support synthesis engine. Your job is to 
combine outputs from multiple clinical tools into a single, cohesive report for a clinician.

CRITICAL RULES:
1. Be concise and clinically precise
2. Prioritize safety — drug interactions and critical findings go first
3. Clearly distinguish between tool-verified facts and model-generated reasoning
4. Always include caveats and limitations
5. Cite sources when available
6. This report SUPPORTS clinical decision-making — it does NOT replace clinician judgment
7. Include a standard disclaimer about AI-generated content"""

SYNTHESIS_PROMPT = """Synthesize the following clinical tool outputs into a cohesive 
Clinical Decision Support report.

═══ PATIENT PROFILE ═══
{patient_profile}

═══ CLINICAL REASONING (MedGemma) ═══
{clinical_reasoning}

═══ DRUG INTERACTION CHECK ═══
{drug_interactions}

═══ CLINICAL GUIDELINES ═══
{guidelines}

═══ CONFLICTS & GAPS DETECTED ═══
{conflicts}

Create a comprehensive CDS report including:
1. Patient Summary — concise summary of the case
2. Differential Diagnosis — ranked with reasoning, integrating guideline concordance
3. Drug Interaction Warnings — any flagged interactions with clinical significance
4. Guideline-Concordant Recommendations — actionable steps aligned with guidelines
5. Conflicts & Gaps — PROMINENTLY include every detected conflict. For each conflict,
   state what the guideline recommends, what the patient's current state is, and the
   suggested resolution. This section is CRITICAL for patient safety.
6. Suggested Next Steps — prioritized actions for the clinician, incorporating conflict resolutions
7. Caveats — limitations, uncertainties, and important disclaimers
8. Sources — cited guidelines and data sources used"""


class SynthesisTool:
    """Synthesizes all tool outputs into a final CDS report using MedGemma."""

    def __init__(self):
        self.medgemma = MedGemmaService()

    async def run(
        self,
        patient_profile: Optional[PatientProfile],
        clinical_reasoning: Optional[ClinicalReasoningResult],
        drug_interactions: Optional[DrugInteractionResult],
        guideline_retrieval: Optional[GuidelineRetrievalResult],
        conflict_detection: Optional[ConflictDetectionResult] = None,
    ) -> CDSReport:
        """
        Synthesize all available tool outputs into a final CDS report.

        Args:
            patient_profile: Structured patient data
            clinical_reasoning: Differential diagnosis and recommendations
            drug_interactions: Drug interaction check results
            guideline_retrieval: Retrieved clinical guideline excerpts

        Returns:
            CDSReport — the final clinician-facing report
        """
        prompt = SYNTHESIS_PROMPT.format(
            patient_profile=self._format_profile(patient_profile),
            clinical_reasoning=self._format_reasoning(clinical_reasoning),
            drug_interactions=self._format_interactions(drug_interactions),
            guidelines=self._format_guidelines(guideline_retrieval),
            conflicts=self._format_conflicts(conflict_detection),
        )

        report = await self.medgemma.generate_structured(
            prompt=prompt,
            response_model=CDSReport,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=4096,
        )

        # Add standard disclaimer to caveats
        report.caveats.append(
            "This report is AI-generated and intended for clinical decision SUPPORT only. "
            "It does not replace professional medical judgment. All recommendations should "
            "be verified by a qualified clinician before acting on them."
        )

        # Ensure detected conflicts are always surfaced even if LLM doesn't
        # populate the conflicts field in its structured output
        if conflict_detection and conflict_detection.conflicts:
            if not report.conflicts:
                report.conflicts = conflict_detection.conflicts

        logger.info("Synthesis complete — CDS report generated")
        return report

    @staticmethod
    def _format_profile(profile: Optional[PatientProfile]) -> str:
        if not profile:
            return "Patient profile not available"
        parts = [
            f"Age: {profile.age or 'Unknown'}, Gender: {profile.gender.value}",
            f"Chief Complaint: {profile.chief_complaint}",
            f"HPI: {profile.history_of_present_illness}",
        ]
        if profile.past_medical_history:
            parts.append(f"PMH: {', '.join(profile.past_medical_history)}")
        if profile.current_medications:
            meds = "; ".join(f"{m.name} {m.dose or ''}" for m in profile.current_medications)
            parts.append(f"Medications: {meds}")
        if profile.allergies:
            parts.append(f"Allergies: {', '.join(profile.allergies)}")
        if profile.lab_results:
            labs = "; ".join(
                f"{l.test_name}: {l.value}{' [ABNORMAL]' if l.is_abnormal else ''}"
                for l in profile.lab_results
            )
            parts.append(f"Labs: {labs}")
        return "\n".join(parts)

    @staticmethod
    def _format_reasoning(reasoning: Optional[ClinicalReasoningResult]) -> str:
        if not reasoning:
            return "Clinical reasoning not available"
        parts = []
        if reasoning.differential_diagnosis:
            parts.append("Differential Diagnosis:")
            for i, dx in enumerate(reasoning.differential_diagnosis, 1):
                parts.append(
                    f"  {i}. {dx.diagnosis} (likelihood: {dx.likelihood.value}) — {dx.reasoning}"
                )
        if reasoning.risk_assessment:
            parts.append(f"Risk Assessment: {reasoning.risk_assessment}")
        if reasoning.recommended_workup:
            parts.append("Recommended Workup:")
            for action in reasoning.recommended_workup:
                parts.append(
                    f"  - [{action.priority.value.upper()}] {action.action} — {action.rationale}"
                )
        return "\n".join(parts)

    @staticmethod
    def _format_interactions(interactions: Optional[DrugInteractionResult]) -> str:
        if not interactions:
            return "Drug interaction check not performed"
        if not interactions.interactions_found:
            return f"No interactions found among {len(interactions.medications_checked)} medications checked"
        parts = [f"Checked {len(interactions.medications_checked)} medications:"]
        for ix in interactions.interactions_found:
            parts.append(
                f"  ⚠ {ix.drug_a} + {ix.drug_b} [{ix.severity.value.upper()}]: {ix.description}"
            )
        if interactions.warnings:
            parts.append("Warnings: " + "; ".join(interactions.warnings))
        return "\n".join(parts)

    @staticmethod
    def _format_guidelines(guidelines: Optional[GuidelineRetrievalResult]) -> str:
        if not guidelines or not guidelines.excerpts:
            return "No guidelines retrieved"
        parts = [f"Query: {guidelines.query}", "Retrieved excerpts:"]
        for excerpt in guidelines.excerpts:
            score = f" (relevance: {excerpt.relevance_score})" if excerpt.relevance_score else ""
            parts.append(f"  [{excerpt.source}] {excerpt.title}{score}")
            parts.append(f"    {excerpt.excerpt[:300]}...")
        return "\n".join(parts)

    @staticmethod
    def _format_conflicts(conflicts: Optional[ConflictDetectionResult]) -> str:
        if not conflicts or not conflicts.conflicts:
            return "No conflicts detected between guidelines and patient data"
        parts = [
            f"{len(conflicts.conflicts)} conflict(s) detected "
            f"across {conflicts.guidelines_checked} guidelines:"
        ]
        for i, c in enumerate(conflicts.conflicts, 1):
            parts.append(
                f"\n  {i}. [{c.severity.value.upper()}] {c.conflict_type.value.upper()}"
            )
            parts.append(f"     Guideline ({c.guideline_source}): {c.guideline_text}")
            parts.append(f"     Patient data: {c.patient_data}")
            parts.append(f"     Issue: {c.description}")
            if c.suggested_resolution:
                parts.append(f"     Suggested resolution: {c.suggested_resolution}")
        return "\n".join(parts)
