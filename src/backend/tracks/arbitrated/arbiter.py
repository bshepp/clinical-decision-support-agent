# [Track D: Arbitrated Parallel]
"""
Track D — Arbiter agent.

The arbiter receives all specialist differentials, identifies agreements
and disagreements, produces a consensus differential, and (for round 2+)
generates _tailored_ resubmission prompts for specialists that disagree
with the consensus.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Dict, List, Optional

from app.models.schemas import ClinicalReasoningResult, PatientProfile
from app.services.medgemma import MedGemmaService
from tracks.arbitrated.config import ArbitratedConfig, SpecialistDef
from tracks.shared.cost_tracker import (
    CostLedger,
    LLMCallRecord,
    estimate_cost,
    estimate_tokens,
)

logger = logging.getLogger(__name__)

ARBITER_SYSTEM = """You are a chief medical officer arbitrating between multiple specialist opinions.
You receive differential diagnoses from multiple domain specialists — each has a different
clinical lens (e.g. cardiologist, infectious disease, general internist).

Your role:
1. IDENTIFY AGREEMENTS — diagnoses that appear across multiple specialists.
2. IDENTIFY DISAGREEMENTS — diagnoses championed by one specialist but missed/contradicted by others.
3. WEIGH EVIDENCE — specialists who provide stronger evidence-based reasoning should be weighted more.
4. PRODUCE A CONSENSUS DIFFERENTIAL — your final ranked list, with citations to which specialists
   support each diagnosis and why.
5. FLAG UNCERTAINTY — where specialists fundamentally disagree, explicitly state the disagreement
   and recommend workup to resolve it.
6. For subsequent rounds, generate TAILORED FEEDBACK for specialists who should reconsider.

You are the final arbiter. Your differential may differ from any individual specialist."""

ARBITER_MERGE_PROMPT = """
PATIENT PROFILE:
{patient_summary}

══════ SPECIALIST OPINIONS (Round {round}) ══════
{specialist_outputs}

══════ ARBITRATION TASK ══════
1. List every unique diagnosis across all specialists.
2. For each, note which specialists support it and their reasoning.
3. Produce YOUR consensus differential diagnosis, ranked by likelihood.
4. For each diagnosis, cite which specialist(s) informed your ranking.
5. If specialists disagree, state what additional evidence would resolve the disagreement.

Return a JSON object matching ClinicalReasoningResult schema."""

ARBITER_FEEDBACK_PROMPT = """
PATIENT PROFILE:
{patient_summary}

══════ CURRENT CONSENSUS (after round {round}) ══════
{consensus}

══════ SPECIALIST {specialist_name} ({specialist_id}) OPINION ══════
{specialist_output}

══════ TASK ══════
The specialist above DISAGREES with the consensus on key points.
Write a brief, specific prompt (2-4 sentences) asking this specialist to:
  • Reconsider the consensus top diagnoses and provide counter-evidence if they still disagree.
  • Address any diagnoses they missed that the consensus includes.
  • Sharpen their reasoning with specific supporting/contradicting evidence.

Return ONLY the feedback text — no JSON, no headers."""


class Arbiter:
    """
    Merges specialist outputs and produces consensus differentials.

    Also generates tailored resubmission prompts for multi-round experiments.
    """

    def __init__(self, config: ArbitratedConfig):
        self.config = config
        self.medgemma = MedGemmaService()

    async def merge(
        self,
        profile: PatientProfile,
        specialist_results: Dict[str, ClinicalReasoningResult],
        specialist_defs: Dict[str, SpecialistDef],
        ledger: CostLedger,
        round_num: int = 1,
    ) -> ClinicalReasoningResult:
        """
        Merge specialist differentials into a consensus.

        Args:
            specialist_results: specialist_id → ClinicalReasoningResult
            specialist_defs: specialist_id → SpecialistDef (for names)
            ledger: Cost tracker
            round_num: Current arbiter round (1-indexed)
        """
        patient_summary = self._format_profile(profile)
        specialist_text = self._format_specialist_outputs(specialist_results, specialist_defs)

        prompt = ARBITER_MERGE_PROMPT.format(
            patient_summary=patient_summary,
            round=round_num,
            specialist_outputs=specialist_text,
        )

        input_tokens = estimate_tokens(ARBITER_SYSTEM + prompt)
        t0 = time.monotonic()

        consensus = await self.medgemma.generate_structured(
            prompt=prompt,
            response_model=ClinicalReasoningResult,
            system_prompt=ARBITER_SYSTEM,
            temperature=self.config.arbiter_temperature,
            max_tokens=self.config.max_tokens_arbiter,
        )

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        output_tokens = estimate_tokens(str(consensus.model_dump_json()))

        ledger.calls.append(LLMCallRecord(
            call_id=str(uuid.uuid4())[:8],
            track_id=ledger.track_id,
            step_name=f"arbiter_merge_round{round_num}",
            iteration=round_num,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            latency_ms=elapsed_ms,
            temperature=self.config.arbiter_temperature,
            max_tokens_requested=self.config.max_tokens_arbiter,
            estimated_cost_usd=estimate_cost(input_tokens, output_tokens),
            timestamp=time.time(),
        ))

        logger.info(
            f"  [Arbiter round {round_num}] consensus: "
            f"{len(consensus.differential_diagnosis)} diagnoses, "
            f"top: {consensus.differential_diagnosis[0].diagnosis if consensus.differential_diagnosis else '?'}"
        )
        return consensus

    async def generate_feedback(
        self,
        profile: PatientProfile,
        consensus: ClinicalReasoningResult,
        specialist_results: Dict[str, ClinicalReasoningResult],
        specialist_defs: Dict[str, SpecialistDef],
        ledger: CostLedger,
        round_num: int = 1,
    ) -> Dict[str, str]:
        """
        Generate tailored resubmission prompts for specialists that
        disagree with the consensus.

        Returns specialist_id → feedback string.
        """
        consensus_top = {
            dx.diagnosis.lower().strip()
            for dx in consensus.differential_diagnosis[:3]
        }
        patient_summary = self._format_profile(profile)
        consensus_text = self._format_differential(consensus)

        feedback: Dict[str, str] = {}

        for sid, result in specialist_results.items():
            # Check if this specialist agrees with the consensus top-3
            spec_top = {
                dx.diagnosis.lower().strip()
                for dx in result.differential_diagnosis[:3]
            }
            overlap = len(consensus_top & spec_top)
            # If strong agreement, no feedback needed
            if overlap >= 2:
                continue

            spec_def = specialist_defs.get(sid)
            if not spec_def:
                continue

            prompt = ARBITER_FEEDBACK_PROMPT.format(
                patient_summary=patient_summary,
                round=round_num,
                consensus=consensus_text,
                specialist_name=spec_def.name,
                specialist_id=sid,
                specialist_output=self._format_differential(result),
            )

            input_tokens = estimate_tokens(ARBITER_SYSTEM + prompt)
            t0 = time.monotonic()

            fb_text = await self.medgemma.generate(
                prompt=prompt,
                system_prompt=ARBITER_SYSTEM,
                temperature=0.3,
                max_tokens=512,
            )

            elapsed_ms = int((time.monotonic() - t0) * 1000)
            output_tokens = estimate_tokens(fb_text)

            ledger.calls.append(LLMCallRecord(
                call_id=str(uuid.uuid4())[:8],
                track_id=ledger.track_id,
                step_name=f"arbiter_feedback_{sid}",
                iteration=round_num,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                latency_ms=elapsed_ms,
                temperature=0.3,
                max_tokens_requested=512,
                estimated_cost_usd=estimate_cost(input_tokens, output_tokens),
                timestamp=time.time(),
            ))

            feedback[sid] = fb_text
            logger.info(f"  [Arbiter → {spec_def.name}] feedback generated ({len(fb_text)} chars)")

        return feedback

    @staticmethod
    def _format_profile(profile: PatientProfile) -> str:
        parts = [
            f"Age: {profile.age or 'Unknown'}, Gender: {profile.gender.value}",
            f"Chief Complaint: {profile.chief_complaint}",
            f"HPI: {profile.history_of_present_illness or 'N/A'}",
        ]
        if profile.past_medical_history:
            parts.append(f"PMH: {', '.join(profile.past_medical_history)}")
        return "\n".join(parts)

    @staticmethod
    def _format_specialist_outputs(
        results: Dict[str, ClinicalReasoningResult],
        defs: Dict[str, SpecialistDef],
    ) -> str:
        sections = []
        for sid, result in results.items():
            name = defs[sid].name if sid in defs else sid
            lines = [f"─── {name} ({sid}) ───"]
            for i, dx in enumerate(result.differential_diagnosis, 1):
                lines.append(
                    f"  {i}. {dx.diagnosis} ({dx.likelihood.value}) — {dx.reasoning}"
                )
            if result.risk_assessment:
                lines.append(f"  Risk: {result.risk_assessment}")
            sections.append("\n".join(lines))
        return "\n\n".join(sections)

    @staticmethod
    def _format_differential(reasoning: ClinicalReasoningResult) -> str:
        lines = []
        for i, dx in enumerate(reasoning.differential_diagnosis, 1):
            lines.append(f"{i}. {dx.diagnosis} ({dx.likelihood.value}) — {dx.reasoning}")
        return "\n".join(lines) if lines else "(empty)"
