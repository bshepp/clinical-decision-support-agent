# [Track C: Iterative Refinement]
"""
Track C — Core refinement loop.

On each iteration:
  1. Feed the current differential to a self-critique prompt
  2. Ask the model to identify weaknesses, missing diagnoses, and re-rank
  3. Compare the new differential to the previous one
  4. If converged (similarity > threshold) or max iters reached, stop
  5. Otherwise loop

Each LLM call is recorded in the CostLedger for cost/benefit charting.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import List, Optional, Tuple

from app.models.schemas import (
    ClinicalReasoningResult,
    DiagnosisCandidate,
    PatientProfile,
)
from app.services.medgemma import MedGemmaService
from tracks.iterative.config import IterativeConfig
from tracks.shared.cost_tracker import (
    CostLedger,
    estimate_cost,
    estimate_tokens,
    LLMCallRecord,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Self-critique prompt
# ──────────────────────────────────────────────

CRITIQUE_SYSTEM = """You are a senior physician reviewing a colleague's differential diagnosis.
Your goal is to find weaknesses, missing diagnoses, and ranking errors.
Be constructive but rigorous. You may add, remove, reorder, or refine diagnoses.
Always justify changes with clinical evidence from the patient data."""

CRITIQUE_PROMPT = """
PATIENT PROFILE:
{patient_summary}

CURRENT DIFFERENTIAL (iteration {iteration}):
{current_differential}

INSTRUCTIONS:
1. For each diagnosis, state whether you AGREE, DISAGREE, or want to MODIFY it.
2. Identify any MISSING diagnoses that should be on the list.
3. Propose a REVISED differential diagnosis, ranked by likelihood.
4. Explain your chain-of-thought reasoning for every change.
5. Rate each diagnosis as "low", "moderate", or "high" likelihood.

Return ONLY the revised differential as a JSON object matching
ClinicalReasoningResult schema (differential_diagnosis, risk_assessment,
recommended_workup, reasoning_chain).
"""


class IterativeRefiner:
    """
    Runs the iterative self-critique loop for a single patient case.

    Usage:
        refiner = IterativeRefiner(config, ledger)
        final, history = await refiner.refine(profile, initial_reasoning)
    """

    def __init__(self, config: IterativeConfig, ledger: CostLedger):
        self.config = config
        self.ledger = ledger
        self.medgemma = MedGemmaService()

    async def refine(
        self,
        profile: PatientProfile,
        initial_reasoning: ClinicalReasoningResult,
    ) -> Tuple[ClinicalReasoningResult, List[ClinicalReasoningResult]]:
        """
        Iteratively refine the differential diagnosis.

        Args:
            profile: Structured patient data
            initial_reasoning: Track A's first-pass reasoning result

        Returns:
            (final_reasoning, iteration_history) — history[0] is the initial,
            history[-1] is the final.
        """
        history: List[ClinicalReasoningResult] = [initial_reasoning]
        current = initial_reasoning

        patient_summary = self._format_profile(profile)

        for iteration in range(1, self.config.max_iterations + 1):
            logger.info(
                f"  [Iteration {iteration}/{self.config.max_iterations}] "
                f"Current top dx: {self._top_dx(current)}"
            )

            # Build the critique prompt
            diff_text = self._format_differential(current)
            prompt = CRITIQUE_PROMPT.format(
                patient_summary=patient_summary,
                iteration=iteration,
                current_differential=diff_text,
            )

            # Call MedGemma for the self-critique
            input_tokens = estimate_tokens(CRITIQUE_SYSTEM + prompt)
            t0 = time.monotonic()

            try:
                revised = await self.medgemma.generate_structured(
                    prompt=prompt,
                    response_model=ClinicalReasoningResult,
                    system_prompt=CRITIQUE_SYSTEM,
                    temperature=self.config.critique_temperature,
                    max_tokens=self.config.max_tokens_critique,
                )
            except (ValueError, Exception) as e:
                logger.warning(
                    f"  [Iteration {iteration}] Structured parse failed, "
                    f"stopping refinement early: {e}"
                )
                # Return best result so far instead of crashing
                return current, history

            elapsed_ms = int((time.monotonic() - t0) * 1000)
            output_tokens = estimate_tokens(str(revised.model_dump_json()))

            # Record cost
            self.ledger.calls.append(LLMCallRecord(
                call_id=str(uuid.uuid4())[:8],
                track_id=self.ledger.track_id,
                step_name="iterative_critique",
                iteration=iteration,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                latency_ms=elapsed_ms,
                temperature=self.config.critique_temperature,
                max_tokens_requested=self.config.max_tokens_critique,
                estimated_cost_usd=estimate_cost(input_tokens, output_tokens),
                timestamp=time.time(),
            ))

            history.append(revised)

            # Check convergence
            if self._has_converged(current, revised):
                logger.info(
                    f"  Converged at iteration {iteration} "
                    f"(threshold {self.config.convergence_threshold})"
                )
                return revised, history

            current = revised

        logger.info(f"  Reached max iterations ({self.config.max_iterations})")
        return current, history

    def _has_converged(
        self,
        prev: ClinicalReasoningResult,
        curr: ClinicalReasoningResult,
    ) -> bool:
        """
        Check if the differential is stable between two iterations.

        Uses simple top-N diagnosis name overlap as a proxy for convergence.
        If the top diagnoses haven't changed, the model is repeating itself.
        """
        prev_names = {dx.diagnosis.lower().strip() for dx in prev.differential_diagnosis[:5]}
        curr_names = {dx.diagnosis.lower().strip() for dx in curr.differential_diagnosis[:5]}

        if not prev_names or not curr_names:
            return False

        overlap = len(prev_names & curr_names)
        union = len(prev_names | curr_names)
        jaccard = overlap / union if union > 0 else 0.0

        return jaccard >= (1 - self.config.convergence_threshold)

    @staticmethod
    def _top_dx(reasoning: ClinicalReasoningResult) -> str:
        if reasoning.differential_diagnosis:
            return reasoning.differential_diagnosis[0].diagnosis
        return "(empty)"

    @staticmethod
    def _format_profile(profile: PatientProfile) -> str:
        parts = [
            f"Age: {profile.age or 'Unknown'}, Gender: {profile.gender.value}",
            f"Chief Complaint: {profile.chief_complaint}",
            f"HPI: {profile.history_of_present_illness or 'N/A'}",
        ]
        if profile.past_medical_history:
            parts.append(f"PMH: {', '.join(profile.past_medical_history)}")
        if profile.current_medications:
            meds = "; ".join(f"{m.name} {m.dose or ''}" for m in profile.current_medications)
            parts.append(f"Medications: {meds}")
        return "\n".join(parts)

    @staticmethod
    def _format_differential(reasoning: ClinicalReasoningResult) -> str:
        lines = []
        for i, dx in enumerate(reasoning.differential_diagnosis, 1):
            lines.append(
                f"{i}. {dx.diagnosis} (likelihood: {dx.likelihood.value}) — {dx.reasoning}"
            )
        if reasoning.risk_assessment:
            lines.append(f"\nRisk Assessment: {reasoning.risk_assessment}")
        return "\n".join(lines) if lines else "(no differential generated)"
