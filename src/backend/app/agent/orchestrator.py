# [Track A: Baseline]
"""
Agent Orchestrator — the brain of the CDS Agent.

Controls the multi-step pipeline:
  1. Parse patient data (MedGemma)
  2. Clinical reasoning / differential diagnosis (MedGemma)
  3. Drug interaction check (OpenFDA + RxNorm APIs)
  4. Guideline retrieval (RAG over ChromaDB)
  5. Conflict detection (MedGemma)
  6. Synthesis into CDS report (MedGemma)

Each step is a tool call. The orchestrator manages state, handles errors,
and streams step updates to the frontend via a callback.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime
from typing import AsyncGenerator, Callable, Optional

from app.models.schemas import (
    AgentState,
    AgentStep,
    AgentStepStatus,
    CaseSubmission,
    CDSReport,
)
from app.tools.patient_parser import PatientParserTool
from app.tools.clinical_reasoning import ClinicalReasoningTool
from app.tools.drug_interactions import DrugInteractionTool
from app.tools.guideline_retrieval import GuidelineRetrievalTool
from app.tools.conflict_detection import ConflictDetectionTool
from app.tools.synthesis import SynthesisTool


# Type for the callback that streams step updates
StepCallback = Callable[[AgentStep], None]


class Orchestrator:
    """
    Orchestrates the clinical decision support agent pipeline.

    Usage:
        orchestrator = Orchestrator()
        async for step_update in orchestrator.run(case):
            # stream step_update to frontend
            ...
        result = orchestrator.get_result()
    """

    def __init__(self):
        # Initialize tools
        self.patient_parser = PatientParserTool()
        self.clinical_reasoning = ClinicalReasoningTool()
        self.drug_interaction = DrugInteractionTool()
        self.guideline_retrieval = GuidelineRetrievalTool()
        self.conflict_detection = ConflictDetectionTool()
        self.synthesis = SynthesisTool()

        # State
        self._state: Optional[AgentState] = None

    @property
    def state(self) -> Optional[AgentState]:
        return self._state

    def _create_steps(self, case: CaseSubmission) -> list[AgentStep]:
        """Define the pipeline steps based on the case configuration."""
        steps = [
            AgentStep(
                step_id="parse",
                step_name="Parsing Patient Data",
                tool_name="patient_parser",
            ),
            AgentStep(
                step_id="reason",
                step_name="Clinical Reasoning",
                tool_name="clinical_reasoning",
            ),
        ]
        if case.include_drug_check:
            steps.append(
                AgentStep(
                    step_id="drugs",
                    step_name="Drug Interaction Check",
                    tool_name="drug_interactions",
                )
            )
        if case.include_guidelines:
            steps.append(
                AgentStep(
                    step_id="guidelines",
                    step_name="Guideline Retrieval",
                    tool_name="guideline_retrieval",
                )
            )
        if case.include_guidelines:
            steps.append(
                AgentStep(
                    step_id="conflicts",
                    step_name="Conflict Detection",
                    tool_name="conflict_detection",
                )
            )
        steps.append(
            AgentStep(
                step_id="synthesize",
                step_name="Synthesizing Report",
                tool_name="synthesis",
            )
        )
        return steps

    async def run(self, case: CaseSubmission) -> AsyncGenerator[AgentStep, None]:
        """
        Run the full agent pipeline. Yields step updates as they happen.

        This is the main entry point. Each step is executed sequentially,
        with state flowing from one step to the next. Steps that don't
        depend on each other (drug check + guidelines) run in parallel.

        If a critical step (parse, reason) fails, subsequent dependent
        steps are marked as SKIPPED to avoid cascading errors.
        """
        case_id = str(uuid.uuid4())[:8]
        steps = self._create_steps(case)

        self._state = AgentState(
            case_id=case_id,
            steps=steps,
            started_at=datetime.utcnow(),
        )

        try:
            # ── Step 1: Parse patient data ──
            yield self._mark_running("parse")
            step = await self._run_step("parse", self._step_parse, case.patient_text)
            yield step

            if step.status == AgentStepStatus.FAILED:
                # Can't continue without patient profile — skip remaining steps
                for skipped in self._skip_remaining_steps("parse"):
                    yield skipped
                self._state.completed_at = datetime.utcnow()
                return

            # ── Step 2: Clinical reasoning ──
            yield self._mark_running("reason")
            step = await self._run_step("reason", self._step_reason)
            yield step

            if step.status == AgentStepStatus.FAILED:
                for skipped in self._skip_remaining_steps("reason"):
                    yield skipped
                self._state.completed_at = datetime.utcnow()
                return

            # ── Step 3 & 4: Drug check + Guidelines (parallel) ──
            parallel_tasks = []
            if case.include_drug_check:
                yield self._mark_running("drugs")
                parallel_tasks.append(("drugs", self._step_drug_check))
            if case.include_guidelines:
                yield self._mark_running("guidelines")
                parallel_tasks.append(("guidelines", self._step_guidelines))

            if parallel_tasks:
                results = await asyncio.gather(
                    *[self._run_step(sid, fn) for sid, fn in parallel_tasks],
                    return_exceptions=True,
                )
                for result in results:
                    if isinstance(result, Exception):
                        # Log but don't fail — graceful degradation
                        pass
                    else:
                        yield result

            # ── Step 5: Conflict Detection ──
            if case.include_guidelines:
                yield self._mark_running("conflicts")
                yield await self._run_step("conflicts", self._step_conflict_detection)

            # ── Step 6: Synthesis ──
            yield self._mark_running("synthesize")
            yield await self._run_step("synthesize", self._step_synthesize)

            self._state.completed_at = datetime.utcnow()

        except Exception as e:
            # Mark remaining steps as failed
            for step in self._state.steps:
                if step.status == AgentStepStatus.PENDING:
                    step.status = AgentStepStatus.FAILED
                    step.error = f"Pipeline aborted: {str(e)}"
            raise

    def _skip_remaining_steps(self, after_step_id: str) -> list[AgentStep]:
        """Mark all steps after after_step_id as skipped. Returns them for yielding."""
        skipped = []
        found = False
        for step in self._state.steps:
            if step.step_id == after_step_id:
                found = True
                continue
            if found and step.status == AgentStepStatus.PENDING:
                step.status = AgentStepStatus.SKIPPED
                step.error = f"Skipped: prerequisite step '{after_step_id}' failed"
                skipped.append(step)
        return skipped

    def _mark_running(self, step_id: str) -> AgentStep:
        """Mark a step as RUNNING and return it for immediate yielding."""
        step = self._get_step(step_id)
        step.status = AgentStepStatus.RUNNING
        return step

    async def _run_step(self, step_id: str, fn, *args) -> AgentStep:
        """Execute a single step, tracking status and timing."""
        step = self._get_step(step_id)
        step.status = AgentStepStatus.RUNNING
        start = time.monotonic()

        try:
            await fn(*args)
            step.status = AgentStepStatus.COMPLETED
        except Exception as e:
            step.status = AgentStepStatus.FAILED
            step.error = str(e)
        finally:
            step.duration_ms = int((time.monotonic() - start) * 1000)

        return step

    def _get_step(self, step_id: str) -> AgentStep:
        for step in self._state.steps:
            if step.step_id == step_id:
                return step
        raise ValueError(f"Unknown step: {step_id}")

    # ──────────────────────────────────────────────
    # Step implementations
    # ──────────────────────────────────────────────

    async def _step_parse(self, patient_text: str):
        """Step 1: Parse raw patient text into structured profile."""
        profile = await self.patient_parser.run(patient_text)
        self._state.patient_profile = profile

        step = self._get_step("parse")
        step.input_summary = patient_text[:100] + "..." if len(patient_text) > 100 else patient_text
        step.output_summary = f"Parsed: {profile.chief_complaint}, {len(profile.current_medications)} meds, {len(profile.lab_results)} labs"

    async def _step_reason(self):
        """Step 2: Clinical reasoning over the structured patient profile."""
        if not self._state.patient_profile:
            raise RuntimeError("Patient profile not available — parse step must run first")

        result = await self.clinical_reasoning.run(self._state.patient_profile)
        self._state.clinical_reasoning = result

        step = self._get_step("reason")
        step.output_summary = (
            f"{len(result.differential_diagnosis)} diagnoses, "
            f"{len(result.recommended_workup)} recommendations"
        )

    async def _step_drug_check(self):
        """Step 3: Check drug interactions for current + proposed medications."""
        if not self._state.patient_profile:
            raise RuntimeError("Patient profile not available")

        meds = self._state.patient_profile.current_medications
        # Also include any medications proposed by the reasoning step
        proposed_meds = []
        if self._state.clinical_reasoning:
            for action in self._state.clinical_reasoning.recommended_workup:
                if "medication" in action.action.lower() or "prescribe" in action.action.lower():
                    proposed_meds.append(action.action)

        result = await self.drug_interaction.run(meds, proposed_meds)
        self._state.drug_interactions = result

        step = self._get_step("drugs")
        step.output_summary = f"{len(result.interactions_found)} interactions found"

    async def _step_guidelines(self):
        """Step 4: Retrieve relevant clinical guidelines via RAG."""
        if not self._state.clinical_reasoning:
            raise RuntimeError("Clinical reasoning not available")

        # Build query from the top diagnosis
        top_dx = self._state.clinical_reasoning.differential_diagnosis
        if top_dx:
            query = f"{top_dx[0].diagnosis} clinical guidelines management"
        else:
            query = self._state.patient_profile.chief_complaint + " clinical guidelines"

        result = await self.guideline_retrieval.run(query)
        self._state.guideline_retrieval = result

        step = self._get_step("guidelines")
        step.output_summary = f"{len(result.excerpts)} guideline excerpts retrieved"

    async def _step_conflict_detection(self):
        """Step 5: Detect conflicts between guidelines and patient data."""
        result = await self.conflict_detection.run(
            patient_profile=self._state.patient_profile,
            clinical_reasoning=self._state.clinical_reasoning,
            drug_interactions=self._state.drug_interactions,
            guideline_retrieval=self._state.guideline_retrieval,
        )
        self._state.conflict_detection = result

        step = self._get_step("conflicts")
        n = len(result.conflicts)
        if n == 0:
            step.output_summary = "No conflicts detected"
        else:
            step.output_summary = f"{n} conflict(s) detected — {result.summary}"

    async def _step_synthesize(self):
        """Step 6: Synthesize all tool outputs into a final CDS report."""
        report = await self.synthesis.run(
            patient_profile=self._state.patient_profile,
            clinical_reasoning=self._state.clinical_reasoning,
            drug_interactions=self._state.drug_interactions,
            guideline_retrieval=self._state.guideline_retrieval,
            conflict_detection=self._state.conflict_detection,
        )
        self._state.final_report = report

        step = self._get_step("synthesize")
        step.output_summary = "Clinical Decision Support report generated"

    def get_result(self) -> Optional[CDSReport]:
        """Return the final report, if synthesis completed."""
        if self._state:
            return self._state.final_report
        return None
