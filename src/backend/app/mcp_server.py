"""
MCP (Model Context Protocol) server for the CDS Agent.

Exposes the clinical decision support pipeline as MCP tools and resources,
enabling any MCP-compatible client (Claude Desktop, VS Code Copilot, etc.)
to invoke clinical analysis directly.

Two interaction patterns:
  1. **Blocking** — `analyze_patient_case` runs the full 6-step pipeline
     and returns the complete CDSReport. Simple but slow (30-90s).
  2. **Async submit/poll** — `submit_case` kicks off the pipeline and
     returns a case_id immediately. `check_case_status` reports progress.
     `get_case_result` returns the final report when ready.

Individual pipeline steps are also exposed as standalone tools for
targeted use (e.g., just drug interaction checks).

Resources:
  - guidelines://list — all 62 clinical guidelines
  - guidelines://{specialty} — filtered by medical specialty

Start with:
    python -m app.mcp_server            # stdio transport (Claude Desktop)
    python -m app.mcp_server --sse      # SSE transport (web clients)
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from mcp.server import FastMCP

from app.agent.orchestrator import Orchestrator
from app.models.schemas import (
    AgentStepStatus,
    CaseSubmission,
    CDSReport,
)
from app.tools.fhir_adapter import FHIRAdapter, fhir_to_text
from app.tools.cds_hooks import cds_report_to_hooks_response

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Initialize MCP server
# ──────────────────────────────────────────────

mcp = FastMCP(
    "CDS Agent",
)

# In-memory store for async cases
_active_cases: Dict[str, Orchestrator] = {}
_case_timestamps: Dict[str, float] = {}
_CASE_TTL = 600  # 10 minutes

GUIDELINES_PATH = Path(__file__).parent / "data" / "clinical_guidelines.json"


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _evict_expired():
    """Remove cases older than TTL."""
    now = time.time()
    expired = [cid for cid, ts in _case_timestamps.items() if now - ts > _CASE_TTL]
    for cid in expired:
        _active_cases.pop(cid, None)
        _case_timestamps.pop(cid, None)


def _report_to_dict(report: CDSReport) -> dict:
    """Serialize a CDSReport to a clean dict for MCP text output."""
    return json.loads(report.model_dump_json())


def _format_step_progress(orchestrator: Orchestrator) -> dict:
    """Build a progress summary from orchestrator state."""
    state = orchestrator.state
    if not state:
        return {"status": "initializing", "steps_completed": 0, "total_steps": 0}

    steps_info: list[dict[str, Any]] = []
    completed = 0
    current_step = None
    for step in state.steps:
        info: dict[str, Any] = {
            "step_id": step.step_id,
            "name": step.step_name,
            "status": step.status.value,
        }
        if step.duration_ms is not None:
            info["duration_ms"] = step.duration_ms
        if step.output_summary:
            info["output_summary"] = step.output_summary
        if step.error:
            info["error"] = step.error
        steps_info.append(info)

        if step.status == AgentStepStatus.COMPLETED:
            completed += 1
        elif step.status == AgentStepStatus.RUNNING:
            current_step = step.step_name

    total = len(state.steps)
    is_done = state.completed_at is not None
    elapsed = None
    if state.started_at:
        end = state.completed_at or datetime.utcnow()
        elapsed = round((end - state.started_at).total_seconds(), 1)

    return {
        "case_id": state.case_id,
        "status": "completed" if is_done else ("running" if current_step else "pending"),
        "steps_completed": completed,
        "total_steps": total,
        "current_step": current_step,
        "elapsed_seconds": elapsed,
        "has_report": state.final_report is not None,
        "steps": steps_info,
    }


def _load_guidelines() -> list[dict]:
    """Load Clinical guidelines from JSON file."""
    if GUIDELINES_PATH.exists():
        return json.loads(GUIDELINES_PATH.read_text(encoding="utf-8"))
    return []


# ──────────────────────────────────────────────
# TOOL: Blocking full pipeline
# ──────────────────────────────────────────────

@mcp.tool()
async def analyze_patient_case(
    patient_text: str,
    include_drug_check: bool = True,
    include_guidelines: bool = True,
) -> str:
    """
    Run the full CDS pipeline on a patient case and return the complete report.

    This is a BLOCKING call — it runs all 6 pipeline steps sequentially
    and returns the final Clinical Decision Support report. Typical
    runtime is 30-90 seconds depending on endpoint load.

    Use this when you want to wait for the complete result in a single call.
    For long cases, consider using submit_case + check_case_status instead.

    Args:
        patient_text: Free-text patient case description. Include chief
            complaint, history, medications, labs, vitals, allergies, etc.
            Minimum 10 characters.
        include_drug_check: Whether to check drug interactions (default: True).
        include_guidelines: Whether to retrieve clinical guidelines (default: True).

    Returns:
        JSON string containing the full CDSReport with:
        - patient_summary: Concise patient summary
        - differential_diagnosis: Ranked diagnoses with likelihood and evidence
        - drug_interaction_warnings: Detected drug interactions
        - guideline_recommendations: Guideline-concordant recommendations
        - suggested_next_steps: Recommended workup and actions
        - conflicts: Guideline vs. patient data conflicts
        - caveats: Limitations and disclaimers
        - sources_cited: Referenced guidelines and sources
    """
    case = CaseSubmission(
        patient_text=patient_text,
        include_drug_check=include_drug_check,
        include_guidelines=include_guidelines,
    )
    orchestrator = Orchestrator()

    async for _step in orchestrator.run(case):
        pass  # Run all steps to completion

    report = orchestrator.get_result()
    if not report:
        return json.dumps({"error": "Pipeline completed but no report was generated."})

    result = _report_to_dict(report)
    result["_pipeline"] = _format_step_progress(orchestrator)
    return json.dumps(result, indent=2, default=str)


# ──────────────────────────────────────────────
# TOOLS: Async submit / poll / retrieve
# ──────────────────────────────────────────────

@mcp.tool()
async def submit_case(
    patient_text: str,
    include_drug_check: bool = True,
    include_guidelines: bool = True,
) -> str:
    """
    Submit a patient case for asynchronous analysis. Returns immediately
    with a case_id — use check_case_status to poll progress and
    get_case_result to retrieve the final report.

    Use this pattern instead of analyze_patient_case when:
    - You want to show the user progress updates as each step completes
    - You're worried about timeout on the blocking call
    - You want to submit multiple cases in quick succession

    Args:
        patient_text: Free-text patient case description (minimum 10 chars).
        include_drug_check: Whether to check drug interactions.
        include_guidelines: Whether to retrieve clinical guidelines.

    Returns:
        JSON with case_id and instructions for polling.
    """
    _evict_expired()

    case = CaseSubmission(
        patient_text=patient_text,
        include_drug_check=include_drug_check,
        include_guidelines=include_guidelines,
    )
    orchestrator = Orchestrator()

    async def _run():
        async for _step in orchestrator.run(case):
            pass

    asyncio.create_task(_run())
    await asyncio.sleep(0.15)  # Let orchestrator initialize state

    case_id = orchestrator.state.case_id if orchestrator.state else "unknown"
    _active_cases[case_id] = orchestrator
    _case_timestamps[case_id] = time.time()

    return json.dumps({
        "case_id": case_id,
        "status": "running",
        "message": (
            f"Pipeline started. Use check_case_status('{case_id}') to poll progress, "
            f"then get_case_result('{case_id}') to retrieve the report when done."
        ),
    })


@mcp.tool()
async def check_case_status(case_id: str) -> str:
    """
    Check the progress of a previously submitted case.

    Returns the current pipeline status including which step is running,
    how many steps are complete, and elapsed time. Call this periodically
    (every 5-10 seconds) after submit_case until status is 'completed'.

    Args:
        case_id: The case_id returned by submit_case.

    Returns:
        JSON with status, progress (e.g., "4/6 steps"), current step name,
        elapsed time, and per-step details.
    """
    _evict_expired()

    orchestrator = _active_cases.get(case_id)
    if not orchestrator:
        return json.dumps({
            "error": f"Case '{case_id}' not found. It may have expired (TTL: {_CASE_TTL}s) or never existed.",
        })

    return json.dumps(_format_step_progress(orchestrator), indent=2, default=str)


@mcp.tool()
async def get_case_result(case_id: str, format: str = "report") -> str:
    """
    Retrieve the final CDS report for a completed case.

    Call this after check_case_status shows status 'completed'.
    If the pipeline is still running, returns the current progress instead.

    Args:
        case_id: The case_id returned by submit_case.
        format: Output format — 'report' (default) for the CDSReport,
                'cds_hooks' for CDS Hooks 2.0 card format.

    Returns:
        JSON containing the full CDSReport or CDS Hooks response, plus
        pipeline execution metadata.
    """
    _evict_expired()

    orchestrator = _active_cases.get(case_id)
    if not orchestrator:
        return json.dumps({
            "error": f"Case '{case_id}' not found. It may have expired (TTL: {_CASE_TTL}s) or never existed.",
        })

    progress = _format_step_progress(orchestrator)

    if not progress.get("has_report"):
        return json.dumps({
            "status": "not_ready",
            "message": "Pipeline has not completed yet. Call check_case_status to see progress.",
            "progress": progress,
        }, indent=2, default=str)

    report = orchestrator.get_result()
    if report is None:
        return json.dumps({"error": "Pipeline completed but no report was generated."})

    if format == "cds_hooks":
        result = cds_report_to_hooks_response(report)
    else:
        result = _report_to_dict(report)

    result["_pipeline"] = progress
    return json.dumps(result, indent=2, default=str)


# ──────────────────────────────────────────────
# TOOLS: Individual pipeline steps
# ──────────────────────────────────────────────

@mcp.tool()
async def parse_patient_data(patient_text: str) -> str:
    """
    Parse free-text patient information into a structured PatientProfile.

    Extracts: demographics, chief complaint, HPI, past medical history,
    medications (with RxCUI codes), allergies, lab results, vital signs,
    social/family history.

    This is Step 1 of the pipeline — useful standalone for structuring
    unstructured clinical notes.

    Args:
        patient_text: Free-text patient case description.

    Returns:
        JSON PatientProfile with structured fields.
    """
    from app.tools.patient_parser import PatientParserTool
    parser = PatientParserTool()
    profile = await parser.run(patient_text)
    return json.dumps(json.loads(profile.model_dump_json()), indent=2, default=str)


@mcp.tool()
async def get_differential_diagnosis(patient_text: str) -> str:
    """
    Generate a differential diagnosis for a patient case.

    Runs Step 1 (parse) + Step 2 (clinical reasoning) of the pipeline.
    Returns ranked diagnoses with ICD-10 codes, likelihood estimates,
    supporting evidence, and reasoning chains.

    Args:
        patient_text: Free-text patient case description.

    Returns:
        JSON with differential_diagnosis list, risk_assessment,
        recommended_workup, and reasoning_chain.
    """
    from app.tools.patient_parser import PatientParserTool
    from app.tools.clinical_reasoning import ClinicalReasoningTool

    parser = PatientParserTool()
    profile = await parser.run(patient_text)

    reasoner = ClinicalReasoningTool()
    result = await reasoner.run(profile)

    return json.dumps(json.loads(result.model_dump_json()), indent=2, default=str)


@mcp.tool()
async def check_drug_interactions(medications: list[str]) -> str:
    """
    Check for drug-drug interactions among a list of medications.

    Queries OpenFDA and RxNorm for known interactions. Returns severity
    ratings (low/moderate/high/critical) and clinical significance.

    Args:
        medications: List of medication names, e.g. ["lisinopril", "metformin", "warfarin"].

    Returns:
        JSON with interactions_found (drug pairs + severity), medications_checked,
        and any warnings.
    """
    from app.tools.drug_interactions import DrugInteractionTool
    from app.models.schemas import Medication

    med_objects = [Medication(name=m, dose=None, rxcui=None) for m in medications]
    checker = DrugInteractionTool()
    result = await checker.run(med_objects, [])

    return json.dumps(json.loads(result.model_dump_json()), indent=2, default=str)


@mcp.tool()
async def retrieve_guidelines(query: str) -> str:
    """
    Retrieve relevant clinical guidelines for a medical query.

    Uses RAG (retrieval-augmented generation) over a curated knowledge base
    of 62 clinical guidelines across 14 specialties. Returns the most
    relevant guideline excerpts with sources and relevance scores.

    Args:
        query: Clinical query, e.g. "hypertension management in diabetic patient"
               or "chest pain evaluation guidelines".

    Returns:
        JSON with matched guideline excerpts, sources, and relevance scores.
    """
    from app.tools.guideline_retrieval import GuidelineRetrievalTool

    retriever = GuidelineRetrievalTool()
    result = await retriever.run(query)

    return json.dumps(json.loads(result.model_dump_json()), indent=2, default=str)


@mcp.tool()
async def analyze_fhir_bundle(fhir_bundle: dict) -> str:
    """
    Analyze a FHIR R4 Bundle through the full CDS pipeline.

    Accepts a FHIR Bundle containing Patient, Condition, MedicationStatement,
    Observation, and AllergyIntolerance resources. Converts to a structured
    patient profile and runs the complete 6-step analysis pipeline.

    Args:
        fhir_bundle: FHIR R4 Bundle JSON object, or a single FHIR resource dict.

    Returns:
        JSON CDSReport with differential diagnosis, drug interactions,
        guideline recommendations, conflicts, and suggested next steps.
    """
    patient_text = fhir_to_text(fhir_bundle)
    if not patient_text or len(patient_text.strip()) < 10:
        return json.dumps({"error": "FHIR bundle did not contain enough clinical data to analyze."})

    case = CaseSubmission(
        patient_text=patient_text,
        include_drug_check=True,
        include_guidelines=True,
    )
    orchestrator = Orchestrator()

    async for _step in orchestrator.run(case):
        pass

    report = orchestrator.get_result()
    if not report:
        return json.dumps({"error": "Pipeline completed but no report was generated."})

    result = _report_to_dict(report)
    result["_fhir_input_summary"] = patient_text[:500]
    result["_pipeline"] = _format_step_progress(orchestrator)
    return json.dumps(result, indent=2, default=str)


# ──────────────────────────────────────────────
# RESOURCES: Clinical guidelines
# ──────────────────────────────────────────────

@mcp.resource("guidelines://list")
async def list_guidelines() -> str:
    """
    List all clinical guidelines in the knowledge base.

    Returns summary of all 62 guidelines across 14 specialties with
    titles, sources, and specialty classifications.
    """
    guidelines = _load_guidelines()
    summary = []
    for g in guidelines:
        summary.append({
            "id": g["id"],
            "specialty": g["specialty"],
            "title": g["title"],
            "source": g.get("source", ""),
        })
    return json.dumps({
        "total_guidelines": len(summary),
        "specialties": sorted(set(g["specialty"] for g in summary)),
        "guidelines": summary,
    }, indent=2)


@mcp.resource("guidelines://{specialty}")
async def get_guidelines_by_specialty(specialty: str) -> str:
    """
    Get clinical guidelines filtered by medical specialty.

    Available specialties include: Cardiology, Endocrinology, Pulmonology,
    Nephrology, Gastroenterology, Neurology, Infectious Disease,
    Hematology, Oncology, Rheumatology, Psychiatry, Emergency Medicine,
    Pediatrics, Obstetrics.

    Returns full guideline text for the requested specialty.
    """
    guidelines = _load_guidelines()
    matches = [
        g for g in guidelines
        if g.get("specialty", "").lower() == specialty.lower()
    ]

    if not matches:
        all_specialties = sorted(set(g.get("specialty", "") for g in guidelines))
        return json.dumps({
            "error": f"No guidelines found for specialty '{specialty}'.",
            "available_specialties": all_specialties,
        }, indent=2)

    return json.dumps({
        "specialty": specialty,
        "count": len(matches),
        "guidelines": matches,
    }, indent=2)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main():
    """Run the MCP server."""
    transport = "stdio"
    if "--sse" in sys.argv:
        transport = "sse"

    logger.info(f"Starting CDS Agent MCP server (transport={transport})")
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
