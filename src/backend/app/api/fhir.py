"""
FHIR R4 and CDS Hooks API endpoints.

Provides:
- POST /api/fhir/analyze       — Accept a FHIR Bundle and run the CDS pipeline
- GET  /api/cds-hooks/services  — CDS Hooks service discovery
- POST /api/cds-hooks/cds-agent — CDS Hooks invocation (patient-view hook)
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from app.agent.orchestrator import Orchestrator
from app.models.schemas import CaseResponse, CaseSubmission
from app.tools.cds_hooks import cds_report_to_hooks_response, get_cds_hooks_discovery
from app.tools.fhir_adapter import FHIRAdapter, fhir_to_text

router = APIRouter()


# ──────────────────────────────────────────────
# FHIR R4 input endpoint
# ──────────────────────────────────────────────

@router.post("/fhir/analyze", response_model=CaseResponse)
async def analyze_fhir_bundle(bundle: Dict[str, Any]):
    """
    Accept a FHIR R4 Bundle (or single resource) and run the CDS pipeline.

    The FHIR resources are converted to a free-text patient summary,
    then processed through the standard 6-step pipeline.

    Accepts:
        - FHIR Bundle (resourceType: "Bundle")
        - Single FHIR resource (e.g., Patient, Condition)

    Returns:
        CaseResponse with case_id for polling/WebSocket updates.
    """
    try:
        adapter = FHIRAdapter()
        patient_text = fhir_to_text(bundle)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Failed to parse FHIR resource: {str(e)}",
        )

    if not patient_text or len(patient_text.strip()) < 10:
        raise HTTPException(
            status_code=422,
            detail="FHIR Bundle did not contain enough clinical data to analyze.",
        )

    # Create a CaseSubmission from the extracted text
    case = CaseSubmission(
        patient_text=patient_text,
        include_drug_check=True,
        include_guidelines=True,
    )
    orchestrator = Orchestrator()
    case_id = str(uuid.uuid4())[:8]

    async def _run_pipeline():
        async for _step in orchestrator.run(case):
            pass

    asyncio.create_task(_run_pipeline())
    await asyncio.sleep(0.15)

    actual_id = orchestrator.state.case_id if orchestrator.state else case_id

    return CaseResponse(
        case_id=actual_id,
        status="running",
        message="FHIR data parsed. Agent pipeline started.",
    )


# ──────────────────────────────────────────────
# CDS Hooks — Discovery endpoint
# ──────────────────────────────────────────────

@router.get("/cds-hooks/services")
async def cds_hooks_discovery():
    """
    CDS Hooks discovery endpoint.

    Returns the list of CDS services available, per the
    CDS Hooks 2.0 specification.
    """
    return get_cds_hooks_discovery()


# ──────────────────────────────────────────────
# CDS Hooks — Service invocation
# ──────────────────────────────────────────────

@router.post("/cds-hooks/cds-agent")
async def cds_hooks_invoke(request: Dict[str, Any]):
    """
    CDS Hooks service invocation for the patient-view hook.

    Expects a CDS Hooks request with:
    {
        "hookInstance": "...",
        "hook": "patient-view",
        "context": { "patientId": "..." },
        "prefetch": {
            "patient": { ... FHIR Patient ... },
            "conditions": { ... FHIR Bundle ... },
            "medications": { ... FHIR Bundle ... },
            ...
        }
    }

    Returns: CDS Hooks response with cards.
    """
    # Validate hook
    hook = request.get("hook")
    if hook != "patient-view":
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported hook: {hook}. Only 'patient-view' is supported.",
        )

    prefetch = request.get("prefetch", {})
    if not prefetch:
        raise HTTPException(
            status_code=400,
            detail="No prefetch data provided. Cannot generate CDS without patient data.",
        )

    # Build a combined FHIR bundle from prefetch resources
    entries = []
    for _key, resource in prefetch.items():
        if isinstance(resource, dict):
            resource_type = resource.get("resourceType", "")
            if resource_type == "Bundle":
                # Extract entries from sub-bundles
                for entry in resource.get("entry", []):
                    if "resource" in entry:
                        entries.append({"resource": entry["resource"]})
            else:
                entries.append({"resource": resource})

    combined_bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": entries,
    }

    # Convert FHIR to text and run pipeline
    try:
        patient_text = fhir_to_text(combined_bundle)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Failed to parse prefetch FHIR data: {str(e)}",
        )

    if not patient_text or len(patient_text.strip()) < 10:
        # Return empty cards if insufficient data
        return {"cards": []}

    # Run the pipeline synchronously for CDS Hooks (needs immediate response)
    case = CaseSubmission(
        patient_text=patient_text,
        include_drug_check=True,
        include_guidelines=True,
    )
    orchestrator = Orchestrator()

    try:
        async for _step in orchestrator.run(case):
            pass
    except Exception as e:
        # Return an error card rather than crashing
        return {
            "cards": [{
                "uuid": str(uuid.uuid4()),
                "summary": "CDS Agent encountered an error",
                "detail": f"Pipeline error: {str(e)}",
                "indicator": "warning",
                "source": {"label": "CDS Agent"},
            }],
        }

    report = orchestrator.get_result()
    if not report:
        return {"cards": []}

    return cds_report_to_hooks_response(report, service_id="cds-agent")
