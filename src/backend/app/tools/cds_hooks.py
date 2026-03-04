"""
CDS Hooks response generation.

Converts a CDSReport into the HL7 CDS Hooks response format, producing
structured "cards" that can be rendered in an EHR's clinical workflow.

Reference: https://cds-hooks.hl7.org/2.0/

CDS Hooks response structure:
{
  "cards": [
    {
      "uuid": "...",
      "summary": "Short summary (<=140 chars)",
      "detail": "Markdown detail",
      "indicator": "info" | "warning" | "critical",
      "source": { "label": "...", "url": "..." },
      "suggestions": [ { "label": "...", "actions": [...] } ],
      "links": [ { "label": "...", "url": "...", "type": "absolute" } ]
    }
  ]
}

Cards generated:
  1. Differential Diagnosis card — top diagnoses with reasoning
  2. Drug Interaction cards — one per high/critical interaction
  3. Guideline Recommendations card — key guideline-based suggestions
  4. Conflict/Alert cards — one per detected conflict
  5. Suggested Next Steps card — recommended workup/actions
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.models.schemas import (
    CDSReport,
    ClinicalConflict,
    DiagnosisCandidate,
    DrugInteraction,
    RecommendedAction,
    Severity,
)


# ──────────────────────────────────────────────
# CDS Hooks indicator mapping
# ──────────────────────────────────────────────

def _severity_to_indicator(severity: Severity) -> str:
    """Map internal Severity enum to CDS Hooks indicator."""
    mapping = {
        Severity.LOW: "info",
        Severity.MODERATE: "warning",
        Severity.HIGH: "critical",
        Severity.CRITICAL: "critical",
    }
    return mapping.get(severity, "info")


# ──────────────────────────────────────────────
# Card builders
# ──────────────────────────────────────────────

def _build_differential_card(
    diagnoses: List[DiagnosisCandidate],
    patient_summary: str,
) -> Optional[Dict[str, Any]]:
    """Build a CDS Hooks card for differential diagnosis."""
    if not diagnoses:
        return None

    # Summary: top diagnosis
    top_dx = diagnoses[0]
    summary = f"Differential Diagnosis: {top_dx.diagnosis} ({top_dx.likelihood.value})"
    if len(summary) > 140:
        summary = summary[:137] + "..."

    # Detail: full differential in markdown
    detail_lines = [f"**Patient Summary:** {patient_summary}\n"] if patient_summary else []
    detail_lines.append("### Differential Diagnosis\n")

    for i, dx in enumerate(diagnoses[:5], 1):
        likelihood_display = dx.likelihood.value
        if dx.likelihood_raw:
            likelihood_display = f'{dx.likelihood.value} (model: "{dx.likelihood_raw}")'

        detail_lines.append(f"**{i}. {dx.diagnosis}** — Likelihood: {likelihood_display}")
        if dx.icd10_code:
            detail_lines.append(f"   ICD-10: {dx.icd10_code}")
        if dx.supporting_evidence:
            evidence = ", ".join(dx.supporting_evidence[:3])
            detail_lines.append(f"   Evidence: {evidence}")
        if dx.reasoning:
            detail_lines.append(f"   Reasoning: {dx.reasoning[:200]}")
        detail_lines.append("")

    return {
        "uuid": str(uuid.uuid4()),
        "summary": summary,
        "detail": "\n".join(detail_lines),
        "indicator": "info",
        "source": {
            "label": "CDS Agent — Clinical Reasoning",
            "topic": {
                "code": "differential-diagnosis",
                "system": "https://cds-agent.example.org/topics",
            },
        },
    }


def _build_drug_interaction_cards(
    interactions: List[DrugInteraction],
) -> List[Dict[str, Any]]:
    """Build CDS Hooks cards for drug interactions."""
    cards = []

    for interaction in interactions:
        severity = interaction.severity
        indicator = _severity_to_indicator(severity)

        # Only generate cards for moderate+ severity
        if severity in (Severity.LOW,):
            continue

        summary = f"Drug Interaction: {interaction.drug_a} ↔ {interaction.drug_b} ({severity.value})"
        if len(summary) > 140:
            summary = summary[:137] + "..."

        detail_lines = [
            f"### Drug Interaction Alert\n",
            f"**{interaction.drug_a}** ↔ **{interaction.drug_b}**\n",
            f"**Severity:** {severity.value.upper()}\n",
            f"**Description:** {interaction.description}\n",
        ]
        if interaction.clinical_significance:
            detail_lines.append(f"**Clinical Significance:** {interaction.clinical_significance}\n")
        detail_lines.append(f"*Source: {interaction.source}*")

        cards.append({
            "uuid": str(uuid.uuid4()),
            "summary": summary,
            "detail": "\n".join(detail_lines),
            "indicator": indicator,
            "source": {
                "label": f"CDS Agent — Drug Check ({interaction.source})",
                "topic": {
                    "code": "drug-interaction",
                    "system": "https://cds-agent.example.org/topics",
                },
            },
        })

    return cards


def _build_conflict_cards(
    conflicts: List[ClinicalConflict],
) -> List[Dict[str, Any]]:
    """Build CDS Hooks cards for clinical conflicts."""
    cards = []

    for conflict in conflicts:
        indicator = _severity_to_indicator(conflict.severity)

        summary = f"{conflict.conflict_type.value.replace('_', ' ').title()}: {conflict.description[:100]}"
        if len(summary) > 140:
            summary = summary[:137] + "..."

        detail_lines = [
            f"### Clinical Conflict Detected\n",
            f"**Type:** {conflict.conflict_type.value.replace('_', ' ').title()}\n",
            f"**Severity:** {conflict.severity.value.upper()}\n",
            f"**Description:** {conflict.description}\n",
        ]
        if conflict.guideline_source:
            detail_lines.append(f"**Guideline:** {conflict.guideline_source}")
        if conflict.guideline_text:
            detail_lines.append(f"**Guideline Recommends:** {conflict.guideline_text}")
        if conflict.patient_data:
            detail_lines.append(f"**Patient Data:** {conflict.patient_data}")
        if conflict.suggested_resolution:
            detail_lines.append(f"\n**Suggested Resolution:** {conflict.suggested_resolution}")

        card = {
            "uuid": str(uuid.uuid4()),
            "summary": summary,
            "detail": "\n".join(detail_lines),
            "indicator": indicator,
            "source": {
                "label": "CDS Agent — Conflict Detection",
                "topic": {
                    "code": "guideline-conflict",
                    "system": "https://cds-agent.example.org/topics",
                },
            },
        }

        # Add suggestion if resolution is available
        if conflict.suggested_resolution:
            card["suggestions"] = [{
                "label": conflict.suggested_resolution[:80],
                "uuid": str(uuid.uuid4()),
            }]

        cards.append(card)

    return cards


def _build_recommendations_card(
    recommendations: List[str],
    sources: List[str],
) -> Optional[Dict[str, Any]]:
    """Build a CDS Hooks card for guideline recommendations."""
    if not recommendations:
        return None

    summary = f"Clinical Guidelines: {len(recommendations)} recommendation(s)"

    detail_lines = ["### Guideline-Based Recommendations\n"]
    for i, rec in enumerate(recommendations[:10], 1):
        detail_lines.append(f"{i}. {rec}")

    if sources:
        detail_lines.append("\n**Sources:**")
        for source in sources[:5]:
            detail_lines.append(f"- {source}")

    return {
        "uuid": str(uuid.uuid4()),
        "summary": summary,
        "detail": "\n".join(detail_lines),
        "indicator": "info",
        "source": {
            "label": "CDS Agent — Guideline Retrieval",
            "topic": {
                "code": "guideline-recommendation",
                "system": "https://cds-agent.example.org/topics",
            },
        },
    }


def _build_next_steps_card(
    next_steps: List[RecommendedAction],
) -> Optional[Dict[str, Any]]:
    """Build a CDS Hooks card for suggested next steps."""
    if not next_steps:
        return None

    # Find highest priority
    priorities = [s.priority for s in next_steps]
    highest = max(priorities, key=lambda p: list(Severity).index(p))
    indicator = _severity_to_indicator(highest)

    summary = f"Suggested Actions: {len(next_steps)} next step(s)"

    detail_lines = ["### Suggested Next Steps\n"]
    for i, step in enumerate(next_steps[:10], 1):
        priority_badge = step.priority.value.upper()
        detail_lines.append(f"**{i}. [{priority_badge}] {step.action}**")
        if step.rationale:
            detail_lines.append(f"   Rationale: {step.rationale}")
        detail_lines.append("")

    # Generate FHIR-style suggestions for EHR integration
    suggestions = []
    for step in next_steps[:5]:
        suggestions.append({
            "label": step.action[:80],
            "uuid": str(uuid.uuid4()),
        })

    card = {
        "uuid": str(uuid.uuid4()),
        "summary": summary,
        "detail": "\n".join(detail_lines),
        "indicator": indicator,
        "source": {
            "label": "CDS Agent — Clinical Synthesis",
            "topic": {
                "code": "suggested-action",
                "system": "https://cds-agent.example.org/topics",
            },
        },
    }

    if suggestions:
        card["suggestions"] = suggestions

    return card


def _build_caveats_card(
    caveats: List[str],
) -> Optional[Dict[str, Any]]:
    """Build a CDS Hooks card for limitations and caveats."""
    if not caveats:
        return None

    summary = f"AI Limitations: {len(caveats)} caveat(s) — review before acting"

    detail_lines = [
        "### Caveats & Limitations\n",
        "⚠️ **This is an AI-generated clinical decision support report. "
        "All recommendations require independent clinical validation.**\n",
    ]
    for i, caveat in enumerate(caveats[:10], 1):
        detail_lines.append(f"{i}. {caveat}")

    return {
        "uuid": str(uuid.uuid4()),
        "summary": summary,
        "detail": "\n".join(detail_lines),
        "indicator": "info",
        "source": {
            "label": "CDS Agent — Disclaimers",
            "topic": {
                "code": "caveat",
                "system": "https://cds-agent.example.org/topics",
            },
        },
    }


# ──────────────────────────────────────────────
# Main conversion function
# ──────────────────────────────────────────────

def cds_report_to_hooks_response(
    report: CDSReport,
    service_id: str = "cds-agent",
) -> Dict[str, Any]:
    """
    Convert a CDSReport into a CDS Hooks 2.0 response.

    Args:
        report: The completed CDSReport from the synthesis step
        service_id: CDS Hooks service identifier

    Returns:
        Dict matching the CDS Hooks response schema:
        {
            "cards": [...],
            "systemActions": [],
            "extension": { ... metadata ... }
        }
    """
    cards = []

    # 1. Differential diagnosis card
    dx_card = _build_differential_card(
        report.differential_diagnosis,
        report.patient_summary,
    )
    if dx_card:
        cards.append(dx_card)

    # 2. Drug interaction cards (one per interaction)
    interaction_cards = _build_drug_interaction_cards(
        report.drug_interaction_warnings,
    )
    cards.extend(interaction_cards)

    # 3. Conflict alert cards (highest priority)
    conflict_cards = _build_conflict_cards(report.conflicts)
    cards.extend(conflict_cards)

    # 4. Recommendations card
    rec_card = _build_recommendations_card(
        report.guideline_recommendations,
        report.sources_cited,
    )
    if rec_card:
        cards.append(rec_card)

    # 5. Next steps card
    steps_card = _build_next_steps_card(report.suggested_next_steps)
    if steps_card:
        cards.append(steps_card)

    # 6. Caveats card (always last)
    caveats_card = _build_caveats_card(report.caveats)
    if caveats_card:
        cards.append(caveats_card)

    # Sort cards: critical → warning → info
    indicator_order = {"critical": 0, "warning": 1, "info": 2}
    cards.sort(key=lambda c: indicator_order.get(c.get("indicator", "info"), 2))

    response = {
        "cards": cards,
        "systemActions": [],
        "extension": {
            "com.cds-agent.metadata": {
                "service_id": service_id,
                "generated_at": report.generated_at.isoformat() if report.generated_at else datetime.now(timezone.utc).isoformat(),
                "total_diagnoses": len(report.differential_diagnosis),
                "total_interactions": len(report.drug_interaction_warnings),
                "total_conflicts": len(report.conflicts),
                "total_recommendations": len(report.guideline_recommendations),
                "total_next_steps": len(report.suggested_next_steps),
                "total_caveats": len(report.caveats),
            },
        },
    }

    return response


# ──────────────────────────────────────────────
# CDS Hooks service discovery
# ──────────────────────────────────────────────

def get_cds_hooks_discovery() -> Dict[str, Any]:
    """
    Return the CDS Hooks discovery response for this service.

    This would be served at the /cds-services endpoint.
    """
    return {
        "services": [
            {
                "hook": "patient-view",
                "title": "Clinical Decision Support Agent",
                "description": (
                    "AI-powered clinical decision support using MedGemma. "
                    "Provides differential diagnosis, drug interaction checks, "
                    "guideline-based recommendations, and conflict detection."
                ),
                "id": "cds-agent",
                "prefetch": {
                    "patient": "Patient/{{context.patientId}}",
                    "conditions": "Condition?patient={{context.patientId}}&clinical-status=active",
                    "medications": "MedicationStatement?patient={{context.patientId}}&status=active",
                    "allergies": "AllergyIntolerance?patient={{context.patientId}}&clinical-status=active",
                    "observations": "Observation?patient={{context.patientId}}&category=laboratory&_sort=-date&_count=20",
                    "vitals": "Observation?patient={{context.patientId}}&category=vital-signs&_sort=-date&_count=10",
                },
            },
        ],
    }
