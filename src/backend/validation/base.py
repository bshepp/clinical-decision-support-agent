"""
Base classes and utilities for the validation framework.

Provides:
  - ValidationCase: a single test case with input + ground truth
  - ValidationResult: scored result for a single case
  - ValidationSummary: aggregate metrics for a dataset
  - run_cds_pipeline(): runs a case through the orchestrator directly
  - fuzzy_match(): soft string matching for diagnosis comparison
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── CDS pipeline imports ──
import sys

# Ensure the backend app is importable
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.agent.orchestrator import Orchestrator
from app.models.schemas import CaseSubmission, CDSReport, AgentState, AgentStepStatus


# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────

@dataclass
class ValidationCase:
    """A single validation test case."""
    case_id: str
    source_dataset: str                    # "medqa", "mtsamples", "pmc"
    input_text: str                        # Clinical text fed to the pipeline
    ground_truth: Dict[str, Any]           # Dataset-specific ground truth
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of running one case through the pipeline + scoring."""
    case_id: str
    source_dataset: str
    success: bool                          # Pipeline completed without crash
    scores: Dict[str, float]              # Metric name → score (0.0–1.0)
    pipeline_time_ms: int = 0
    step_results: Dict[str, str] = field(default_factory=dict)  # step_id → status
    report_summary: Optional[str] = None
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)       # Extra scoring info


@dataclass
class ValidationSummary:
    """Aggregate metrics for a dataset validation run."""
    dataset: str
    total_cases: int
    successful_cases: int
    failed_cases: int
    metrics: Dict[str, float]              # Metric name → average score
    per_case: List[ValidationResult]
    run_duration_sec: float
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


# ──────────────────────────────────────────────
# Pipeline runner
# ──────────────────────────────────────────────

async def run_cds_pipeline(
    patient_text: str,
    include_drug_check: bool = True,
    include_guidelines: bool = True,
    timeout_sec: int = 180,
) -> tuple[Optional[AgentState], Optional[CDSReport], Optional[str]]:
    """
    Run a single case through the CDS pipeline directly (no HTTP server needed).

    Returns:
        (state, report, error) — error is None on success
    """
    case = CaseSubmission(
        patient_text=patient_text,
        include_drug_check=include_drug_check,
        include_guidelines=include_guidelines,
    )
    orchestrator = Orchestrator()

    try:
        async for _step_update in orchestrator.run(case):
            pass  # consume all step updates

        report = orchestrator.get_result()

        # If no report was produced, collect errors from failed steps
        if report is None and orchestrator.state:
            failed_steps = [
                s for s in orchestrator.state.steps
                if s.status == AgentStepStatus.FAILED
            ]
            if failed_steps:
                error_msgs = [f"{s.step_id}: {s.error}" for s in failed_steps]
                return orchestrator.state, None, "; ".join(error_msgs)
            return orchestrator.state, None, "Pipeline completed but produced no report"

        return orchestrator.state, report, None
    except asyncio.TimeoutError:
        return orchestrator.state, None, f"Pipeline timed out after {timeout_sec}s"
    except Exception as e:
        return orchestrator.state, None, str(e)


# ──────────────────────────────────────────────
# Fuzzy string matching for diagnosis comparison
# ──────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, normalize whitespace."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def fuzzy_match(candidate: str, target: str, threshold: float = 0.6) -> bool:
    """
    Check if candidate text is a fuzzy match for target.

    Uses token overlap (Jaccard-like) rather than edit distance —
    medical terms are long and we care about semantic overlap, not typos.

    Args:
        candidate: Text from the pipeline output
        target: Ground truth text
        threshold: Minimum token overlap ratio (0.0–1.0)
    """
    c_tokens = set(normalize_text(candidate).split())
    t_tokens = set(normalize_text(target).split())

    if not t_tokens:
        return False

    # If target is a substring of candidate (or vice versa), that's a match
    if normalize_text(target) in normalize_text(candidate):
        return True
    if normalize_text(candidate) in normalize_text(target):
        return True

    # Token overlap
    overlap = len(c_tokens & t_tokens)
    denominator = min(len(c_tokens), len(t_tokens))
    if denominator == 0:
        return False

    return (overlap / denominator) >= threshold


def diagnosis_in_differential(
    target_diagnosis: str,
    report: CDSReport,
    top_n: Optional[int] = None,
) -> tuple[bool, int, str]:
    """
    Check if target_diagnosis appears in the report's differential.

    Returns:
        (found, rank, match_location) — rank is 0-indexed position, or -1 if not found.
        match_location is one of: "differential", "next_steps", "recommendations",
        "fulltext", or "not_found".
    """
    diagnoses = report.differential_diagnosis
    if top_n:
        diagnoses = diagnoses[:top_n]

    for i, dx in enumerate(diagnoses):
        if fuzzy_match(dx.diagnosis, target_diagnosis):
            return True, i, "differential"

    # Check suggested_next_steps (for management-type answers)
    for i, action in enumerate(report.suggested_next_steps):
        if fuzzy_match(action.action, target_diagnosis):
            return True, len(diagnoses) + i, "next_steps"

    # Check guideline recommendations (for treatment-type answers)
    for i, rec in enumerate(report.guideline_recommendations):
        if fuzzy_match(rec, target_diagnosis):
            return True, len(diagnoses) + len(report.suggested_next_steps) + i, "recommendations"

    # Broad fulltext check (patient_summary, recommendations, next steps combined)
    full_text = " ".join([
        report.patient_summary or "",
        " ".join(report.guideline_recommendations),
        " ".join(a.action for a in report.suggested_next_steps),
        " ".join(dx.reasoning for dx in report.differential_diagnosis),
    ])
    if fuzzy_match(full_text, target_diagnosis, threshold=0.3):
        return True, len(diagnoses), "fulltext"

    return False, -1, "not_found"


# ──────────────────────────────────────────────
# I/O utilities
# ──────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def ensure_data_dir():
    """Create the data directory if it doesn't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _result_to_dict(r: ValidationResult) -> dict:
    """Convert a ValidationResult to a serialisable dict."""
    return {
        "case_id": r.case_id,
        "source_dataset": r.source_dataset,
        "success": r.success,
        "scores": r.scores,
        "pipeline_time_ms": r.pipeline_time_ms,
        "step_results": r.step_results,
        "report_summary": r.report_summary,
        "error": r.error,
        "details": r.details,
    }


# ──────────────────────────────────────────────
# Incremental checkpoint (JSONL)
# ──────────────────────────────────────────────

def checkpoint_path(dataset: str) -> Path:
    """Return the path to the checkpoint JSONL for *dataset*."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"{dataset}_checkpoint.jsonl"


def save_incremental(result: ValidationResult, dataset: str) -> None:
    """Append a single case result to the checkpoint JSONL file."""
    path = checkpoint_path(dataset)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_result_to_dict(result), default=str) + "\n")


def load_checkpoint(dataset: str) -> List[ValidationResult]:
    """
    Load previously-completed results from the checkpoint file.

    Returns a list of ValidationResult objects (may be empty).
    """
    path = checkpoint_path(dataset)
    if not path.exists():
        return []

    results: List[ValidationResult] = []
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        if not line.strip():
            continue
        d = json.loads(line)
        results.append(ValidationResult(
            case_id=d["case_id"],
            source_dataset=d.get("source_dataset", dataset),
            success=d["success"],
            scores=d["scores"],
            pipeline_time_ms=d.get("pipeline_time_ms", 0),
            step_results=d.get("step_results", {}),
            report_summary=d.get("report_summary"),
            error=d.get("error"),
            details=d.get("details", {}),
        ))
    return results


def clear_checkpoint(dataset: str) -> None:
    """Delete checkpoint file for a fresh run."""
    path = checkpoint_path(dataset)
    if path.exists():
        path.unlink()


# ──────────────────────────────────────────────
# Final results save
# ──────────────────────────────────────────────

def save_results(summary: ValidationSummary, filename: Optional[str] = None):
    """Save validation results to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if filename is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{summary.dataset}_{ts}.json"

    path = RESULTS_DIR / filename

    # Convert to serializable dict
    data = {
        "dataset": summary.dataset,
        "total_cases": summary.total_cases,
        "successful_cases": summary.successful_cases,
        "failed_cases": summary.failed_cases,
        "metrics": summary.metrics,
        "run_duration_sec": summary.run_duration_sec,
        "timestamp": summary.timestamp,
        "per_case": [_result_to_dict(r) for r in summary.per_case],
    }

    path.write_text(json.dumps(data, indent=2, default=str))
    return path


def print_summary(summary: ValidationSummary):
    """Pretty-print validation results to console."""
    print(f"\n{'='*60}")
    print(f"  Validation Results: {summary.dataset.upper()}")
    print(f"{'='*60}")
    print(f"  Total cases:      {summary.total_cases}")
    print(f"  Successful:       {summary.successful_cases}")
    print(f"  Failed:           {summary.failed_cases}")
    print(f"  Duration:         {summary.run_duration_sec:.1f}s")
    print(f"\n  Metrics:")
    for metric, value in sorted(summary.metrics.items()):
        if "time" in metric and isinstance(value, (int, float)):
            print(f"    {metric:30s} {value:.0f}ms")
        elif isinstance(value, float):
            print(f"    {metric:30s} {value:.1%}")
        else:
            print(f"    {metric:30s} {value}")
    print(f"{'='*60}\n")
