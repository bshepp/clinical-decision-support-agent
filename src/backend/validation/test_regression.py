"""
Regression test suite for the CDS pipeline.

Captures every case that has previously caused a pipeline failure or
produced incorrect/dangerous output. Each case includes:
  - The exact input that triggered the issue
  - The bug that was found
  - What the expected behavior should be
  - Assertions to verify the fix holds

Run this suite after any change to schemas, tools, or orchestrator to
ensure previously-fixed bugs don't regress.

Usage:
    python -m validation.test_regression
    python -m pytest validation/test_regression.py -v
"""
from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Ensure imports
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from validation.base import (
    ValidationResult,
    ValidationSummary,
    run_cds_pipeline,
    save_results,
    print_summary,
)
from app.models.schemas import (
    Confidence,
    DiagnosisCandidate,
    PatientProfile,
    Severity,
)


# ──────────────────────────────────────────────
# Regression case definitions
# ──────────────────────────────────────────────

@dataclass
class RegressionCase:
    """A single regression test case."""
    case_id: str
    title: str                                    # Short description
    bug_description: str                          # What went wrong originally
    input_text: str                               # The triggering input
    assertions: List[Callable]                    # List of assertion functions
    category: str = "pipeline"                    # Category tag
    fixed_in: str = ""                            # Git ref or date when fixed
    requires_pipeline: bool = True                # Whether to run full pipeline


@dataclass
class RegressionResult:
    """Result of running one regression test."""
    case_id: str
    title: str
    passed: bool
    assertions_passed: int
    assertions_total: int
    failures: List[str]
    elapsed_ms: int = 0
    error: Optional[str] = None


# ──────────────────────────────────────────────
# Schema-level regression tests (no pipeline needed)
# ──────────────────────────────────────────────

def _regression_very_low_likelihood() -> RegressionCase:
    """
    BUG: MedGemma returned "very low" as likelihood for a DiagnosisCandidate,
    but the Confidence enum only had low/moderate/high. This caused a
    Pydantic validation error and crashed the pipeline.

    FIX: Expanded Confidence to 5-point Likert scale + fuzzy mapping.
    """
    def assert_very_low_accepted(report, state, error):
        dx = DiagnosisCandidate(
            diagnosis="Test Condition",
            likelihood="very low",
            supporting_evidence=["test"],
        )
        assert dx.likelihood == Confidence.VERY_LOW, \
            f"Expected VERY_LOW, got {dx.likelihood}"
        assert dx.likelihood_raw is None, \
            f"Expected no raw value, got {dx.likelihood_raw}"

    def assert_very_high_accepted(report, state, error):
        dx = DiagnosisCandidate(
            diagnosis="Test Condition",
            likelihood="very high",
            supporting_evidence=["test"],
        )
        assert dx.likelihood == Confidence.VERY_HIGH, \
            f"Expected VERY_HIGH, got {dx.likelihood}"

    def assert_fuzzy_mapping_works(report, state, error):
        test_cases = {
            "unlikely": Confidence.VERY_LOW,
            "rare": Confidence.VERY_LOW,
            "possible": Confidence.LOW,
            "medium": Confidence.MODERATE,
            "probable": Confidence.HIGH,
            "highly likely": Confidence.VERY_HIGH,
            "almost certain": Confidence.VERY_HIGH,
        }
        for raw_val, expected in test_cases.items():
            dx = DiagnosisCandidate(
                diagnosis="Test",
                likelihood=raw_val,
                supporting_evidence=[],
            )
            assert dx.likelihood == expected, \
                f"Fuzzy mapping failed: '{raw_val}' → {dx.likelihood}, expected {expected}"

    def assert_unrecognized_captured(report, state, error):
        dx = DiagnosisCandidate(
            diagnosis="Test",
            likelihood="super duper likely",
            supporting_evidence=[],
        )
        assert dx.likelihood == Confidence.UNRECOGNIZED, \
            f"Expected UNRECOGNIZED, got {dx.likelihood}"
        assert dx.likelihood_raw == "super duper likely", \
            f"Expected raw value preserved, got {dx.likelihood_raw}"

    return RegressionCase(
        case_id="reg_schema_001",
        title="Confidence enum accepts 'very low' likelihood",
        bug_description=(
            "MedGemma returned 'very low' as likelihood. The Confidence enum "
            "only had low/moderate/high, causing a Pydantic ValidationError."
        ),
        input_text="",  # No pipeline run needed
        assertions=[
            assert_very_low_accepted,
            assert_very_high_accepted,
            assert_fuzzy_mapping_works,
            assert_unrecognized_captured,
        ],
        category="schema",
        fixed_in="2024-01 Confidence enum expansion",
        requires_pipeline=False,
    )


def _regression_severity_case_insensitive() -> RegressionCase:
    """
    BUG: Pipeline could fail if severity came back as "High" instead of "high".
    FIX: field_validator normalizes to lowercase.
    """
    def assert_severity_case_insensitive(report, state, error):
        from app.models.schemas import ClinicalConflict, ConflictType
        # Test that string→enum normalization works via model_validate
        conflict = ClinicalConflict.model_validate({
            "conflict_type": "OMISSION",
            "severity": "High",
            "description": "Test conflict",
            "guideline_source": "test",
            "guideline_text": "test",
            "patient_data": "test",
            "suggested_resolution": None,
        })
        assert conflict.severity == Severity.HIGH, \
            f"Expected HIGH, got {conflict.severity}"
        assert conflict.conflict_type == ConflictType.OMISSION, \
            f"Expected OMISSION, got {conflict.conflict_type}"

    return RegressionCase(
        case_id="reg_schema_002",
        title="Severity and ConflictType are case-insensitive",
        bug_description="Enum values from LLM could be mixed-case, causing validation errors.",
        input_text="",
        assertions=[assert_severity_case_insensitive],
        category="schema",
        requires_pipeline=False,
    )


def _regression_empty_medications() -> RegressionCase:
    """
    BUG: Pipeline should handle a patient with no medications without crashing
    the drug interaction checker.
    """
    def assert_no_crash(report, state, error):
        assert error is None, f"Pipeline crashed: {error}"
        assert report is not None, "Pipeline produced no report"

    def assert_no_false_interactions(report, state, error):
        if report:
            # With no medications, there should be no drug interactions
            # (or at most warnings about no medications to check)
            assert len(report.drug_interaction_warnings) == 0, \
                f"Found {len(report.drug_interaction_warnings)} false interactions with no medications"

    return RegressionCase(
        case_id="reg_pipeline_001",
        title="Empty medication list does not crash drug checker",
        bug_description="Drug interaction tool could fail with empty medication list.",
        input_text="""
        42-year-old female presents with headache for 3 days.
        No current medications. No allergies. No past medical history.
        Vitals: BP 128/82, HR 76, Temp 98.6°F.
        """,
        assertions=[assert_no_crash, assert_no_false_interactions],
        category="pipeline",
        requires_pipeline=True,
    )


def _regression_long_input() -> RegressionCase:
    """
    BUG: Very long inputs could exceed token limits and crash.
    FIX: Patient parser truncates input gracefully.
    """
    def assert_no_crash(report, state, error):
        # May not produce a perfect report but should not crash
        assert error is None or "timeout" in (error or "").lower(), \
            f"Pipeline crashed with long input: {error}"

    return RegressionCase(
        case_id="reg_pipeline_002",
        title="Long input (3000+ chars) does not crash pipeline",
        bug_description="Very long patient descriptions could exceed model token limits.",
        input_text=(
            "65-year-old male with extensive medical history. " * 100 +
            "Chief complaint: chest pain. " +
            "PMH: HTN, DM2, CAD, CHF, COPD, CKD stage 3, gout, " +
            "hypothyroidism, BPH, depression, anxiety, GERD, OSA. " +
            "Medications: metformin, lisinopril, metoprolol, atorvastatin, "
            "aspirin, furosemide, allopurinol, levothyroxine, tamsulosin, "
            "sertraline, omeprazole. " +
            "Allergies: PCN, sulfa. " +
            "Social: former smoker 30 pack-years, quit 5 years ago. " +
            "Family history: father MI at 55, mother stroke at 70. " * 20
        ),
        assertions=[assert_no_crash],
        category="pipeline",
        requires_pipeline=True,
    )


def _regression_special_characters() -> RegressionCase:
    """
    BUG: Special characters in patient text could break JSON parsing.
    """
    def assert_no_crash(report, state, error):
        assert error is None, f"Pipeline crashed with special chars: {error}"

    def assert_has_report(report, state, error):
        assert report is not None, "No report produced for special character input"

    return RegressionCase(
        case_id="reg_pipeline_003",
        title="Special characters in input don't break JSON parsing",
        bug_description="Quotes, newlines, and special chars in patient text could break structured output parsing.",
        input_text="""
        50-year-old male, "John Doe" (patient's name redacted).
        CC: chest pain — described as "crushing" and "like an elephant sitting on my chest."
        Labs: Troponin I: 0.5 ng/mL (↑), BNP: 1200 pg/mL (↑↑).
        Meds: Aspirin 81mg/day, Metoprolol 25mg b.i.d.
        Notes: Patient states he's "never felt this bad before." Dr. Smith's note:
        "Consider ACS protocol — r/o NSTEMI vs. unstable angina."
        """,
        assertions=[assert_no_crash, assert_has_report],
        category="pipeline",
        requires_pipeline=True,
    )


def _regression_patient_profile_defaults() -> RegressionCase:
    """
    BUG: PatientProfile should handle missing optional fields via defaults.
    """
    def assert_minimal_profile_valid(report, state, error):
        profile = PatientProfile(
            chief_complaint="headache",
            history_of_present_illness="",
        )
        assert profile.age is None
        assert profile.gender.value == "unknown"
        assert profile.current_medications == []
        assert profile.allergies == []
        assert profile.vital_signs is None

    return RegressionCase(
        case_id="reg_schema_003",
        title="PatientProfile defaults work for minimal input",
        bug_description="Missing optional fields in PatientProfile could cause KeyError.",
        input_text="",
        assertions=[assert_minimal_profile_valid],
        category="schema",
        requires_pipeline=False,
    )


# ──────────────────────────────────────────────
# Registry of all regression cases
# ──────────────────────────────────────────────

def get_all_regression_cases() -> List[RegressionCase]:
    """Return all registered regression test cases."""
    return [
        _regression_very_low_likelihood(),
        _regression_severity_case_insensitive(),
        _regression_empty_medications(),
        _regression_long_input(),
        _regression_special_characters(),
        _regression_patient_profile_defaults(),
    ]


# ──────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────

async def run_regression_suite(
    include_pipeline_tests: bool = True,
    include_drug_check: bool = True,
    include_guidelines: bool = True,
    delay_between_cases: float = 2.0,
) -> tuple[List[RegressionResult], ValidationSummary]:
    """
    Run all regression tests and produce results.

    Args:
        include_pipeline_tests: Whether to run tests that require the full pipeline
        include_drug_check: Whether to include drug interaction check in pipeline tests
        include_guidelines: Whether to include guideline retrieval in pipeline tests
        delay_between_cases: Seconds to wait between pipeline tests
    """
    cases = get_all_regression_cases()
    results: List[RegressionResult] = []
    validation_results: List[ValidationResult] = []
    start_time = time.time()

    for i, case in enumerate(cases):
        if case.requires_pipeline and not include_pipeline_tests:
            print(f"\n  [{i+1}/{len(cases)}] {case.case_id}: SKIPPED (pipeline test)")
            continue

        print(f"\n  [{i+1}/{len(cases)}] {case.case_id}: {case.title}")
        case_start = time.monotonic()

        report = None
        state = None
        error = None

        # Run pipeline if needed
        if case.requires_pipeline:
            try:
                state, report, error = await run_cds_pipeline(
                    patient_text=case.input_text,
                    include_drug_check=include_drug_check,
                    include_guidelines=include_guidelines,
                )
            except Exception as e:
                error = str(e)

        # Run assertions
        passed_count = 0
        failures = []
        for j, assertion_fn in enumerate(case.assertions):
            try:
                assertion_fn(report, state, error)
                passed_count += 1
            except AssertionError as e:
                failures.append(f"Assertion {j+1}: {str(e)}")
            except Exception as e:
                failures.append(f"Assertion {j+1} raised {type(e).__name__}: {str(e)}")

        elapsed_ms = int((time.monotonic() - case_start) * 1000)
        all_passed = len(failures) == 0

        reg_result = RegressionResult(
            case_id=case.case_id,
            title=case.title,
            passed=all_passed,
            assertions_passed=passed_count,
            assertions_total=len(case.assertions),
            failures=failures,
            elapsed_ms=elapsed_ms,
            error=error,
        )
        results.append(reg_result)

        # Convert to ValidationResult for summary compatibility
        scores = {
            "pass": 1.0 if all_passed else 0.0,
            "assertion_pass_rate": passed_count / len(case.assertions) if case.assertions else 1.0,
        }
        val_result = ValidationResult(
            case_id=case.case_id,
            source_dataset="regression",
            success=all_passed,
            scores=scores,
            pipeline_time_ms=elapsed_ms,
            error="; ".join(failures) if failures else None,
            details={
                "title": case.title,
                "category": case.category,
                "bug_description": case.bug_description,
                "assertions_passed": passed_count,
                "assertions_total": len(case.assertions),
                "failures": failures,
            },
        )
        validation_results.append(val_result)

        # Console output
        icon = "+" if all_passed else "-"
        status = "PASS" if all_passed else "FAIL"
        print(f"    {icon} [{status}] {passed_count}/{len(case.assertions)} assertions ({elapsed_ms}ms)")
        for f in failures:
            print(f"      FAIL: {f}")

        if case.requires_pipeline and i < len(cases) - 1:
            next_case = cases[i + 1] if i + 1 < len(cases) else None
            if next_case and next_case.requires_pipeline:
                await asyncio.sleep(delay_between_cases)

    # Build summary
    total = len(results)
    passed_total = sum(1 for r in results if r.passed)

    metrics = {
        "pass_rate": passed_total / total if total else 0.0,
        "total_assertions": sum(r.assertions_total for r in results),
        "passed_assertions": sum(r.assertions_passed for r in results),
    }

    # Per-category
    by_cat: dict = {}
    for r, case in zip(results, cases):
        by_cat.setdefault(case.category, []).append(r)
    for cat, cat_results in by_cat.items():
        cat_pass = sum(1 for r in cat_results if r.passed)
        metrics[f"count_{cat}"] = len(cat_results)
        metrics[f"pass_rate_{cat}"] = cat_pass / len(cat_results)

    summary = ValidationSummary(
        dataset="regression",
        total_cases=total,
        successful_cases=passed_total,
        failed_cases=total - passed_total,
        metrics=metrics,
        per_case=validation_results,
        run_duration_sec=time.time() - start_time,
    )

    return results, summary


# ──────────────────────────────────────────────
# Standalone runner
# ──────────────────────────────────────────────

async def main():
    """Run regression suite standalone."""
    import argparse

    parser = argparse.ArgumentParser(description="Regression Test Suite")
    parser.add_argument("--schema-only", action="store_true",
                       help="Only run schema tests (no pipeline)")
    parser.add_argument("--no-drugs", action="store_true",
                       help="Skip drug interaction check")
    parser.add_argument("--delay", type=float, default=2.0,
                       help="Delay between pipeline tests (seconds)")
    args = parser.parse_args()

    print("Regression Test Suite")
    print("=" * 40)

    cases = get_all_regression_cases()
    schema_count = sum(1 for c in cases if not c.requires_pipeline)
    pipeline_count = sum(1 for c in cases if c.requires_pipeline)
    print(f"  Schema tests: {schema_count}")
    print(f"  Pipeline tests: {pipeline_count}")

    results, summary = await run_regression_suite(
        include_pipeline_tests=not args.schema_only,
        include_drug_check=not args.no_drugs,
        delay_between_cases=args.delay,
    )

    print_summary(summary)

    # Final pass/fail
    all_passed = all(r.passed for r in results)
    print(f"\n{'='*40}")
    if all_passed:
        print(f"  ALL {len(results)} REGRESSION TESTS PASSED")
    else:
        failed = [r for r in results if not r.passed]
        print(f"  {len(failed)}/{len(results)} REGRESSION TESTS FAILED:")
        for r in failed:
            print(f"    - {r.case_id}: {r.title}")
            for f in r.failures:
                print(f"      {f}")
    print(f"{'='*40}")

    path = save_results(summary)
    print(f"\nResults saved to: {path}")

    # Exit code for CI
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
