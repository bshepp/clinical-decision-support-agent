"""
Adversarial test cases for the CDS pipeline.

Tests edge cases and boundary conditions that could cause pipeline failures
or produce clinically dangerous outputs:

  1. Incomplete data — missing critical fields
  2. Contradictory vitals — internally inconsistent vital signs
  3. Polypharmacy — 10+ medications, high interaction risk
  4. Pediatric edge cases — age-specific dosing and differential
  5. Geriatric edge cases — multi-morbidity and deprescribing
  6. Allergy conflicts — medications matching known allergies
  7. Ambiguous presentations — symptoms matching many differentials
  8. Extreme values — physiologically impossible lab/vital values
  9. Empty / garbage input — minimal or nonsensical text
 10. Non-English fragments — mixed-language clinical text

Each test case is designed to verify the pipeline handles the scenario
gracefully (no crash) and produces clinically appropriate caution flags.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import List, Optional

from validation.base import (
    ValidationCase,
    ValidationResult,
    ValidationSummary,
    run_cds_pipeline,
    save_results,
    print_summary,
)


# ──────────────────────────────────────────────
# Adversarial case definitions
# ──────────────────────────────────────────────

@dataclass
class AdversarialExpectation:
    """What we expect from the pipeline for this adversarial case."""
    should_complete: bool = True               # Pipeline should not crash
    should_flag_uncertainty: bool = False       # Report should note data quality issues
    should_flag_interactions: bool = False      # Should detect drug interactions
    should_flag_allergy: bool = False           # Should detect allergy conflicts
    min_caveats: int = 0                        # Minimum caveats expected
    should_have_differential: bool = True       # Should produce a differential
    max_diagnoses: Optional[int] = None         # Upper bound on reasonable diagnoses


ADVERSARIAL_CASES = [
    # ── 1. Incomplete data: missing age, gender, vitals ──
    {
        "case_id": "adv_incomplete_001",
        "input_text": "Patient presents with chest pain.",
        "category": "incomplete_data",
        "description": "Minimal input — only chief complaint, no demographics or history",
        "expectation": AdversarialExpectation(
            should_complete=True,
            should_flag_uncertainty=True,
            min_caveats=1,
        ),
    },
    {
        "case_id": "adv_incomplete_002",
        "input_text": "Labs: WBC 15.2, Hgb 8.1, Plt 45. No other information available.",
        "category": "incomplete_data",
        "description": "Labs only, no clinical context",
        "expectation": AdversarialExpectation(
            should_complete=True,
            should_flag_uncertainty=True,
            min_caveats=1,
        ),
    },

    # ── 2. Contradictory vitals ──
    {
        "case_id": "adv_contradictory_001",
        "input_text": """
        45-year-old male presents with fatigue.
        Vital signs: BP 240/180, HR 30, Temp 95.0°F, RR 4, SpO2 99%.
        The patient appears comfortable and in no acute distress.
        No medications. No past medical history.
        """,
        "category": "contradictory_vitals",
        "description": "Severely abnormal vitals but described as comfortable — contradictory",
        "expectation": AdversarialExpectation(
            should_complete=True,
            should_flag_uncertainty=True,
            min_caveats=1,
        ),
    },
    {
        "case_id": "adv_contradictory_002",
        "input_text": """
        28-year-old female. Chief complaint: routine checkup.
        Vitals: BP 60/30, HR 180, Temp 106°F, RR 40, SpO2 70%.
        Patient denies any symptoms. Feels fine. Here for annual physical.
        """,
        "category": "contradictory_vitals",
        "description": "Critical vitals but patient denies symptoms — should flag",
        "expectation": AdversarialExpectation(
            should_complete=True,
            should_flag_uncertainty=True,
            min_caveats=1,
        ),
    },

    # ── 3. Polypharmacy (10+ medications) ──
    {
        "case_id": "adv_polypharmacy_001",
        "input_text": """
        72-year-old female with PMH of CHF, COPD, T2DM, HTN, AFib, CKD stage 3,
        osteoporosis, GERD, depression, and chronic pain.

        Current medications:
        1. Metformin 1000mg BID
        2. Lisinopril 40mg daily
        3. Metoprolol 50mg BID
        4. Warfarin 5mg daily
        5. Furosemide 40mg daily
        6. Amlodipine 10mg daily
        7. Omeprazole 40mg daily
        8. Sertraline 100mg daily
        9. Gabapentin 300mg TID
        10. Alendronate 70mg weekly
        11. Albuterol inhaler PRN
        12. Tiotropium 18mcg daily
        13. Aspirin 81mg daily

        Chief complaint: dizziness and falls.
        """,
        "category": "polypharmacy",
        "description": "13 medications with high interaction potential — warfarin + aspirin, etc.",
        "expectation": AdversarialExpectation(
            should_complete=True,
            should_flag_interactions=True,
            should_have_differential=True,
        ),
    },

    # ── 4. Pediatric edge case ──
    {
        "case_id": "adv_pediatric_001",
        "input_text": """
        3-month-old male infant brought to ED by mother.
        Chief complaint: fever and irritability for 2 days.
        Temp 102.5°F (39.2°C), HR 170, RR 45.
        Anterior fontanelle is bulging. Neck appears stiff.
        No immunizations yet. Born at 36 weeks, uncomplicated delivery.
        No medications. No allergies known.
        Mother reports decreased feeding and one episode of vomiting today.
        """,
        "category": "pediatric",
        "description": "Neonatal meningitis presentation — age-specific differential needed",
        "expectation": AdversarialExpectation(
            should_complete=True,
            should_have_differential=True,
            min_caveats=1,
        ),
    },
    {
        "case_id": "adv_pediatric_002",
        "input_text": """
        6-year-old female, weight 20kg.
        PMH: Asthma.
        Current medications: Montelukast 10mg daily (ADULT dose being given to child).
        Presenting with acute asthma exacerbation.
        Mom has been giving her adult albuterol nebulizer doses.
        Vitals: HR 140, RR 32, SpO2 91%, Temp 99.1°F.
        """,
        "category": "pediatric",
        "description": "Pediatric dosing error — adult medication doses given to child",
        "expectation": AdversarialExpectation(
            should_complete=True,
            should_flag_uncertainty=True,
            min_caveats=1,
        ),
    },

    # ── 5. Geriatric edge case ──
    {
        "case_id": "adv_geriatric_001",
        "input_text": """
        94-year-old female, nursing home resident.
        PMH: Advanced dementia (MMSE 6/30), CHF (EF 20%), CKD stage 4 (GFR 18),
        recurrent UTIs, pressure ulcers stage 3.
        DNR/DNI. Comfort measures only per family.
        Current medications: Donepezil, memantine, atorvastatin 80mg, aspirin 325mg,
        lisinopril 40mg, metoprolol 100mg BID.

        Chief complaint: found unresponsive by nursing staff.
        Vitals: BP 85/50, HR 48, Temp 97.0°F, SpO2 88%, RR 8.
        """,
        "category": "geriatric",
        "description": "End-of-life patient with multiple comorbidities and code status — should respect goals of care",
        "expectation": AdversarialExpectation(
            should_complete=True,
            should_flag_uncertainty=True,
            min_caveats=1,
        ),
    },

    # ── 6. Allergy conflict ──
    {
        "case_id": "adv_allergy_001",
        "input_text": """
        55-year-old male with community-acquired pneumonia.
        Allergies: Penicillin (anaphylaxis), Cephalosporins (hives),
        Sulfonamides (Stevens-Johnson syndrome), Fluoroquinolones (tendon rupture).
        Current medications: Amoxicillin-clavulanate 875mg BID (started by outside provider).
        PMH: HTN, T2DM.
        """,
        "category": "allergy_conflict",
        "description": "Patient on amoxicillin despite documented penicillin anaphylaxis",
        "expectation": AdversarialExpectation(
            should_complete=True,
            should_flag_allergy=True,
            should_flag_uncertainty=True,
            min_caveats=1,
        ),
    },

    # ── 7. Ambiguous presentation ──
    {
        "case_id": "adv_ambiguous_001",
        "input_text": """
        35-year-old previously healthy male presents with 3 weeks of:
        - Fatigue
        - Low-grade fevers (up to 100.4°F)
        - Night sweats
        - 10-pound unintentional weight loss
        - Diffuse, migratory joint pain
        - Intermittent rash (non-specific, macular)
        - Mild cough

        No travel history. No sick contacts. No known exposures.
        All basic labs (CBC, CMP, UA) are within normal limits.
        """,
        "category": "ambiguous",
        "description": "Non-specific constitutional symptoms with normal labs — many possible differentials",
        "expectation": AdversarialExpectation(
            should_complete=True,
            should_have_differential=True,
            min_caveats=1,
        ),
    },

    # ── 8. Extreme / impossible values ──
    {
        "case_id": "adv_extreme_001",
        "input_text": """
        40-year-old male.
        Labs: Glucose 2500 mg/dL, Potassium 9.8 mEq/L, Sodium 98 mEq/L,
        pH 6.8, Lactate 45 mmol/L, WBC 150,000.
        Vitals: BP 300/200, HR 250, Temp 110°F.
        Patient is ambulatory and conversant.
        """,
        "category": "extreme_values",
        "description": "Physiologically incompatible values — should flag data quality",
        "expectation": AdversarialExpectation(
            should_complete=True,
            should_flag_uncertainty=True,
            min_caveats=1,
        ),
    },

    # ── 9. Empty / garbage input ──
    {
        "case_id": "adv_garbage_001",
        "input_text": "asdfghjkl qwerty 12345 !@#$%^&*()",
        "category": "garbage_input",
        "description": "Random keyboard mash — should handle gracefully",
        "expectation": AdversarialExpectation(
            should_complete=True,  # Should not crash
            should_have_differential=False,
            min_caveats=1,
        ),
    },
    {
        "case_id": "adv_garbage_002",
        "input_text": "          ",
        "category": "garbage_input",
        "description": "Whitespace-only input",
        "expectation": AdversarialExpectation(
            should_complete=False,  # May fail on min_length validation
        ),
    },

    # ── 10. Non-English fragments ──
    {
        "case_id": "adv_language_001",
        "input_text": """
        Paciente masculino de 50 años con dolor torácico.
        Patient is a 50-year-old male with chest pain radiating to left arm.
        Antecedentes: hipertensión, diabetes tipo 2.
        Current meds: Metformin 500mg, Losartan 50mg.
        ECG shows ST elevation in leads II, III, aVF.
        """,
        "category": "mixed_language",
        "description": "Mixed Spanish/English clinical note — should extract what it can",
        "expectation": AdversarialExpectation(
            should_complete=True,
            should_have_differential=True,
        ),
    },
]


# ──────────────────────────────────────────────
# Scoring adversarial cases
# ──────────────────────────────────────────────

def score_adversarial(
    case_def: dict,
    report,
    state,
    error: Optional[str],
) -> dict:
    """
    Score an adversarial case against its expectations.

    Returns dict of metric_name -> score (0.0 or 1.0).
    """
    exp: AdversarialExpectation = case_def["expectation"]
    scores = {}

    # 1. Completion check
    completed = report is not None and error is None
    if exp.should_complete:
        scores["completion"] = 1.0 if completed else 0.0
    else:
        # Expected to fail gracefully (e.g., validation error)
        scores["completion"] = 1.0 if not completed else 0.5  # Still okay if it completes

    if not report:
        # Can't score anything else if no report
        scores["graceful_failure"] = 1.0 if error else 0.0
        return scores

    # 2. Uncertainty flagging
    if exp.should_flag_uncertainty:
        has_caveats = len(report.caveats) > 0
        uncertainty_words = {"uncertain", "limited", "insufficient", "unclear",
                           "caveat", "caution", "note", "warning", "incomplete",
                           "contradictory", "inconsistent", "unable", "cannot"}
        report_text = " ".join([
            report.patient_summary or "",
            " ".join(report.caveats),
            " ".join(c.description for c in report.conflicts),
        ]).lower()
        flags_uncertainty = has_caveats or any(w in report_text for w in uncertainty_words)
        scores["flags_uncertainty"] = 1.0 if flags_uncertainty else 0.0

    # 3. Minimum caveats
    if exp.min_caveats > 0:
        scores["has_min_caveats"] = 1.0 if len(report.caveats) >= exp.min_caveats else 0.0

    # 4. Drug interaction flagging
    if exp.should_flag_interactions:
        scores["flags_interactions"] = 1.0 if report.drug_interaction_warnings else 0.0

    # 5. Allergy conflict flagging
    if exp.should_flag_allergy:
        allergy_words = {"allergy", "allergic", "anaphylaxis", "hypersensitivity",
                        "contraindicated", "penicillin"}
        report_text = " ".join([
            " ".join(c.description for c in report.conflicts),
            " ".join(report.caveats),
            report.patient_summary or "",
        ]).lower()
        scores["flags_allergy"] = 1.0 if any(w in report_text for w in allergy_words) else 0.0

    # 6. Differential diagnosis check
    if exp.should_have_differential:
        scores["has_differential"] = 1.0 if report.differential_diagnosis else 0.0

    # 7. Max diagnoses (sanity check)
    if exp.max_diagnoses is not None:
        n_dx = len(report.differential_diagnosis)
        scores["reasonable_dx_count"] = 1.0 if n_dx <= exp.max_diagnoses else 0.0

    return scores


# ──────────────────────────────────────────────
# Validation harness
# ──────────────────────────────────────────────

async def validate_adversarial(
    include_drug_check: bool = True,
    include_guidelines: bool = True,
    delay_between_cases: float = 2.0,
    cases: Optional[List[dict]] = None,
) -> ValidationSummary:
    """
    Run all adversarial test cases through the pipeline.

    Args:
        include_drug_check: Whether to run drug interaction check
        include_guidelines: Whether to include guideline retrieval
        delay_between_cases: Seconds to wait between cases
        cases: Optional list of specific cases (defaults to ADVERSARIAL_CASES)
    """
    test_cases = cases or ADVERSARIAL_CASES
    results: List[ValidationResult] = []
    start_time = time.time()

    for i, case_def in enumerate(test_cases):
        case_id = case_def["case_id"]
        category = case_def["category"]
        desc = case_def["description"]

        print(f"\n  [{i+1}/{len(test_cases)}] {case_id} ({category}): ", end="", flush=True)

        case_start = time.monotonic()

        try:
            state, report, error = await run_cds_pipeline(
                patient_text=case_def["input_text"],
                include_drug_check=include_drug_check,
                include_guidelines=include_guidelines,
            )
        except Exception as e:
            state, report, error = None, None, str(e)

        elapsed_ms = int((time.monotonic() - case_start) * 1000)

        step_results = {}
        if state:
            step_results = {s.step_id: s.status.value for s in state.steps}

        scores = score_adversarial(case_def, report, state, error)

        details = {
            "category": category,
            "description": desc,
            "num_caveats": len(report.caveats) if report else 0,
            "num_conflicts": len(report.conflicts) if report else 0,
            "num_diagnoses": len(report.differential_diagnosis) if report else 0,
            "num_interactions": len(report.drug_interaction_warnings) if report else 0,
            "caveats": report.caveats[:5] if report else [],
            "error": error,
        }

        # Console output
        passed = all(v >= 0.5 for v in scores.values())
        status = "PASS" if passed else "FAIL"
        icon = "+" if passed else "-"
        score_summary = " ".join(f"{k}={'Y' if v >= 0.5 else 'N'}" for k, v in scores.items())
        print(f"{icon} [{status}] {score_summary} ({elapsed_ms}ms)")
        if not passed:
            print(f"    Description: {desc}")

        result = ValidationResult(
            case_id=case_id,
            source_dataset="adversarial",
            success=report is not None,
            scores=scores,
            pipeline_time_ms=elapsed_ms,
            step_results=step_results,
            report_summary=report.patient_summary[:200] if report else None,
            error=error,
            details=details,
        )
        results.append(result)

        if i < len(test_cases) - 1:
            await asyncio.sleep(delay_between_cases)

    # Aggregate
    total = len(results)
    successful = sum(1 for r in results if r.success)

    # Per-metric averages
    all_metric_names = set()
    for r in results:
        all_metric_names.update(r.scores.keys())

    metrics = {}
    for m in sorted(all_metric_names):
        values = [r.scores[m] for r in results if m in r.scores]
        metrics[m] = sum(values) / len(values) if values else 0.0

    # Pass rate
    pass_count = sum(
        1 for r in results
        if all(v >= 0.5 for v in r.scores.values())
    )
    metrics["pass_rate"] = pass_count / total if total else 0.0

    # Per-category breakdown
    by_category: dict = {}
    for r in results:
        cat = r.details.get("category", "unknown")
        by_category.setdefault(cat, []).append(r)

    for cat, cat_results in by_category.items():
        cat_pass = sum(1 for r in cat_results if all(v >= 0.5 for v in r.scores.values()))
        metrics[f"pass_rate_{cat}"] = cat_pass / len(cat_results)
        metrics[f"count_{cat}"] = len(cat_results)

    times = [r.pipeline_time_ms for r in results if r.success]
    metrics["avg_pipeline_time_ms"] = sum(times) / len(times) if times else 0

    summary = ValidationSummary(
        dataset="adversarial",
        total_cases=total,
        successful_cases=successful,
        failed_cases=total - successful,
        metrics=metrics,
        per_case=results,
        run_duration_sec=time.time() - start_time,
    )

    return summary


# ──────────────────────────────────────────────
# Standalone runner
# ──────────────────────────────────────────────

async def main():
    """Run adversarial tests standalone."""
    import argparse

    parser = argparse.ArgumentParser(description="Adversarial Test Suite")
    parser.add_argument("--no-drugs", action="store_true", help="Skip drug interaction check")
    parser.add_argument("--no-guidelines", action="store_true", help="Skip guideline retrieval")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between cases (seconds)")
    args = parser.parse_args()

    print("Adversarial Test Suite")
    print("=" * 40)
    print(f"  Test cases: {len(ADVERSARIAL_CASES)}")

    summary = await validate_adversarial(
        include_drug_check=not args.no_drugs,
        include_guidelines=not args.no_guidelines,
        delay_between_cases=args.delay,
    )

    print_summary(summary)
    path = save_results(summary)
    print(f"Results saved to: {path}")


if __name__ == "__main__":
    asyncio.run(main())
