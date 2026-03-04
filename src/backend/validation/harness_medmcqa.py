"""
MedMCQA dataset fetcher and validation harness.

Downloads MedMCQA questions (Indian medical entrance exam — AIIMS/PGI)
and evaluates the CDS pipeline's diagnostic and clinical accuracy.

Source: https://huggingface.co/datasets/openlifescienceai/medmcqa
Format: JSON with {question, opa, opb, opc, opd, cop, choice_type, exp, subject_name, topic_name}

Fields:
  - question: the clinical question
  - opa/opb/opc/opd: four answer options
  - cop: correct option index (0=A, 1=B, 2=C, 3=D)
  - choice_type: "single" or "multi"
  - exp: explanation
  - subject_name: medical subject (e.g., "Medicine", "Surgery")
  - topic_name: specific topic within subject

Metrics:
  - mcq_accuracy: LLM selects correct MCQ option
  - top1_accuracy: Correct answer matches #1 differential
  - top3_accuracy: Correct answer in top 3 differentials
  - mentioned_accuracy: Correct answer mentioned anywhere in report
  - parse_success_rate: Pipeline completed without crashing
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import List, Optional

import httpx

from validation.base import (
    DATA_DIR,
    ValidationCase,
    ValidationResult,
    ValidationSummary,
    clear_checkpoint,
    diagnosis_in_differential,
    ensure_data_dir,
    fuzzy_match,
    load_checkpoint,
    normalize_text,
    print_summary,
    run_cds_pipeline,
    save_incremental,
    save_results,
    score_case,
)
from validation.question_classifier import (
    classify_question_from_text,
    QuestionType,
    PIPELINE_APPROPRIATE_TYPES,
)
from app.services.medgemma import MedGemmaService

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Data fetching
# ──────────────────────────────────────────────

# HuggingFace Hub API for streaming the dataset
MEDMCQA_HF_URL = "https://datasets-server.huggingface.co/rows?dataset=openlifescienceai/medmcqa&config=default&split=validation&offset={offset}&length={length}"

# Fallback: direct parquet download
MEDMCQA_PARQUET_URL = "https://huggingface.co/datasets/openlifescienceai/medmcqa/resolve/main/data/validation-00000-of-00001.parquet"


async def fetch_medmcqa(
    max_cases: int = 50,
    seed: int = 42,
    subjects: Optional[List[str]] = None,
) -> List[ValidationCase]:
    """
    Download MedMCQA validation set and convert to ValidationCase objects.

    Args:
        max_cases: Maximum number of cases to sample
        seed: Random seed for reproducible sampling
        subjects: Optional list of subjects to filter (e.g., ["Medicine", "Surgery"])
    """
    ensure_data_dir()
    cache_path = DATA_DIR / "medmcqa_validation.json"

    if cache_path.exists():
        print(f"  Loading MedMCQA from cache: {cache_path}")
        raw_cases = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        print(f"  Downloading MedMCQA validation set...")
        raw_cases = await _download_medmcqa(cache_path)

    if not raw_cases:
        raise RuntimeError("Failed to fetch MedMCQA data. Check network connection.")

    # Filter by subject if requested
    if subjects:
        subjects_lower = {s.lower() for s in subjects}
        raw_cases = [
            c for c in raw_cases
            if c.get("subject_name", "").lower() in subjects_lower
        ]
        print(f"  Filtered to {len(raw_cases)} cases in subjects: {subjects}")

    # Filter to single-choice only (multi not suitable for pipeline scoring)
    raw_cases = [c for c in raw_cases if c.get("choice_type", "single") == "single"]

    # Sample
    random.seed(seed)
    if len(raw_cases) > max_cases:
        raw_cases = random.sample(raw_cases, max_cases)

    # Convert to ValidationCase
    cases = []
    option_keys = ["opa", "opb", "opc", "opd"]
    idx_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}

    for i, item in enumerate(raw_cases):
        question = item.get("question", "")
        options = {
            "A": item.get("opa", ""),
            "B": item.get("opb", ""),
            "C": item.get("opc", ""),
            "D": item.get("opd", ""),
        }
        correct_idx = item.get("cop", 0)
        correct_letter = idx_to_letter.get(correct_idx, "A")
        correct_answer = options.get(correct_letter, "")

        # Build clinical vignette from question
        vignette, question_stem = _split_question(question)

        case_obj = ValidationCase(
            case_id=f"medmcqa_{i:04d}",
            source_dataset="medmcqa",
            input_text=vignette,
            ground_truth={
                "correct_answer": correct_answer,
                "answer_idx": correct_letter,
                "options": options,
                "full_question": question,
                "explanation": item.get("exp", ""),
            },
            metadata={
                "question_stem": question_stem,
                "clinical_vignette": vignette,
                "full_question_with_stem": question,
                "subject_name": item.get("subject_name", ""),
                "topic_name": item.get("topic_name", ""),
            },
        )

        # Classify question type
        case_obj.metadata["question_type"] = classify_question_from_text(question).value
        cases.append(case_obj)

    print(f"  Loaded {len(cases)} MedMCQA cases")
    return cases


async def _download_medmcqa(cache_path: Path) -> List[dict]:
    """Download MedMCQA validation set from HuggingFace."""
    all_rows = []

    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        # Fetch in batches via the datasets-server API
        offset = 0
        batch_size = 100
        max_fetch = 4000  # MedMCQA validation has ~4183 rows

        while offset < max_fetch:
            url = MEDMCQA_HF_URL.format(offset=offset, length=batch_size)
            try:
                r = await client.get(url)
                r.raise_for_status()
                data = r.json()
                rows = data.get("rows", [])
                if not rows:
                    break
                # Extract the row data from the HF format
                for row in rows:
                    row_data = row.get("row", row)
                    all_rows.append(row_data)
                offset += batch_size
                # Small progress indicator
                if offset % 500 == 0:
                    print(f"    Fetched {len(all_rows)} rows...")
            except Exception as e:
                logger.warning(f"Batch fetch failed at offset {offset}: {e}")
                if all_rows:
                    break  # Use what we have
                raise

        if all_rows:
            cache_path.write_text(
                json.dumps(all_rows, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"  Cached {len(all_rows)} MedMCQA cases to {cache_path}")

    return all_rows


def _split_question(question: str) -> tuple:
    """
    Split a MedMCQA question into (clinical_vignette, question_stem).

    MedMCQA questions vary — some are pure factual, some have clinical scenarios.
    """
    stems = [
        r"which of the following",
        r"what is the most likely",
        r"what is the best",
        r"what is the most appropriate",
        r"what is the diagnosis",
        r"the most likely diagnosis is",
        r"what is the treatment",
        r"the treatment of choice",
        r"drug of choice",
        r"investigation of choice",
        r"most common",
        r"all of the following.*except",
        r"true about",
        r"false about",
    ]

    text = question.strip()
    for stem in stems:
        pattern = re.compile(
            r'\.?\s*([A-Z][^.]*?' + stem + r'[^.]*[\?\.]?)\s*$',
            re.IGNORECASE,
        )
        match = pattern.search(text)
        if match:
            vignette = text[:match.start()].strip()
            q_stem = match.group(1).strip()
            if len(vignette) > 30:
                return vignette, q_stem

    return text, ""


# ──────────────────────────────────────────────
# MCQ answer selection (reuses pattern from harness_medqa)
# ──────────────────────────────────────────────

MCQ_SELECTION_PROMPT = """You are a medical expert answering a multiple-choice question.

A clinical decision support system has produced the following analysis of a patient case:

=== CDS REPORT ===
{report_summary}

=== QUESTION ===
{full_question}

=== OPTIONS ===
{options_text}

Based on the CDS analysis and your medical knowledge, select the single best answer.
Respond with ONLY the letter (A, B, C, or D) on the first line, then a one-sentence justification.
"""


async def select_mcq_answer(
    case: ValidationCase,
    report,
    state=None,
) -> tuple:
    """
    Use MedGemma to select an MCQ answer given CDS report context.

    Returns:
        (selected_letter, justification) e.g. ("B", "The patient's symptoms...")
    """
    parts = []
    if report.patient_summary:
        parts.append(f"Patient Summary: {report.patient_summary}")
    if report.differential_diagnosis:
        dx_list = ", ".join(d.diagnosis for d in report.differential_diagnosis[:5])
        parts.append(f"Differential Diagnosis: {dx_list}")
    if report.suggested_next_steps:
        steps = ", ".join(a.action for a in report.suggested_next_steps[:5])
        parts.append(f"Suggested Next Steps: {steps}")
    if report.guideline_recommendations:
        recs = ", ".join(report.guideline_recommendations[:3])
        parts.append(f"Guideline Recommendations: {recs}")
    report_summary = "\n".join(parts) if parts else "No report available."

    options = case.ground_truth.get("options", {})
    options_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    full_question = case.ground_truth.get("full_question", case.input_text)

    prompt = MCQ_SELECTION_PROMPT.format(
        report_summary=report_summary,
        full_question=full_question,
        options_text=options_text,
    )

    service = MedGemmaService()
    response = await service.generate(
        prompt=prompt,
        max_tokens=100,
        temperature=0.1,
    )

    lines = response.strip().split("\n")
    selected = lines[0].strip().rstrip(".").upper()

    for char in selected:
        if char in "ABCD":
            selected = char
            break
    else:
        selected = "X"

    justification = " ".join(lines[1:]).strip() if len(lines) > 1 else ""
    return selected, justification


# ──────────────────────────────────────────────
# Validation harness
# ──────────────────────────────────────────────

async def validate_medmcqa(
    cases: List[ValidationCase],
    include_drug_check: bool = False,
    include_guidelines: bool = True,
    include_mcq: bool = True,
    delay_between_cases: float = 2.0,
    resume: bool = False,
) -> ValidationSummary:
    """
    Run MedMCQA cases through the CDS pipeline and score results.

    Args:
        cases: List of MedMCQA ValidationCases
        include_drug_check: Whether to run drug interaction check
        include_guidelines: Whether to include guideline retrieval
        include_mcq: Whether to run MCQ answer selection step
        delay_between_cases: Seconds to wait between cases (rate limiting)
        resume: If True, skip cases already in checkpoint and continue
    """
    results: List[ValidationResult] = []
    start_time = time.time()

    # Resume support
    completed_ids: set = set()
    if resume:
        prior = load_checkpoint("medmcqa")
        if prior:
            results.extend(prior)
            completed_ids = {r.case_id for r in prior}
            print(f"  Resuming: {len(prior)} cases loaded from checkpoint, {len(cases) - len(completed_ids)} remaining")
    else:
        clear_checkpoint("medmcqa")

    for i, case in enumerate(cases):
        if case.case_id in completed_ids:
            print(f"\n  [{i+1}/{len(cases)}] {case.case_id}: (cached) skipped")
            continue

        print(f"\n  [{i+1}/{len(cases)}] {case.case_id}: ", end="", flush=True)

        case_start = time.monotonic()

        state, report, error = await run_cds_pipeline(
            patient_text=case.input_text,
            include_drug_check=include_drug_check,
            include_guidelines=include_guidelines,
        )

        elapsed_ms = int((time.monotonic() - case_start) * 1000)

        step_results = {}
        if state:
            step_results = {s.step_id: s.status.value for s in state.steps}

        scores = {}
        details = {}
        correct_answer = case.ground_truth["correct_answer"]
        question_type = case.metadata.get("question_type", "other")

        if report:
            scores = score_case(
                target_answer=correct_answer,
                report=report,
                question_type=question_type,
                reasoning_result=state.clinical_reasoning if state else None,
            )
            scores["parse_success"] = 1.0

            match_location = scores.pop("match_location", "not_found")
            match_rank = scores.pop("match_rank", -1)

            # MCQ answer selection
            if include_mcq and case.ground_truth.get("options"):
                try:
                    selected, justification = await select_mcq_answer(case, report, state)
                    mcq_correct_idx = case.ground_truth.get("answer_idx", "")
                    scores["mcq_accuracy"] = 1.0 if selected.upper() == mcq_correct_idx.upper() else 0.0
                    details["mcq_selected"] = selected
                    details["mcq_justification"] = justification
                    details["mcq_correct"] = mcq_correct_idx
                except Exception as e:
                    logger.warning(f"MCQ selection failed for {case.case_id}: {e}")
                    scores["mcq_accuracy"] = 0.0
                    details["mcq_error"] = str(e)

            all_dx = [dx.diagnosis for dx in report.differential_diagnosis]
            all_next = [a.action for a in report.suggested_next_steps]
            all_recs = list(report.guideline_recommendations)

            details.update({
                "correct_answer": correct_answer,
                "question_type": question_type,
                "subject_name": case.metadata.get("subject_name", ""),
                "topic_name": case.metadata.get("topic_name", ""),
                "top_diagnosis": all_dx[0] if all_dx else "NONE",
                "all_diagnoses": all_dx,
                "all_next_steps": all_next[:5],
                "all_recommendations": all_recs[:5],
                "num_diagnoses": len(report.differential_diagnosis),
                "match_location": match_location,
                "match_rank": match_rank,
                "patient_summary": report.patient_summary[:300] if report.patient_summary else "",
                "explanation": case.ground_truth.get("explanation", ""),
            })

            mentioned = scores.get("mentioned_accuracy", 0.0) > 0
            mcq_tag = ""
            if "mcq_accuracy" in scores:
                mcq_tag = f" mcq={'Y' if scores['mcq_accuracy'] > 0 else 'N'}"
            loc_tag = f"[{match_location}]" if mentioned else ""
            status_icon = "+" if mentioned else "-"
            print(f"{status_icon} [{question_type}] top1={'Y' if scores.get('top1_accuracy', 0) > 0 else 'N'} mentioned={'Y' if mentioned else 'N'}{mcq_tag} {loc_tag} ({elapsed_ms}ms)")
        else:
            scores = {
                "top1_accuracy": 0.0,
                "top3_accuracy": 0.0,
                "mentioned_accuracy": 0.0,
                "differential_accuracy": 0.0,
                "parse_success": 0.0,
            }
            details = {
                "correct_answer": correct_answer,
                "question_type": question_type,
                "subject_name": case.metadata.get("subject_name", ""),
                "error": error,
                "match_location": "not_found",
            }
            print(f"- FAILED: {error[:80] if error else 'unknown'}")

        result = ValidationResult(
            case_id=case.case_id,
            source_dataset="medmcqa",
            success=report is not None,
            scores=scores,
            pipeline_time_ms=elapsed_ms,
            step_results=step_results,
            report_summary=report.patient_summary[:200] if report else None,
            error=error,
            details=details,
        )
        results.append(result)
        save_incremental(result, "medmcqa")

        if i < len(cases) - 1:
            await asyncio.sleep(delay_between_cases)

    # Aggregate metrics
    total = len(results)
    successful = sum(1 for r in results if r.success)

    metric_names = [
        "top1_accuracy", "top3_accuracy", "mentioned_accuracy",
        "differential_accuracy", "parse_success", "mcq_accuracy",
    ]
    metrics = {}
    for m in metric_names:
        values = [r.scores.get(m, 0.0) for r in results if m in r.scores]
        metrics[m] = sum(values) / len(values) if values else 0.0

    times = [r.pipeline_time_ms for r in results if r.success]
    metrics["avg_pipeline_time_ms"] = sum(times) / len(times) if times else 0

    # Stratified by question type
    by_type: dict = {}
    for r in results:
        qt = r.details.get("question_type", "other")
        by_type.setdefault(qt, []).append(r)

    for qt, type_results in by_type.items():
        n = len(type_results)
        metrics[f"count_{qt}"] = n
        for m in ["top1_accuracy", "top3_accuracy", "mentioned_accuracy", "mcq_accuracy"]:
            values = [r.scores.get(m, 0.0) for r in type_results if m in r.scores]
            if values:
                metrics[f"{m}_{qt}"] = sum(values) / len(values)

    # Stratified by subject
    by_subject: dict = {}
    for r in results:
        subj = r.details.get("subject_name", "unknown")
        by_subject.setdefault(subj, []).append(r)

    for subj, subj_results in by_subject.items():
        n = len(subj_results)
        metrics[f"count_subject_{subj}"] = n
        for m in ["mcq_accuracy", "mentioned_accuracy"]:
            values = [r.scores.get(m, 0.0) for r in subj_results if m in r.scores]
            if values:
                metrics[f"{m}_subject_{subj}"] = sum(values) / len(values)

    # Pipeline-appropriate subset
    appropriate_types = {t.value for t in PIPELINE_APPROPRIATE_TYPES}
    appropriate_results = [
        r for r in results
        if r.details.get("question_type", "other") in appropriate_types
    ]
    if appropriate_results:
        for m in ["top1_accuracy", "top3_accuracy", "mentioned_accuracy"]:
            values = [r.scores.get(m, 0.0) for r in appropriate_results]
            metrics[f"{m}_pipeline_appropriate"] = sum(values) / len(values) if values else 0.0
        metrics["count_pipeline_appropriate"] = len(appropriate_results)

    summary = ValidationSummary(
        dataset="medmcqa",
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
    """Run MedMCQA validation standalone."""
    import argparse

    parser = argparse.ArgumentParser(description="MedMCQA Validation")
    parser.add_argument("--max-cases", type=int, default=10, help="Number of cases to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--include-drugs", action="store_true", help="Include drug interaction check")
    parser.add_argument("--no-mcq", action="store_true", help="Disable MCQ answer selection step")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between cases (seconds)")
    parser.add_argument("--subjects", nargs="+", help="Filter by subject (e.g., Medicine Surgery)")
    args = parser.parse_args()

    print("MedMCQA Validation Harness")
    print("=" * 40)

    cases = await fetch_medmcqa(
        max_cases=args.max_cases,
        seed=args.seed,
        subjects=args.subjects,
    )
    summary = await validate_medmcqa(
        cases,
        include_drug_check=args.include_drugs,
        include_mcq=not args.no_mcq,
        delay_between_cases=args.delay,
    )

    print_summary(summary)
    path = save_results(summary)
    print(f"Results saved to: {path}")


if __name__ == "__main__":
    asyncio.run(main())
