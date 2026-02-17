"""
MedQA dataset fetcher and validation harness.

Downloads MedQA USMLE 4-option questions and evaluates the CDS pipeline's
ability to arrive at the correct diagnosis / answer.

Source: https://github.com/jind11/MedQA
Format: JSONL with {question, options: {A, B, C, D}, answer_idx, answer}

Metrics:
  - top1_accuracy: Correct answer matches #1 differential diagnosis
  - top3_accuracy: Correct answer in top 3 differential diagnoses
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
    classify_question,
    QuestionType,
    PIPELINE_APPROPRIATE_TYPES,
)
from app.services.medgemma import MedGemmaService

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Data fetching
# ──────────────────────────────────────────────

# HuggingFace direct download (JSONL)
MEDQA_JSONL_URL = "https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options/resolve/main/phrases_no_exclude_test.jsonl"


async def fetch_medqa(max_cases: int = 50, seed: int = 42) -> List[ValidationCase]:
    """
    Download MedQA test set and convert to ValidationCase objects.

    Args:
        max_cases: Maximum number of cases to sample
        seed: Random seed for reproducible sampling
    """
    ensure_data_dir()
    cache_path = DATA_DIR / "medqa_test.jsonl"

    # Try to load from cache
    if cache_path.exists():
        print(f"  Loading MedQA from cache: {cache_path}")
        raw_cases = _load_jsonl(cache_path)
    else:
        print(f"  Downloading MedQA test set...")
        raw_cases = await _download_medqa_jsonl(cache_path)

    if not raw_cases:
        raise RuntimeError("Failed to fetch MedQA data. Check network connection.")

    # Sample
    random.seed(seed)
    if len(raw_cases) > max_cases:
        raw_cases = random.sample(raw_cases, max_cases)

    # Convert to ValidationCase
    cases = []
    for i, item in enumerate(raw_cases):
        question = item.get("question", "")
        options = item.get("options", item.get("answer_choices", {}))
        answer_idx = item.get("answer_idx", item.get("answer", ""))
        answer_text = item.get("answer", "")

        # Handle different formats
        if isinstance(options, dict):
            if answer_idx in options:
                answer_text = options[answer_idx]
        elif isinstance(options, list):
            # Some formats have options as a list
            idx = ord(answer_idx) - ord('A') if isinstance(answer_idx, str) and len(answer_idx) == 1 else 0
            if idx < len(options):
                answer_text = options[idx]

        # Split question into vignette + stem (P3: preserve the question stem)
        vignette, question_stem = _split_question(question)

        case_obj = ValidationCase(
            case_id=f"medqa_{i:04d}",
            source_dataset="medqa",
            input_text=vignette,
            ground_truth={
                "correct_answer": answer_text,
                "answer_idx": answer_idx,
                "options": options,
                "full_question": question,
            },
            metadata={
                "question_stem": question_stem,
                "clinical_vignette": vignette,
                "full_question_with_stem": question,
            },
        )

        # Classify question type (P1)
        case_obj.metadata["question_type"] = classify_question(case_obj).value
        cases.append(case_obj)

    print(f"  Loaded {len(cases)} MedQA cases")
    return cases


async def _download_medqa_jsonl(cache_path: Path) -> List[dict]:
    """Download MedQA JSONL from GitHub."""
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        try:
            r = await client.get(MEDQA_JSONL_URL)
            r.raise_for_status()

            lines = r.text.strip().split('\n')
            cases = [json.loads(line) for line in lines if line.strip()]

            # Cache
            cache_path.write_text('\n'.join(json.dumps(c) for c in cases))
            print(f"  Cached {len(cases)} MedQA cases to {cache_path}")
            return cases

        except Exception as e:
            print(f"  Warning: Failed to download MedQA: {e}")
            return []


def _load_jsonl(path: Path) -> List[dict]:
    """Load JSONL file."""
    cases = []
    for line in path.read_text(encoding="utf-8").strip().split('\n'):
        if line.strip():
            cases.append(json.loads(line))
    return cases


def _split_question(question: str) -> tuple:
    """
    Split a USMLE question into (clinical_vignette, question_stem).

    Returns:
        (vignette, stem) where vignette is the clinical narrative and
        stem is the trailing question sentence (e.g. "Which of the following...").
        If no stem is found, returns (full_question, "").
    """
    stems = [
        r"which of the following",
        r"what is the most likely",
        r"what is the best next step",
        r"what is the most appropriate",
        r"what is the diagnosis",
        r"the most likely diagnosis is",
        r"this patient most likely has",
        r"what would be the next step",
        r"what is the next best",
        r"what is the underlying",
        r"what is the mechanism",
        r"which vitamin",
        r"which enzyme",
        r"which receptor",
        r"which drug",
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
            if len(vignette) > 50:
                return vignette, q_stem

    # Fallback: no stem detected
    return text, ""


# ──────────────────────────────────────────────
# MCQ answer selection (P6)
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
    # Build report summary
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

    # Build options text
    options = case.ground_truth.get("options", {})
    if isinstance(options, dict):
        options_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    elif isinstance(options, list):
        options_text = "\n".join(
            f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)
        )
    else:
        options_text = str(options)

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

    # Parse response: first line should be the letter
    lines = response.strip().split("\n")
    selected = lines[0].strip().rstrip(".").upper()

    # Extract just the letter if response is longer
    for char in selected:
        if char in "ABCDEFGH":
            selected = char
            break
    else:
        selected = "X"  # Unparseable

    justification = " ".join(lines[1:]).strip() if len(lines) > 1 else ""

    return selected, justification


# ──────────────────────────────────────────────
# Validation harness
# ──────────────────────────────────────────────

async def validate_medqa(
    cases: List[ValidationCase],
    include_drug_check: bool = False,
    include_guidelines: bool = True,
    include_mcq: bool = True,
    delay_between_cases: float = 2.0,
    resume: bool = False,
) -> ValidationSummary:
    """
    Run MedQA cases through the CDS pipeline and score results.

    Args:
        cases: List of MedQA ValidationCases
        include_drug_check: Whether to run drug interaction check (slower)
        include_guidelines: Whether to include guideline retrieval
        include_mcq: Whether to run MCQ answer selection step (adds 1 LLM call/case)
        delay_between_cases: Seconds to wait between cases (rate limiting)
        resume: If True, skip cases already in checkpoint and continue
    """
    results: List[ValidationResult] = []
    start_time = time.time()

    # Resume support: load completed cases from checkpoint
    completed_ids: set = set()
    if resume:
        prior = load_checkpoint("medqa")
        if prior:
            results.extend(prior)
            completed_ids = {r.case_id for r in prior}
            print(f"  Resuming: {len(prior)} cases loaded from checkpoint, {len(cases) - len(completed_ids)} remaining")
    else:
        clear_checkpoint("medqa")

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

        # Build step results
        step_results = {}
        if state:
            step_results = {s.step_id: s.status.value for s in state.steps}

        # Score (type-aware: P4)
        scores = {}
        details = {}
        correct_answer = case.ground_truth["correct_answer"]
        question_type = case.metadata.get("question_type", "other")

        if report:
            # Type-aware scoring
            scores = score_case(
                target_answer=correct_answer,
                report=report,
                question_type=question_type,
                reasoning_result=state.clinical_reasoning if state else None,
            )
            scores["parse_success"] = 1.0

            # Extract non-float detail fields from scores dict
            match_location = scores.pop("match_location", "not_found")
            match_rank = scores.pop("match_rank", -1)

            # MCQ answer selection (P6)
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

            # Rich details for debugging
            all_dx = [dx.diagnosis for dx in report.differential_diagnosis]
            all_next = [a.action for a in report.suggested_next_steps]
            all_recs = list(report.guideline_recommendations)

            details.update({
                "correct_answer": correct_answer,
                "question_type": question_type,
                "top_diagnosis": all_dx[0] if all_dx else "NONE",
                "all_diagnoses": all_dx,
                "all_next_steps": all_next[:5],
                "all_recommendations": all_recs[:5],
                "num_diagnoses": len(report.differential_diagnosis),
                "match_location": match_location,
                "match_rank": match_rank,
                "patient_summary": report.patient_summary[:300] if report.patient_summary else "",
            })

            # Console output
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
                "error": error,
                "match_location": "not_found",
            }
            print(f"- FAILED: {error[:80] if error else 'unknown'}")

        result = ValidationResult(
            case_id=case.case_id,
            source_dataset="medqa",
            success=report is not None,
            scores=scores,
            pipeline_time_ms=elapsed_ms,
            step_results=step_results,
            report_summary=report.patient_summary[:200] if report else None,
            error=error,
            details=details,
        )
        results.append(result)
        save_incremental(result, "medqa")  # checkpoint after every case

        # Rate limit
        if i < len(cases) - 1:
            await asyncio.sleep(delay_between_cases)

    # Aggregate
    total = len(results)
    successful = sum(1 for r in results if r.success)

    # Overall metrics
    metric_names = [
        "top1_accuracy", "top3_accuracy", "mentioned_accuracy",
        "differential_accuracy", "parse_success", "mcq_accuracy",
    ]
    metrics = {}
    for m in metric_names:
        values = [r.scores.get(m, 0.0) for r in results if m in r.scores]
        metrics[m] = sum(values) / len(values) if values else 0.0

    # Average pipeline time
    times = [r.pipeline_time_ms for r in results if r.success]
    metrics["avg_pipeline_time_ms"] = sum(times) / len(times) if times else 0

    # Stratified metrics by question type (P7)
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

    # Pipeline-appropriate subset (diagnostic + treatment + lab_finding)
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
        dataset="medqa",
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
    """Run MedQA validation standalone."""
    import argparse

    parser = argparse.ArgumentParser(description="MedQA Validation")
    parser.add_argument("--max-cases", type=int, default=10, help="Number of cases to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--include-drugs", action="store_true", help="Include drug interaction check")
    parser.add_argument("--no-mcq", action="store_true", help="Disable MCQ answer selection step")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between cases (seconds)")
    args = parser.parse_args()

    print("MedQA Validation Harness")
    print("=" * 40)

    cases = await fetch_medqa(max_cases=args.max_cases, seed=args.seed)
    summary = await validate_medqa(
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
