"""
HeadQA dataset fetcher and validation harness.

Downloads HeadQA (Spanish healthcare professional exam questions, English
translation) and evaluates the CDS pipeline's clinical knowledge.

Covers medicine, nursing, pharmacology, psychology, biology, and chemistry
from the Spanish healthcare professional licensing exams.

Source: https://huggingface.co/datasets/head_qa
Format: {qid, qtext, ra (correct answer ID), category, year,
         answers: [{aid, atext}, ...]}

Metrics:
  - mcq_accuracy: LLM selects correct MCQ option
  - top1_accuracy: Correct answer matches #1 differential
  - top3_accuracy: Correct answer in top 3 differentials
  - mentioned_accuracy: Correct answer mentioned anywhere in report
  - parse_success_rate: Pipeline completed without crashing
  - Per-category accuracy breakdowns
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
# Constants
# ──────────────────────────────────────────────

# HeadQA categories
HEADQA_CATEGORIES = [
    "medicine",
    "nursing",
    "pharmacology",
    "psychology",
    "biology",
    "chemistry",
]

# HuggingFace Datasets API — English config
HEADQA_HF_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=head_qa&config=en&split=test"
    "&offset={offset}&length={length}"
)


# ──────────────────────────────────────────────
# Data fetching
# ──────────────────────────────────────────────

async def fetch_headqa(
    max_cases: int = 50,
    seed: int = 42,
    categories: Optional[List[str]] = None,
) -> List[ValidationCase]:
    """
    Download HeadQA English test set and convert to ValidationCase objects.

    Args:
        max_cases: Maximum total cases to sample
        seed: Random seed for reproducible sampling
        categories: Optional list of categories to include
                    (defaults to all: medicine, nursing, pharmacology, etc.)
    """
    ensure_data_dir()
    cache_path = DATA_DIR / "headqa_en_test.json"

    if cache_path.exists():
        print(f"  Loading HeadQA from cache: {cache_path}")
        raw_cases = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        print(f"  Downloading HeadQA English test set...")
        raw_cases = await _download_headqa(cache_path)

    if not raw_cases:
        raise RuntimeError("Failed to fetch HeadQA data. Check network connection.")

    # Filter by category
    if categories:
        cats_lower = {c.lower() for c in categories}
        raw_cases = [
            c for c in raw_cases
            if c.get("category", "").lower() in cats_lower
        ]
        print(f"  Filtered to {len(raw_cases)} cases in categories: {categories}")

    # Sample
    random.seed(seed)
    if len(raw_cases) > max_cases:
        raw_cases = random.sample(raw_cases, max_cases)

    # Convert to ValidationCase
    cases = []
    for i, item in enumerate(raw_cases):
        question = item.get("qtext", "")
        answers_raw = item.get("answers", [])
        correct_aid = str(item.get("ra", ""))
        category = item.get("category", "unknown")
        year = item.get("year", "")

        # Build options dict (A, B, C, D, ...) from answers list
        options = {}
        correct_answer = ""
        correct_letter = "A"
        for j, ans in enumerate(answers_raw):
            letter = chr(65 + j)  # A, B, C, D, ...
            atext = ans.get("atext", "")
            options[letter] = atext
            if str(ans.get("aid", "")) == correct_aid:
                correct_answer = atext
                correct_letter = letter

        if not correct_answer and answers_raw:
            # Fallback: ra might be 1-indexed position
            try:
                ra_idx = int(correct_aid) - 1
                if 0 <= ra_idx < len(answers_raw):
                    correct_answer = answers_raw[ra_idx].get("atext", "")
                    correct_letter = chr(65 + ra_idx)
            except (ValueError, IndexError):
                correct_answer = answers_raw[0].get("atext", "")
                correct_letter = "A"

        # Split question
        vignette, question_stem = _split_question(question)

        # Fallback: if vignette is too short for CaseSubmission (min 10 chars),
        # use the full question text
        if len(vignette.strip()) < 10:
            vignette = question

        case_obj = ValidationCase(
            case_id=f"headqa_{i:04d}",
            source_dataset="headqa",
            input_text=vignette,
            ground_truth={
                "correct_answer": correct_answer,
                "answer_idx": correct_letter,
                "options": options,
                "full_question": question,
                "n_options": len(options),
            },
            metadata={
                "question_stem": question_stem,
                "clinical_vignette": vignette,
                "full_question_with_stem": question,
                "category": category,
                "year": str(year),
                "qid": item.get("qid", ""),
            },
        )

        # Classify question type
        case_obj.metadata["question_type"] = classify_question_from_text(question).value
        cases.append(case_obj)

    # Print category distribution
    cat_counts: dict = {}
    for c in cases:
        cat = c.metadata.get("category", "?")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    print(f"  Loaded {len(cases)} HeadQA cases")
    for cat, cnt in sorted(cat_counts.items()):
        print(f"    {cat}: {cnt}")

    return cases


async def _download_headqa(cache_path: Path) -> List[dict]:
    """Download HeadQA English from HuggingFace Datasets API."""
    all_rows = []

    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        offset = 0
        batch_size = 100
        max_fetch = 7000  # HeadQA has ~6,792 questions

        while offset < max_fetch:
            url = HEADQA_HF_URL.format(offset=offset, length=batch_size)
            try:
                r = await client.get(url)
                r.raise_for_status()
                data = r.json()
                rows = data.get("rows", [])
                if not rows:
                    break
                for row in rows:
                    row_data = row.get("row", row)
                    all_rows.append(row_data)
                offset += batch_size
                if offset % 1000 == 0:
                    print(f"    Fetched {len(all_rows)} rows...")
            except Exception as e:
                logger.warning(f"Batch fetch failed at offset {offset}: {e}")
                if all_rows:
                    break
                raise

        if all_rows:
            cache_path.write_text(
                json.dumps(all_rows, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"  Cached {len(all_rows)} HeadQA cases to {cache_path}")

    return all_rows


def _split_question(question: str) -> tuple:
    """Split a HeadQA question into (clinical_vignette, question_stem)."""
    stems = [
        r"which of the following",
        r"what is the most likely",
        r"what is the best",
        r"what is the most appropriate",
        r"indicate which",
        r"select the correct",
        r"the correct answer is",
        r"what is the",
        r"which is correct",
        r"which is true",
        r"which is false",
        r"which statement",
        r"what type of",
        r"which mechanism",
        r"the treatment of choice",
        r"drug of choice",
    ]

    text = question.strip()
    for stem in stems:
        pattern = re.compile(
            r'\.?\s*([A-Z][^.]*?' + stem + r'[^.]*[\?\.\:]?)\s*$',
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
# MCQ answer selection
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
Respond with ONLY the letter ({option_letters}) on the first line, then a one-sentence justification.
"""


async def select_mcq_answer(
    case: ValidationCase,
    report,
    state=None,
) -> tuple:
    """
    Use MedGemma to select an MCQ answer given CDS report context.
    Supports variable number of options.

    Returns:
        (selected_letter, justification)
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
    n_options = case.ground_truth.get("n_options", len(options))
    option_letters = ", ".join(chr(65 + i) for i in range(n_options))
    valid_chars = {chr(65 + i) for i in range(n_options)}

    prompt = MCQ_SELECTION_PROMPT.format(
        report_summary=report_summary,
        full_question=full_question,
        options_text=options_text,
        option_letters=option_letters,
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
        if char in valid_chars:
            selected = char
            break
    else:
        selected = "X"

    justification = " ".join(lines[1:]).strip() if len(lines) > 1 else ""
    return selected, justification


# ──────────────────────────────────────────────
# Validation harness
# ──────────────────────────────────────────────

async def validate_headqa(
    cases: List[ValidationCase],
    include_drug_check: bool = False,
    include_guidelines: bool = True,
    include_mcq: bool = True,
    delay_between_cases: float = 2.0,
    resume: bool = False,
) -> ValidationSummary:
    """
    Run HeadQA cases through the CDS pipeline and score results.

    Provides per-category breakdown (medicine, nursing, pharmacology, etc.)
    in addition to overall metrics.
    """
    results: List[ValidationResult] = []
    start_time = time.time()

    # Resume support
    completed_ids: set = set()
    if resume:
        prior = load_checkpoint("headqa")
        if prior:
            results.extend(prior)
            completed_ids = {r.case_id for r in prior}
            print(f"  Resuming: {len(prior)} cases loaded from checkpoint, {len(cases) - len(completed_ids)} remaining")
    else:
        clear_checkpoint("headqa")

    for i, case in enumerate(cases):
        if case.case_id in completed_ids:
            print(f"\n  [{i+1}/{len(cases)}] {case.case_id}: (cached) skipped")
            continue

        category = case.metadata.get("category", "?")
        print(f"\n  [{i+1}/{len(cases)}] {case.case_id} [{category}]: ", end="", flush=True)

        case_start = time.monotonic()

        # Ensure patient_text meets CaseSubmission min length (10 chars)
        patient_text = case.input_text
        if len(patient_text.strip()) < 10:
            patient_text = case.ground_truth.get("full_question", case.input_text) or case.input_text

        state, report, error = await run_cds_pipeline(
            patient_text=patient_text,
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
                "category": category,
                "year": case.metadata.get("year", ""),
                "top_diagnosis": all_dx[0] if all_dx else "NONE",
                "all_diagnoses": all_dx,
                "all_next_steps": all_next[:5],
                "all_recommendations": all_recs[:5],
                "num_diagnoses": len(report.differential_diagnosis),
                "match_location": match_location,
                "match_rank": match_rank,
                "patient_summary": report.patient_summary[:300] if report.patient_summary else "",
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
                "category": category,
                "error": error,
                "match_location": "not_found",
            }
            print(f"- FAILED: {error[:80] if error else 'unknown'}")

        result = ValidationResult(
            case_id=case.case_id,
            source_dataset="headqa",
            success=report is not None,
            scores=scores,
            pipeline_time_ms=elapsed_ms,
            step_results=step_results,
            report_summary=report.patient_summary[:200] if report else None,
            error=error,
            details=details,
        )
        results.append(result)
        save_incremental(result, "headqa")

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

    # Stratified by HeadQA category
    by_category: dict = {}
    for r in results:
        cat = r.details.get("category", "unknown")
        by_category.setdefault(cat, []).append(r)

    for cat, cat_results in by_category.items():
        n = len(cat_results)
        metrics[f"count_category_{cat}"] = n
        for m in ["mcq_accuracy", "mentioned_accuracy", "top1_accuracy"]:
            values = [r.scores.get(m, 0.0) for r in cat_results if m in r.scores]
            if values:
                metrics[f"{m}_category_{cat}"] = sum(values) / len(values)

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
        dataset="headqa",
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
    """Run HeadQA validation standalone."""
    import argparse

    parser = argparse.ArgumentParser(description="HeadQA Validation")
    parser.add_argument("--max-cases", type=int, default=10, help="Total cases to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--include-drugs", action="store_true", help="Include drug interaction check")
    parser.add_argument("--no-mcq", action="store_true", help="Disable MCQ answer selection step")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between cases (seconds)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument(
        "--categories", nargs="+",
        help="Filter by category (e.g., medicine pharmacology)",
        choices=HEADQA_CATEGORIES,
    )
    args = parser.parse_args()

    print("HeadQA Validation Harness")
    print("=" * 40)
    if args.categories:
        print(f"  Categories: {', '.join(args.categories)}")
    else:
        print(f"  Categories: all ({', '.join(HEADQA_CATEGORIES)})")

    cases = await fetch_headqa(
        max_cases=args.max_cases,
        seed=args.seed,
        categories=args.categories,
    )
    summary = await validate_headqa(
        cases,
        include_drug_check=args.include_drugs,
        include_mcq=not args.no_mcq,
        delay_between_cases=args.delay,
        resume=args.resume,
    )

    print_summary(summary)
    path = save_results(summary)
    print(f"Results saved to: {path}")


if __name__ == "__main__":
    asyncio.run(main())
