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
import random
import time
from pathlib import Path
from typing import List, Optional

import httpx

from validation.base import (
    DATA_DIR,
    ValidationCase,
    ValidationResult,
    ValidationSummary,
    diagnosis_in_differential,
    ensure_data_dir,
    fuzzy_match,
    normalize_text,
    print_summary,
    run_cds_pipeline,
    save_results,
)


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

        # Build clinical vignette (question only, not the options)
        # This simulates what a clinician would present
        clinical_text = _extract_vignette(question)

        cases.append(ValidationCase(
            case_id=f"medqa_{i:04d}",
            source_dataset="medqa",
            input_text=clinical_text,
            ground_truth={
                "correct_answer": answer_text,
                "answer_idx": answer_idx,
                "options": options,
                "full_question": question,
            },
        ))

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


def _extract_vignette(question: str) -> str:
    """
    Extract the clinical vignette from a USMLE question.

    USMLE questions typically end with "Which of the following..." or
    "What is the most likely diagnosis?". We strip the question stem
    to leave just the clinical narrative.
    """
    # Common question stems
    stems = [
        r"which of the following",
        r"what is the most likely",
        r"what is the best next step",
        r"what is the most appropriate",
        r"what is the diagnosis",
        r"the most likely diagnosis is",
        r"this patient most likely has",
        r"what would be the next step",
    ]

    text = question.strip()
    for stem in stems:
        import re
        # Find the last sentence that starts a question
        pattern = re.compile(rf'\.?\s*[A-Z].*{stem}.*[\?\.]?\s*$', re.IGNORECASE)
        match = pattern.search(text)
        if match:
            # Return everything before the question stem sentence
            vignette = text[:match.start()].strip()
            if len(vignette) > 50:  # Sanity check
                return vignette

    # Fallback: return the full text
    return text


# ──────────────────────────────────────────────
# Validation harness
# ──────────────────────────────────────────────

async def validate_medqa(
    cases: List[ValidationCase],
    include_drug_check: bool = False,
    include_guidelines: bool = True,
    delay_between_cases: float = 2.0,
) -> ValidationSummary:
    """
    Run MedQA cases through the CDS pipeline and score results.

    Args:
        cases: List of MedQA ValidationCases
        include_drug_check: Whether to run drug interaction check (slower)
        include_guidelines: Whether to include guideline retrieval
        delay_between_cases: Seconds to wait between cases (rate limiting)
    """
    results: List[ValidationResult] = []
    start_time = time.time()

    for i, case in enumerate(cases):
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

        # Score
        scores = {}
        details = {}
        correct_answer = case.ground_truth["correct_answer"]

        if report:
            # Top-1 accuracy
            found_top1, rank = diagnosis_in_differential(correct_answer, report, top_n=1)
            scores["top1_accuracy"] = 1.0 if found_top1 else 0.0

            # Top-3 accuracy
            found_top3, rank3 = diagnosis_in_differential(correct_answer, report, top_n=3)
            scores["top3_accuracy"] = 1.0 if found_top3 else 0.0

            # Mentioned anywhere
            found_any, rank_any = diagnosis_in_differential(correct_answer, report)
            scores["mentioned_accuracy"] = 1.0 if found_any else 0.0

            # Parse success
            scores["parse_success"] = 1.0

            details = {
                "correct_answer": correct_answer,
                "top_diagnosis": report.differential_diagnosis[0].diagnosis if report.differential_diagnosis else "NONE",
                "num_diagnoses": len(report.differential_diagnosis),
                "found_at_rank": rank_any if found_any else -1,
            }

            status_icon = "✓" if found_top3 else "✗"
            print(f"{status_icon} top1={'Y' if found_top1 else 'N'} top3={'Y' if found_top3 else 'N'} ({elapsed_ms}ms)")
        else:
            scores = {
                "top1_accuracy": 0.0,
                "top3_accuracy": 0.0,
                "mentioned_accuracy": 0.0,
                "parse_success": 0.0,
            }
            details = {"correct_answer": correct_answer, "error": error}
            print(f"✗ FAILED: {error[:80] if error else 'unknown'}")

        results.append(ValidationResult(
            case_id=case.case_id,
            source_dataset="medqa",
            success=report is not None,
            scores=scores,
            pipeline_time_ms=elapsed_ms,
            step_results=step_results,
            report_summary=report.patient_summary[:200] if report else None,
            error=error,
            details=details,
        ))

        # Rate limit
        if i < len(cases) - 1:
            await asyncio.sleep(delay_between_cases)

    # Aggregate
    total = len(results)
    successful = sum(1 for r in results if r.success)

    # Average each metric across successful cases only
    metric_names = ["top1_accuracy", "top3_accuracy", "mentioned_accuracy", "parse_success"]
    metrics = {}
    for m in metric_names:
        values = [r.scores.get(m, 0.0) for r in results]
        metrics[m] = sum(values) / len(values) if values else 0.0

    # Average pipeline time
    times = [r.pipeline_time_ms for r in results if r.success]
    metrics["avg_pipeline_time_ms"] = sum(times) / len(times) if times else 0

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
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between cases (seconds)")
    args = parser.parse_args()

    print("MedQA Validation Harness")
    print("=" * 40)

    cases = await fetch_medqa(max_cases=args.max_cases, seed=args.seed)
    summary = await validate_medqa(
        cases,
        include_drug_check=args.include_drugs,
        delay_between_cases=args.delay,
    )

    print_summary(summary)
    path = save_results(summary)
    print(f"Results saved to: {path}")


if __name__ == "__main__":
    asyncio.run(main())
