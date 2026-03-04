"""
PubMedQA dataset fetcher and validation harness.

Downloads PubMedQA questions (biomedical yes/no/maybe questions derived from
PubMed abstracts) and evaluates the CDS pipeline's reasoning accuracy.

Source: https://huggingface.co/datasets/qiaojin/PubMedQA
Format: JSON with {question, context (list of strings), long_answer, final_decision}

Fields:
  - question: the biomedical question
  - context.contexts: list of context sentences from the abstract
  - context.labels: section labels for each context sentence
  - long_answer: the full-text answer/explanation
  - final_decision: "yes", "no", or "maybe"

Metrics:
  - decision_accuracy: LLM arrives at correct yes/no/maybe decision
  - reasoning_quality: Long answer overlap with pipeline reasoning
  - mentioned_accuracy: Key concepts from answer mentioned in report
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
    ensure_data_dir,
    fuzzy_match,
    load_checkpoint,
    normalize_text,
    print_summary,
    run_cds_pipeline,
    save_incremental,
    save_results,
)
from app.services.medgemma import MedGemmaService

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Data fetching
# ──────────────────────────────────────────────

# PubMedQA labeled subset (pqa_labeled) via HF datasets-server
PUBMEDQA_HF_URL = "https://datasets-server.huggingface.co/rows?dataset=qiaojin/PubMedQA&config=pqa_labeled&split=train&offset={offset}&length={length}"


async def fetch_pubmedqa(
    max_cases: int = 50,
    seed: int = 42,
) -> List[ValidationCase]:
    """
    Download PubMedQA labeled set and convert to ValidationCase objects.

    Uses the pqa_labeled config which has expert-annotated yes/no/maybe labels.

    Args:
        max_cases: Maximum number of cases to sample
        seed: Random seed for reproducible sampling
    """
    ensure_data_dir()
    cache_path = DATA_DIR / "pubmedqa_labeled.json"

    if cache_path.exists():
        print(f"  Loading PubMedQA from cache: {cache_path}")
        raw_cases = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        print(f"  Downloading PubMedQA labeled set...")
        raw_cases = await _download_pubmedqa(cache_path)

    if not raw_cases:
        raise RuntimeError("Failed to fetch PubMedQA data. Check network connection.")

    # Sample
    random.seed(seed)
    if len(raw_cases) > max_cases:
        raw_cases = random.sample(raw_cases, max_cases)

    # Convert to ValidationCase
    cases = []
    for i, item in enumerate(raw_cases):
        question = item.get("question", "")
        pubid = item.get("pubid", str(i))

        # Build context from abstract sections
        context_data = item.get("context", {})
        if isinstance(context_data, dict):
            contexts = context_data.get("contexts", [])
            labels = context_data.get("labels", [])
        elif isinstance(context_data, list):
            contexts = context_data
            labels = []
        else:
            contexts = []
            labels = []

        # Compose clinical vignette from abstract context + question
        context_text = ""
        if contexts:
            for j, ctx in enumerate(contexts):
                label = labels[j] if j < len(labels) else ""
                if label:
                    context_text += f"{label}: {ctx}\n"
                else:
                    context_text += f"{ctx}\n"

        input_text = f"{context_text.strip()}\n\nClinical Question: {question}"

        long_answer = item.get("long_answer", "")
        final_decision = item.get("final_decision", "").lower().strip()

        case_obj = ValidationCase(
            case_id=f"pubmedqa_{pubid}",
            source_dataset="pubmedqa",
            input_text=input_text,
            ground_truth={
                "final_decision": final_decision,  # yes/no/maybe
                "long_answer": long_answer,
                "question": question,
            },
            metadata={
                "pubid": pubid,
                "question": question,
                "context_text": context_text.strip(),
                "num_context_sentences": len(contexts),
            },
        )
        cases.append(case_obj)

    print(f"  Loaded {len(cases)} PubMedQA cases")
    return cases


async def _download_pubmedqa(cache_path: Path) -> List[dict]:
    """Download PubMedQA labeled set from HuggingFace."""
    all_rows = []

    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        offset = 0
        batch_size = 100
        max_fetch = 1000  # pqa_labeled has ~1000 examples

        while offset < max_fetch:
            url = PUBMEDQA_HF_URL.format(offset=offset, length=batch_size)
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
                if offset % 500 == 0:
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
            print(f"  Cached {len(all_rows)} PubMedQA cases to {cache_path}")

    return all_rows


# ──────────────────────────────────────────────
# Decision extraction via LLM
# ──────────────────────────────────────────────

DECISION_PROMPT = """You are a medical expert reviewing a clinical decision support report to answer a biomedical research question.

=== CDS REPORT ===
{report_summary}

=== QUESTION ===
{question}

Based on the evidence in the CDS report, answer the question with one of: YES, NO, or MAYBE.
Respond with ONLY the decision word on the first line (YES, NO, or MAYBE), then a one-sentence justification.
"""


async def extract_decision(
    case: ValidationCase,
    report,
) -> tuple:
    """
    Use MedGemma to extract a yes/no/maybe decision from the CDS report.

    Returns:
        (decision, justification) e.g. ("yes", "The evidence supports...")
    """
    parts = []
    if report.patient_summary:
        parts.append(f"Summary: {report.patient_summary}")
    if report.differential_diagnosis:
        dx_list = ", ".join(d.diagnosis for d in report.differential_diagnosis[:5])
        parts.append(f"Differential: {dx_list}")
    if report.guideline_recommendations:
        recs = "; ".join(report.guideline_recommendations[:5])
        parts.append(f"Recommendations: {recs}")
    if report.suggested_next_steps:
        steps = "; ".join(a.action for a in report.suggested_next_steps[:5])
        parts.append(f"Next Steps: {steps}")
    report_summary = "\n".join(parts) if parts else "No report available."

    question = case.ground_truth.get("question", case.input_text[:200])

    prompt = DECISION_PROMPT.format(
        report_summary=report_summary,
        question=question,
    )

    service = MedGemmaService()
    response = await service.generate(
        prompt=prompt,
        max_tokens=100,
        temperature=0.1,
    )

    lines = response.strip().split("\n")
    decision_raw = lines[0].strip().lower().rstrip(".")

    # Normalize to yes/no/maybe
    if "yes" in decision_raw:
        decision = "yes"
    elif "no" in decision_raw and "maybe" not in decision_raw:
        decision = "no"
    elif "maybe" in decision_raw or "uncertain" in decision_raw:
        decision = "maybe"
    else:
        decision = "unknown"

    justification = " ".join(lines[1:]).strip() if len(lines) > 1 else ""
    return decision, justification


# ──────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────

def _score_pubmedqa_case(
    case: ValidationCase,
    report,
    decision: str,
    state=None,
) -> dict:
    """
    Score a PubMedQA case.

    Metrics:
      - decision_accuracy: 1.0 if decision matches final_decision
      - reasoning_overlap: Token overlap between long_answer and report
      - mentioned_accuracy: Key terms from long_answer found in report
      - parse_success: 1.0 (always, since we got a report)
    """
    gt_decision = case.ground_truth.get("final_decision", "").lower()
    long_answer = case.ground_truth.get("long_answer", "")

    scores = {}

    # Decision accuracy
    scores["decision_accuracy"] = 1.0 if decision == gt_decision else 0.0

    # Reasoning overlap: how much of the long_answer is reflected in the report
    if long_answer and report:
        report_text = _build_report_text(report)
        long_tokens = set(normalize_text(long_answer).split())
        report_tokens = set(normalize_text(report_text).split())

        # Remove stopwords
        stopwords = {"the", "a", "an", "of", "in", "to", "and", "or", "is", "are",
                     "was", "were", "be", "been", "with", "for", "on", "at", "by",
                     "from", "this", "that", "it", "its", "has", "have", "had"}
        long_content = long_tokens - stopwords
        report_content = report_tokens - stopwords

        if long_content:
            overlap = len(long_content & report_content) / len(long_content)
            scores["reasoning_overlap"] = min(overlap, 1.0)
        else:
            scores["reasoning_overlap"] = 0.0
    else:
        scores["reasoning_overlap"] = 0.0

    # Mentioned accuracy: extract key medical terms from long answer
    if long_answer and report:
        report_text = _build_report_text(report)
        # Check if key phrases fuzzy-match
        if fuzzy_match(report_text, long_answer, threshold=0.3):
            scores["mentioned_accuracy"] = 1.0
        else:
            scores["mentioned_accuracy"] = 0.0
    else:
        scores["mentioned_accuracy"] = 0.0

    scores["parse_success"] = 1.0

    return scores


def _build_report_text(report) -> str:
    """Concatenate all report fields into a single searchable string."""
    parts = [
        report.patient_summary or "",
        " ".join(report.guideline_recommendations),
        " ".join(a.action for a in report.suggested_next_steps),
        " ".join(dx.diagnosis + " " + dx.reasoning for dx in report.differential_diagnosis),
    ]
    if report.conflicts:
        parts.append(" ".join(c.description for c in report.conflicts))
    return " ".join(parts)


# ──────────────────────────────────────────────
# Validation harness
# ──────────────────────────────────────────────

async def validate_pubmedqa(
    cases: List[ValidationCase],
    include_drug_check: bool = False,
    include_guidelines: bool = True,
    delay_between_cases: float = 2.0,
    resume: bool = False,
) -> ValidationSummary:
    """
    Run PubMedQA cases through the CDS pipeline and score results.

    Args:
        cases: List of PubMedQA ValidationCases
        include_drug_check: Whether to run drug interaction check
        include_guidelines: Whether to include guideline retrieval
        delay_between_cases: Seconds to wait between cases (rate limiting)
        resume: If True, skip cases already in checkpoint and continue
    """
    results: List[ValidationResult] = []
    start_time = time.time()

    completed_ids: set = set()
    if resume:
        prior = load_checkpoint("pubmedqa")
        if prior:
            results.extend(prior)
            completed_ids = {r.case_id for r in prior}
            print(f"  Resuming: {len(prior)} cases from checkpoint, {len(cases) - len(completed_ids)} remaining")
    else:
        clear_checkpoint("pubmedqa")

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
        gt_decision = case.ground_truth.get("final_decision", "")

        if report:
            # Extract decision
            try:
                decision, justification = await extract_decision(case, report)
            except Exception as e:
                logger.warning(f"Decision extraction failed for {case.case_id}: {e}")
                decision = "unknown"
                justification = f"Error: {e}"

            scores = _score_pubmedqa_case(case, report, decision, state)

            details.update({
                "gt_decision": gt_decision,
                "predicted_decision": decision,
                "decision_justification": justification,
                "long_answer": case.ground_truth.get("long_answer", "")[:300],
                "question": case.ground_truth.get("question", ""),
                "patient_summary": report.patient_summary[:300] if report.patient_summary else "",
                "num_diagnoses": len(report.differential_diagnosis),
            })

            correct = "Y" if scores.get("decision_accuracy", 0) > 0 else "N"
            overlap = scores.get("reasoning_overlap", 0)
            print(f"{'+'if correct=='Y' else '-'} decision={decision}(gt={gt_decision}) correct={correct} overlap={overlap:.0%} ({elapsed_ms}ms)")
        else:
            scores = {
                "decision_accuracy": 0.0,
                "reasoning_overlap": 0.0,
                "mentioned_accuracy": 0.0,
                "parse_success": 0.0,
            }
            details = {
                "gt_decision": gt_decision,
                "error": error,
                "question": case.ground_truth.get("question", ""),
            }
            print(f"- FAILED: {error[:80] if error else 'unknown'}")

        result = ValidationResult(
            case_id=case.case_id,
            source_dataset="pubmedqa",
            success=report is not None,
            scores=scores,
            pipeline_time_ms=elapsed_ms,
            step_results=step_results,
            report_summary=report.patient_summary[:200] if report else None,
            error=error,
            details=details,
        )
        results.append(result)
        save_incremental(result, "pubmedqa")

        if i < len(cases) - 1:
            await asyncio.sleep(delay_between_cases)

    # Aggregate
    total = len(results)
    successful = sum(1 for r in results if r.success)

    metric_names = [
        "decision_accuracy", "reasoning_overlap",
        "mentioned_accuracy", "parse_success",
    ]
    metrics = {}
    for m in metric_names:
        values = [r.scores.get(m, 0.0) for r in results if m in r.scores]
        metrics[m] = sum(values) / len(values) if values else 0.0

    times = [r.pipeline_time_ms for r in results if r.success]
    metrics["avg_pipeline_time_ms"] = sum(times) / len(times) if times else 0

    # Per-decision breakdown
    for decision_val in ["yes", "no", "maybe"]:
        subset = [r for r in results if r.details.get("gt_decision") == decision_val]
        if subset:
            metrics[f"count_{decision_val}"] = len(subset)
            acc = [r.scores.get("decision_accuracy", 0.0) for r in subset]
            metrics[f"decision_accuracy_{decision_val}"] = sum(acc) / len(acc)

    summary = ValidationSummary(
        dataset="pubmedqa",
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
    """Run PubMedQA validation standalone."""
    import argparse

    parser = argparse.ArgumentParser(description="PubMedQA Validation")
    parser.add_argument("--max-cases", type=int, default=10, help="Number of cases to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--include-drugs", action="store_true", help="Include drug interaction check")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between cases (seconds)")
    args = parser.parse_args()

    print("PubMedQA Validation Harness")
    print("=" * 40)

    cases = await fetch_pubmedqa(max_cases=args.max_cases, seed=args.seed)
    summary = await validate_pubmedqa(
        cases,
        include_drug_check=args.include_drugs,
        delay_between_cases=args.delay,
    )

    print_summary(summary)
    path = save_results(summary)
    print(f"Results saved to: {path}")


if __name__ == "__main__":
    asyncio.run(main())
