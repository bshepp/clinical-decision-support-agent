# [Track B: RAG Variants]
"""
Track B — Run all RAG variant experiments against the MedQA dataset.

Usage:
    cd src/backend
    python -m tracks.rag_variants.run_variants              # run all variants
    python -m tracks.rag_variants.run_variants --variant B1_fixed256  # single variant
    python -m tracks.rag_variants.run_variants --max-cases 10         # quick smoke test

Each variant gets its own ChromaDB collection and result file.
Results are saved to  tracks/rag_variants/results/trackB_<variant_id>_<timestamp>.json
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# ── Ensure imports work ──
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.agent.orchestrator import Orchestrator
from app.models.schemas import CaseSubmission, CDSReport, AgentStepStatus
from tracks.rag_variants.config import VARIANTS, RAGVariant
from tracks.rag_variants.retriever import VariantRetriever
from tracks.shared.cost_tracker import CostLedger, record_call, estimate_tokens
from validation.base import (
    ValidationCase,
    ValidationResult,
    ValidationSummary,
    fuzzy_match,
    diagnosis_in_differential,
    save_results,
    print_summary,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
MEDQA_PATH = BACKEND_DIR / "validation" / "data" / "medqa_test.jsonl"


# ──────────────────────────────────────────────
# Variant runner
# ──────────────────────────────────────────────

async def run_variant(
    variant: RAGVariant,
    cases: List[ValidationCase],
    ledger: CostLedger,
) -> ValidationSummary:
    """
    Run the full CDS pipeline for each case, swapping in the variant retriever.

    The variant retriever replaces the default GuidelineRetrievalTool on the
    orchestrator, keeping everything else identical to Track A.
    """
    # Build the modified retriever
    retriever = VariantRetriever(variant)
    results: List[ValidationResult] = []
    start = time.monotonic()

    for i, case in enumerate(cases, 1):
        logger.info(f"  [{variant.variant_id}] case {i}/{len(cases)}: {case.case_id}")
        vr = await _run_single_case(case, retriever, variant, ledger)
        results.append(vr)

    elapsed = time.monotonic() - start

    # Aggregate metrics
    successful = [r for r in results if r.success]
    score_keys = set()
    for r in successful:
        score_keys.update(r.scores.keys())

    metrics = {}
    for key in score_keys:
        vals = [r.scores[key] for r in successful if key in r.scores]
        metrics[key] = sum(vals) / len(vals) if vals else 0.0
    metrics["pipeline_success"] = len(successful) / len(results) if results else 0.0

    summary = ValidationSummary(
        dataset=f"trackB_{variant.variant_id}",
        total_cases=len(results),
        successful_cases=len(successful),
        failed_cases=len(results) - len(successful),
        metrics=metrics,
        per_case=results,
        run_duration_sec=round(elapsed, 1),
    )
    return summary


async def _run_single_case(
    case: ValidationCase,
    retriever: VariantRetriever,
    variant: RAGVariant,
    ledger: CostLedger,
) -> ValidationResult:
    """Run one case through the pipeline with the variant retriever injected."""
    submission = CaseSubmission(
        patient_text=case.input_text,
        include_drug_check=True,
        include_guidelines=True,
    )
    orchestrator = Orchestrator()
    # Swap in Track B's retriever
    orchestrator.guideline_retrieval = retriever

    t0 = time.monotonic()
    error: Optional[str] = None
    report: Optional[CDSReport] = None

    try:
        async for _step in orchestrator.run(submission):
            pass
        report = orchestrator.get_result()
        if report is None and orchestrator.state:
            failed = [
                s for s in orchestrator.state.steps if s.status == AgentStepStatus.FAILED
            ]
            if failed:
                error = "; ".join(f"{s.step_id}: {s.error}" for s in failed)
            else:
                error = "Pipeline completed but produced no report"
    except Exception as e:
        error = str(e)

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    # Score against ground truth
    scores: dict = {}
    details: dict = {}
    if report and "answer" in case.ground_truth:
        gt_answer = case.ground_truth["answer"]
        found, rank, location = diagnosis_in_differential(gt_answer, report)
        scores["top1_accuracy"] = 1.0 if (found and rank == 0) else 0.0
        scores["top3_accuracy"] = 1.0 if (found and rank < 3) else 0.0
        scores["mentioned"] = 1.0 if found else 0.0
        details["rank"] = rank
        details["match_location"] = location

    step_results = {}
    if orchestrator.state:
        for s in orchestrator.state.steps:
            step_results[s.step_id] = s.status.value

    return ValidationResult(
        case_id=case.case_id,
        source_dataset=f"trackB_{variant.variant_id}",
        success=error is None,
        scores=scores,
        pipeline_time_ms=elapsed_ms,
        step_results=step_results,
        report_summary=report.patient_summary[:200] if report else None,
        error=error,
        details=details,
    )


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────

def load_medqa_cases(max_cases: Optional[int] = None) -> List[ValidationCase]:
    """Load MedQA test cases from the validation data directory."""
    if not MEDQA_PATH.exists():
        logger.error(f"MedQA data not found at {MEDQA_PATH}")
        return []

    cases: List[ValidationCase] = []
    with open(MEDQA_PATH, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if max_cases and len(cases) >= max_cases:
                break
            if not line.strip():
                continue
            data = json.loads(line)
            cases.append(ValidationCase(
                case_id=data.get("id", f"medqa_{line_num}"),
                source_dataset="medqa",
                input_text=data.get("question", data.get("input", "")),
                ground_truth={"answer": data.get("answer", data.get("target", ""))},
                metadata=data.get("metadata", {}),
            ))
    return cases


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Track B: RAG variant sweep")
    parser.add_argument("--variant", type=str, default=None, help="Run a single variant by ID")
    parser.add_argument("--max-cases", type=int, default=None, help="Limit cases per variant")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Wait for endpoint to be online (handles scale-to-zero)
    from tracks.shared.endpoint_check import wait_for_endpoint
    if not await wait_for_endpoint(quiet=args.quiet):
        print("ABORT: MedGemma endpoint is not reachable. Resume it and try again.")
        sys.exit(1)

    # Select variants
    variants = VARIANTS
    if args.variant:
        variants = [v for v in VARIANTS if v.variant_id == args.variant]
        if not variants:
            print(f"Unknown variant: {args.variant}")
            print(f"Available: {[v.variant_id for v in VARIANTS]}")
            sys.exit(1)

    cases = load_medqa_cases(args.max_cases)
    if not cases:
        print("No MedQA cases loaded — check data/medqa_test.jsonl")
        sys.exit(1)
    print(f"Loaded {len(cases)} MedQA cases\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_summaries = []

    for variant in variants:
        print(f"\n{'='*60}")
        print(f"  Running variant: {variant.variant_id}")
        print(f"  {variant.description}")
        print(f"{'='*60}")

        ledger = CostLedger(track_id=f"B_{variant.variant_id}")
        summary = await run_variant(variant, cases, ledger)
        all_summaries.append(summary)

        # Save per-variant results
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        fname = f"trackB_{variant.variant_id}_{ts}.json"
        save_path = save_results(summary, filename=fname)
        # Override save location to Track B results dir
        target = RESULTS_DIR / fname
        if save_path != target:
            import shutil
            shutil.move(str(save_path), str(target))

        if not args.quiet:
            print_summary(summary)

    # Print comparison table
    if len(all_summaries) > 1:
        print(f"\n{'='*60}")
        print("  CROSS-VARIANT COMPARISON")
        print(f"{'='*60}")
        header = f"{'Variant':<25} {'Top-1':>7} {'Top-3':>7} {'Mentioned':>10} {'Pipeline':>9}"
        print(header)
        print("-" * len(header))
        for s in all_summaries:
            m = s.metrics
            print(
                f"{s.dataset:<25} "
                f"{m.get('top1_accuracy', 0):.1%}   "
                f"{m.get('top3_accuracy', 0):.1%}   "
                f"{m.get('mentioned', 0):.1%}      "
                f"{m.get('pipeline_success', 0):.1%}"
            )


if __name__ == "__main__":
    asyncio.run(main())
