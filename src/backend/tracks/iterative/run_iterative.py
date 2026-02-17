# [Track C: Iterative Refinement]
"""
Track C — Run iterative refinement experiments against MedQA.

Usage:
    cd src/backend
    python -m tracks.iterative.run_iterative                    # all configs
    python -m tracks.iterative.run_iterative --config C1_3rounds  # single config
    python -m tracks.iterative.run_iterative --max-cases 10       # quick test

Each config runs the full baseline pipeline first, then feeds the initial
reasoning through N self-critique iterations. Results include per-iteration
accuracy AND cost, enabling cost/benefit charts.
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

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.agent.orchestrator import Orchestrator
from app.models.schemas import (
    CaseSubmission,
    CDSReport,
    AgentStepStatus,
    ClinicalReasoningResult,
)
from tracks.iterative.config import CONFIGS, IterativeConfig
from tracks.iterative.refiner import IterativeRefiner
from tracks.shared.cost_tracker import CostLedger
from validation.base import (
    ValidationCase,
    ValidationResult,
    ValidationSummary,
    diagnosis_in_differential,
    run_cds_pipeline,
    save_results,
    print_summary,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
MEDQA_PATH = BACKEND_DIR / "validation" / "data" / "medqa_test.jsonl"


# ──────────────────────────────────────────────
# Per-case runner
# ──────────────────────────────────────────────

async def run_case_iterative(
    case: ValidationCase,
    config: IterativeConfig,
    ledger: CostLedger,
) -> ValidationResult:
    """
    Run one case through:
      1. The baseline pipeline (Track A) to get initial reasoning
      2. The iterative refinement loop (Track C)
      3. Re-synthesize the final report with the refined differential
    """
    t0 = time.monotonic()

    # ── Step 1: Run baseline pipeline ──
    state, report, error = await run_cds_pipeline(
        patient_text=case.input_text,
        include_drug_check=True,
        include_guidelines=True,
    )

    if error or not state or not state.clinical_reasoning or not state.patient_profile:
        return ValidationResult(
            case_id=case.case_id,
            source_dataset=f"trackC_{config.config_id}",
            success=False,
            scores={},
            pipeline_time_ms=int((time.monotonic() - t0) * 1000),
            error=error or "Baseline pipeline failed to produce reasoning",
        )

    # ── Step 2: Iterative refinement ──
    refiner = IterativeRefiner(config, ledger)
    refined_reasoning, history = await refiner.refine(
        profile=state.patient_profile,
        initial_reasoning=state.clinical_reasoning,
    )

    # ── Step 3: Re-synthesize with the refined differential ──
    # Inject the refined reasoning back into the orchestrator state and
    # re-run just the synthesis step
    from app.tools.synthesis import SynthesisTool
    synth = SynthesisTool()
    try:
        refined_report = await synth.run(
            patient_profile=state.patient_profile,
            clinical_reasoning=refined_reasoning,
            drug_interactions=state.drug_interactions,
            guideline_retrieval=state.guideline_retrieval,
            conflict_detection=state.conflict_detection,
        )
    except Exception as e:
        refined_report = report  # Fall back to baseline report
        logger.warning(f"Re-synthesis failed, using baseline report: {e}")

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    # ── Score — compare baseline vs. refined ──
    scores: dict = {}
    details: dict = {"iterations": len(history) - 1}  # subtract the initial

    if "answer" in case.ground_truth:
        gt = case.ground_truth["answer"]

        # Score the baseline
        if report:
            b_found, b_rank, b_loc = diagnosis_in_differential(gt, report)
            scores["baseline_top1"] = 1.0 if (b_found and b_rank == 0) else 0.0
            scores["baseline_mentioned"] = 1.0 if b_found else 0.0

        # Score the refined report
        target_report = refined_report or report
        if target_report:
            r_found, r_rank, r_loc = diagnosis_in_differential(gt, target_report)
            scores["top1_accuracy"] = 1.0 if (r_found and r_rank == 0) else 0.0
            scores["top3_accuracy"] = 1.0 if (r_found and r_rank < 3) else 0.0
            scores["mentioned"] = 1.0 if r_found else 0.0
            details["rank"] = r_rank
            details["match_location"] = r_loc
            details["improved"] = scores.get("top1_accuracy", 0) > scores.get("baseline_top1", 0)

    # Per-iteration differential snapshots (for cost/benefit charts)
    details["per_iteration_top_dx"] = [
        h.differential_diagnosis[0].diagnosis if h.differential_diagnosis else "?"
        for h in history
    ]
    details["cost_ledger"] = ledger.to_dict()

    return ValidationResult(
        case_id=case.case_id,
        source_dataset=f"trackC_{config.config_id}",
        success=True,
        scores=scores,
        pipeline_time_ms=elapsed_ms,
        report_summary=(refined_report or report).patient_summary[:200] if (refined_report or report) else None,
        details=details,
    )


# ──────────────────────────────────────────────
# Experiment runner
# ──────────────────────────────────────────────

async def run_config(
    config: IterativeConfig,
    cases: List[ValidationCase],
) -> ValidationSummary:
    """Run all cases through the iterative config."""
    results: List[ValidationResult] = []
    start = time.monotonic()

    for i, case in enumerate(cases, 1):
        logger.info(f"  [{config.config_id}] case {i}/{len(cases)}: {case.case_id}")
        ledger = CostLedger(track_id=f"C_{config.config_id}")
        vr = await run_case_iterative(case, config, ledger)
        results.append(vr)

    elapsed = time.monotonic() - start
    successful = [r for r in results if r.success]

    metrics = {}
    for key in ("top1_accuracy", "top3_accuracy", "mentioned", "baseline_top1", "baseline_mentioned"):
        vals = [r.scores[key] for r in successful if key in r.scores]
        metrics[key] = sum(vals) / len(vals) if vals else 0.0
    metrics["pipeline_success"] = len(successful) / len(results) if results else 0.0

    # Average iterations used
    iters = [r.details.get("iterations", 0) for r in successful]
    metrics["avg_iterations"] = sum(iters) / len(iters) if iters else 0.0

    # Improvement rate
    improved = [r for r in successful if r.details.get("improved")]
    metrics["improvement_rate"] = len(improved) / len(successful) if successful else 0.0

    return ValidationSummary(
        dataset=f"trackC_{config.config_id}",
        total_cases=len(results),
        successful_cases=len(successful),
        failed_cases=len(results) - len(successful),
        metrics=metrics,
        per_case=results,
        run_duration_sec=round(elapsed, 1),
    )


# ──────────────────────────────────────────────
# Data loading (reuse from validation)
# ──────────────────────────────────────────────

def load_medqa_cases(max_cases: Optional[int] = None) -> List[ValidationCase]:
    if not MEDQA_PATH.exists():
        logger.error(f"MedQA data not found at {MEDQA_PATH}")
        return []
    cases: List[ValidationCase] = []
    with open(MEDQA_PATH, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if max_cases and len(cases) >= max_cases:
                break
            if not line.strip():
                continue
            data = json.loads(line)
            cases.append(ValidationCase(
                case_id=data.get("id", f"medqa_{ln}"),
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
    parser = argparse.ArgumentParser(description="Track C: Iterative refinement experiments")
    parser.add_argument("--config", type=str, default=None, help="Run a single config by ID")
    parser.add_argument("--max-cases", type=int, default=None, help="Limit cases per config")
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

    configs = CONFIGS
    if args.config:
        configs = [c for c in CONFIGS if c.config_id == args.config]
        if not configs:
            print(f"Unknown config: {args.config}")
            print(f"Available: {[c.config_id for c in CONFIGS]}")
            sys.exit(1)

    cases = load_medqa_cases(args.max_cases)
    if not cases:
        print("No MedQA cases loaded")
        sys.exit(1)
    print(f"Loaded {len(cases)} MedQA cases\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"  Running config: {cfg.config_id}")
        print(f"  {cfg.description}")
        print(f"  Max iterations: {cfg.max_iterations}, convergence: {cfg.convergence_threshold}")
        print(f"{'='*60}")

        summary = await run_config(cfg, cases)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        fname = f"trackC_{cfg.config_id}_{ts}.json"
        path = RESULTS_DIR / fname
        # Use validation save then move
        save_path = save_results(summary, filename=fname)
        if save_path != path:
            import shutil
            shutil.move(str(save_path), str(path))

        if not args.quiet:
            print_summary(summary)


if __name__ == "__main__":
    asyncio.run(main())
