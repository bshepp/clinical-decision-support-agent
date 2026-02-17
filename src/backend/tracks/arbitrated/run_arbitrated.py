# [Track D: Arbitrated Parallel]
"""
Track D — Run parallel-specialist + arbiter experiments against MedQA.

Usage:
    cd src/backend
    python -m tracks.arbitrated.run_arbitrated                      # all configs
    python -m tracks.arbitrated.run_arbitrated --config D0_3spec_1round
    python -m tracks.arbitrated.run_arbitrated --max-cases 10

Flow per case:
  1. Run baseline pipeline (Track A) to get patient profile + guidelines
  2. Run N specialists in parallel on the patient profile
  3. Arbiter merges specialist differentials into consensus
  4. (Optional round 2+) Arbiter sends tailored feedback → specialists re-reason → re-merge
  5. Re-synthesize final report with consensus differential
  6. Score against ground truth

Results include per-round accuracy AND cost for cost/benefit charts.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.models.schemas import (
    CaseSubmission,
    CDSReport,
    ClinicalReasoningResult,
    PatientProfile,
)
from tracks.arbitrated.config import CONFIGS, ArbitratedConfig, SpecialistDef
from tracks.arbitrated.specialists import SpecialistAgent, run_specialists_parallel
from tracks.arbitrated.arbiter import Arbiter
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

async def run_case_arbitrated(
    case: ValidationCase,
    config: ArbitratedConfig,
    ledger: CostLedger,
) -> ValidationResult:
    """
    Run one case through the arbitrated parallel pipeline.
    """
    t0 = time.monotonic()

    # ── Step 1: Baseline pipeline for patient profile + supporting data ──
    state, baseline_report, error = await run_cds_pipeline(
        patient_text=case.input_text,
        include_drug_check=True,
        include_guidelines=True,
    )

    if error or not state or not state.patient_profile:
        return ValidationResult(
            case_id=case.case_id,
            source_dataset=f"trackD_{config.config_id}",
            success=False,
            scores={},
            pipeline_time_ms=int((time.monotonic() - t0) * 1000),
            error=error or "Baseline pipeline failed",
        )

    profile = state.patient_profile

    # ── Step 2: Build specialist agents ──
    spec_defs = {s.specialist_id: s for s in config.specialists}
    agents = [
        SpecialistAgent(
            spec=s,
            temperature=config.specialist_temperature,
            max_tokens=config.max_tokens_specialist,
        )
        for s in config.specialists
    ]

    # ── Step 3: Multi-round specialist → arbiter loop ──
    arbiter = Arbiter(config)
    consensus: Optional[ClinicalReasoningResult] = None
    round_results: List[Dict[str, ClinicalReasoningResult]] = []

    for round_num in range(1, config.max_rounds + 1):
        # Run specialists (parallel)
        feedback = {}
        if round_num > 1 and consensus is not None:
            # Generate tailored feedback for disagreeing specialists
            try:
                feedback = await arbiter.generate_feedback(
                    profile=profile,
                    consensus=consensus,
                    specialist_results=round_results[-1],
                    specialist_defs=spec_defs,
                    ledger=ledger,
                    round_num=round_num - 1,
                )
            except (ValueError, Exception) as e:
                logger.warning(f"Arbiter feedback generation failed round {round_num}: {e}")
                feedback = {}

        specialist_results = await run_specialists_parallel(
            specialists=agents,
            profile=profile,
            ledger=ledger,
            iteration=round_num,
            arbiter_feedback=feedback,
        )
        round_results.append(specialist_results)

        if not specialist_results:
            logger.warning(f"All specialists failed in round {round_num}, stopping.")
            break

        # Arbiter merge
        try:
            consensus = await arbiter.merge(
                profile=profile,
                specialist_results=specialist_results,
                specialist_defs=spec_defs,
                ledger=ledger,
                round_num=round_num,
            )
        except (ValueError, Exception) as e:
            logger.warning(f"Arbiter merge failed round {round_num}: {e}")
            # If we had a previous consensus, keep it; otherwise use first specialist result
            if consensus is None and specialist_results:
                first_key = next(iter(specialist_results))
                consensus = specialist_results[first_key]
            break

    # ── Step 4: Re-synthesize with consensus differential ──
    from app.tools.synthesis import SynthesisTool
    synth = SynthesisTool()
    try:
        final_report = await synth.run(
            patient_profile=profile,
            clinical_reasoning=consensus,
            drug_interactions=state.drug_interactions,
            guideline_retrieval=state.guideline_retrieval,
            conflict_detection=state.conflict_detection,
        )
    except Exception as e:
        logger.warning(f"Re-synthesis failed: {e}")
        final_report = baseline_report

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    # ── Score ──
    scores: dict = {}
    details: dict = {
        "rounds": config.max_rounds,
        "specialists": [s.specialist_id for s in config.specialists],
    }

    if "answer" in case.ground_truth:
        gt = case.ground_truth["answer"]

        # Baseline score
        if baseline_report:
            b_found, b_rank, _ = diagnosis_in_differential(gt, baseline_report)
            scores["baseline_top1"] = 1.0 if (b_found and b_rank == 0) else 0.0
            scores["baseline_mentioned"] = 1.0 if b_found else 0.0

        # Arbitrated score
        target = final_report or baseline_report
        if target:
            r_found, r_rank, r_loc = diagnosis_in_differential(gt, target)
            scores["top1_accuracy"] = 1.0 if (r_found and r_rank == 0) else 0.0
            scores["top3_accuracy"] = 1.0 if (r_found and r_rank < 3) else 0.0
            scores["mentioned"] = 1.0 if r_found else 0.0
            details["rank"] = r_rank
            details["match_location"] = r_loc
            details["improved"] = scores.get("top1_accuracy", 0) > scores.get("baseline_top1", 0)

    details["cost_ledger"] = ledger.to_dict()

    # Per-round consensus top dx
    if consensus:
        details["consensus_top_dx"] = [
            dx.diagnosis for dx in consensus.differential_diagnosis[:3]
        ]

    return ValidationResult(
        case_id=case.case_id,
        source_dataset=f"trackD_{config.config_id}",
        success=True,
        scores=scores,
        pipeline_time_ms=elapsed_ms,
        report_summary=(final_report or baseline_report).patient_summary[:200] if (final_report or baseline_report) else None,
        details=details,
    )


# ──────────────────────────────────────────────
# Experiment runner
# ──────────────────────────────────────────────

async def run_config(
    config: ArbitratedConfig,
    cases: List[ValidationCase],
) -> ValidationSummary:
    results: List[ValidationResult] = []
    start = time.monotonic()

    for i, case in enumerate(cases, 1):
        logger.info(f"  [{config.config_id}] case {i}/{len(cases)}: {case.case_id}")
        ledger = CostLedger(track_id=f"D_{config.config_id}")
        vr = await run_case_arbitrated(case, config, ledger)
        results.append(vr)

    elapsed = time.monotonic() - start
    successful = [r for r in results if r.success]

    metrics = {}
    for key in ("top1_accuracy", "top3_accuracy", "mentioned", "baseline_top1", "baseline_mentioned"):
        vals = [r.scores[key] for r in successful if key in r.scores]
        metrics[key] = sum(vals) / len(vals) if vals else 0.0
    metrics["pipeline_success"] = len(successful) / len(results) if results else 0.0

    improved = [r for r in successful if r.details.get("improved")]
    metrics["improvement_rate"] = len(improved) / len(successful) if successful else 0.0

    return ValidationSummary(
        dataset=f"trackD_{config.config_id}",
        total_cases=len(results),
        successful_cases=len(successful),
        failed_cases=len(results) - len(successful),
        metrics=metrics,
        per_case=results,
        run_duration_sec=round(elapsed, 1),
    )


# ──────────────────────────────────────────────
# Data loading
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
    parser = argparse.ArgumentParser(description="Track D: Arbitrated parallel experiments")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
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
        print(f"  Specialists: {[s.specialist_id for s in cfg.specialists]}")
        print(f"  Max rounds: {cfg.max_rounds}")
        print(f"{'='*60}")

        summary = await run_config(cfg, cases)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        fname = f"trackD_{cfg.config_id}_{ts}.json"
        path = RESULTS_DIR / fname
        save_path = save_results(summary, filename=fname)
        if save_path != path:
            import shutil
            shutil.move(str(save_path), str(path))

        if not args.quiet:
            print_summary(summary)


if __name__ == "__main__":
    asyncio.run(main())
