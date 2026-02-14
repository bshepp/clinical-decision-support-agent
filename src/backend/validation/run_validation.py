"""
Unified validation runner for the Clinical Decision Support Agent.

Runs all three dataset validations (MedQA, MTSamples, PMC Case Reports)
and produces a combined summary report.

Usage:
    # From src/backend directory:
    python -m validation.run_validation --all --max-cases 10
    python -m validation.run_validation --medqa --max-cases 20
    python -m validation.run_validation --mtsamples --max-cases 15
    python -m validation.run_validation --pmc --max-cases 10

    # Fetch data only (no pipeline execution):
    python -m validation.run_validation --fetch-only
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure backend is importable
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from validation.base import (
    ValidationSummary,
    print_summary,
    save_results,
)
from validation.harness_medqa import fetch_medqa, validate_medqa
from validation.harness_mtsamples import fetch_mtsamples, validate_mtsamples
from validation.harness_pmc import fetch_pmc_cases, validate_pmc


async def run_all_validations(
    run_medqa: bool = True,
    run_mtsamples: bool = True,
    run_pmc: bool = True,
    max_cases: int = 10,
    seed: int = 42,
    include_drug_check: bool = True,
    include_guidelines: bool = True,
    delay: float = 2.0,
    fetch_only: bool = False,
) -> dict:
    """
    Run validation against selected datasets.

    Returns dict of {dataset_name: ValidationSummary}
    """
    results = {}
    start = time.time()

    # ── MedQA ──
    if run_medqa:
        print("\n" + "=" * 60)
        print("  DATASET 1: MedQA (USMLE-style diagnostic accuracy)")
        print("=" * 60)

        cases = await fetch_medqa(max_cases=max_cases, seed=seed)

        if fetch_only:
            print(f"  Fetched {len(cases)} MedQA cases (fetch-only mode)")
        else:
            summary = await validate_medqa(
                cases,
                include_drug_check=include_drug_check,
                include_guidelines=include_guidelines,
                delay_between_cases=delay,
            )
            print_summary(summary)
            save_results(summary)
            results["medqa"] = summary

    # ── MTSamples ──
    if run_mtsamples:
        print("\n" + "=" * 60)
        print("  DATASET 2: MTSamples (clinical note parsing robustness)")
        print("=" * 60)

        cases = await fetch_mtsamples(max_cases=max_cases, seed=seed)

        if fetch_only:
            print(f"  Fetched {len(cases)} MTSamples cases (fetch-only mode)")
        else:
            summary = await validate_mtsamples(
                cases,
                include_drug_check=include_drug_check,
                include_guidelines=include_guidelines,
                delay_between_cases=delay,
            )
            print_summary(summary)
            save_results(summary)
            results["mtsamples"] = summary

    # ── PMC Case Reports ──
    if run_pmc:
        print("\n" + "=" * 60)
        print("  DATASET 3: PMC Case Reports (real-world diagnostic accuracy)")
        print("=" * 60)

        cases = await fetch_pmc_cases(max_cases=max_cases, seed=seed)

        if fetch_only:
            print(f"  Fetched {len(cases)} PMC cases (fetch-only mode)")
        else:
            summary = await validate_pmc(
                cases,
                include_drug_check=include_drug_check,
                include_guidelines=include_guidelines,
                delay_between_cases=delay,
            )
            print_summary(summary)
            save_results(summary)
            results["pmc"] = summary

    # ── Combined Summary ──
    total_duration = time.time() - start

    if results and not fetch_only:
        _print_combined_summary(results, total_duration)
        _save_combined_report(results, total_duration)

    return results


def _print_combined_summary(results: dict, total_duration: float):
    """Print a combined summary across all datasets."""
    print("\n" + "=" * 70)
    print("  COMBINED VALIDATION REPORT")
    print("=" * 70)

    # Header
    print(f"\n  {'Dataset':<15} {'Cases':>6} {'Success':>8} {'Key Metric':>25} {'Value':>8}")
    print(f"  {'-'*15} {'-'*6} {'-'*8} {'-'*25} {'-'*8}")

    for name, summary in results.items():
        # Pick the most important metric for each dataset
        if name == "medqa":
            key_metric = "top3_accuracy"
        elif name == "mtsamples":
            key_metric = "parse_success"
        elif name == "pmc":
            key_metric = "diagnostic_accuracy"
        else:
            key_metric = list(summary.metrics.keys())[0] if summary.metrics else "N/A"

        value = summary.metrics.get(key_metric, 0.0)
        print(
            f"  {name:<15} {summary.total_cases:>6} "
            f"{summary.successful_cases:>8} "
            f"{key_metric:>25} {value:>7.1%}"
        )

    # All metrics
    print(f"\n  {'─' * 66}")
    for name, summary in results.items():
        print(f"\n  {name.upper()} metrics:")
        for metric, value in sorted(summary.metrics.items()):
            if "time" in metric and isinstance(value, (int, float)):
                print(f"    {metric:<35} {value:.0f}ms")
            elif isinstance(value, float):
                print(f"    {metric:<35} {value:.1%}")

    # Totals
    total_cases = sum(s.total_cases for s in results.values())
    total_success = sum(s.successful_cases for s in results.values())
    print(f"\n  Total cases:     {total_cases}")
    print(f"  Total success:   {total_success}")
    print(f"  Total duration:  {total_duration:.1f}s ({total_duration/60:.1f}min)")
    print(f"  Timestamp:       {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)


def _save_combined_report(results: dict, total_duration: float):
    """Save combined report to JSON."""
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"combined_{ts}.json"

    combined = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_duration_sec": total_duration,
        "datasets": {},
    }

    for name, summary in results.items():
        combined["datasets"][name] = {
            "total_cases": summary.total_cases,
            "successful_cases": summary.successful_cases,
            "failed_cases": summary.failed_cases,
            "metrics": summary.metrics,
            "run_duration_sec": summary.run_duration_sec,
        }

    path.write_text(json.dumps(combined, indent=2, default=str))
    print(f"\n  Combined report saved to: {path}")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="CDS Agent Validation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m validation.run_validation --all --max-cases 10
  python -m validation.run_validation --medqa --max-cases 50
  python -m validation.run_validation --fetch-only
  python -m validation.run_validation --medqa --pmc --max-cases 20 --no-drugs
        """,
    )

    # Dataset selection
    data_group = parser.add_argument_group("Datasets")
    data_group.add_argument("--all", action="store_true", help="Run all three datasets")
    data_group.add_argument("--medqa", action="store_true", help="Run MedQA validation")
    data_group.add_argument("--mtsamples", action="store_true", help="Run MTSamples validation")
    data_group.add_argument("--pmc", action="store_true", help="Run PMC Case Reports validation")

    # Configuration
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument("--max-cases", type=int, default=10, help="Cases per dataset (default: 10)")
    config_group.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    config_group.add_argument("--delay", type=float, default=2.0, help="Delay between cases in seconds (default: 2.0)")
    config_group.add_argument("--no-drugs", action="store_true", help="Skip drug interaction checks")
    config_group.add_argument("--no-guidelines", action="store_true", help="Skip guideline retrieval")
    config_group.add_argument("--fetch-only", action="store_true", help="Only download data, don't run pipeline")

    args = parser.parse_args()

    # Default to --all if nothing specified
    if not any([args.all, args.medqa, args.mtsamples, args.pmc]):
        args.all = True

    run_medqa = args.all or args.medqa
    run_mtsamples = args.all or args.mtsamples
    run_pmc = args.all or args.pmc

    print("╔════════════════════════════════════════════════════════╗")
    print("║   Clinical Decision Support Agent — Validation Suite  ║")
    print("╚════════════════════════════════════════════════════════╝")
    print(f"\n  Datasets:     {'MedQA ' if run_medqa else ''}{'MTSamples ' if run_mtsamples else ''}{'PMC ' if run_pmc else ''}")
    print(f"  Cases/dataset: {args.max_cases}")
    print(f"  Drug check:    {'Yes' if not args.no_drugs else 'No'}")
    print(f"  Guidelines:    {'Yes' if not args.no_guidelines else 'No'}")
    print(f"  Fetch only:    {'Yes' if args.fetch_only else 'No'}")

    asyncio.run(run_all_validations(
        run_medqa=run_medqa,
        run_mtsamples=run_mtsamples,
        run_pmc=run_pmc,
        max_cases=args.max_cases,
        seed=args.seed,
        include_drug_check=not args.no_drugs,
        include_guidelines=not args.no_guidelines,
        delay=args.delay,
        fetch_only=args.fetch_only,
    ))


if __name__ == "__main__":
    main()
