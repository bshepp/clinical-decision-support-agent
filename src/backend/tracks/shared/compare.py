# [Shared: Track Utilities]
"""
Cross-Track Comparison — loads results from all tracks and produces
comparative tables and charts.

Usage:
    python -m tracks.shared.compare --tracks A B C D --dataset medqa
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Ensure imports work
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# ──────────────────────────────────────────────
# Result loading
# ──────────────────────────────────────────────

TRACK_DIRS = {
    "A": BACKEND_DIR / "validation" / "results",
    "B": BACKEND_DIR / "tracks" / "rag_variants" / "results",
    "C": BACKEND_DIR / "tracks" / "iterative" / "results",
    "D": BACKEND_DIR / "tracks" / "arbitrated" / "results",
}


def load_latest_result(track_id: str, dataset: str = "medqa") -> Optional[dict]:
    """Load the most recent result file for a track + dataset."""
    result_dir = TRACK_DIRS.get(track_id)
    if not result_dir or not result_dir.exists():
        return None

    # Find matching files, sorted by name (timestamp suffix) descending
    pattern = f"*{dataset}*.json"
    if track_id != "A":
        pattern = f"track{track_id}_{dataset}*.json"

    files = sorted(result_dir.glob(pattern), reverse=True)
    if not files:
        return None

    with open(files[0], "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_results(dataset: str = "medqa") -> Dict[str, Optional[dict]]:
    """Load latest results for all tracks."""
    return {tid: load_latest_result(tid, dataset) for tid in TRACK_DIRS}


# ──────────────────────────────────────────────
# Comparison table
# ──────────────────────────────────────────────

def compare_tracks(dataset: str = "medqa") -> str:
    """
    Generate a comparison table across all tracks.

    Returns a formatted text table suitable for console or markdown.
    """
    results = load_all_results(dataset)

    header = f"{'Track':<22} {'Top-1':>7} {'Top-3':>7} {'Mentioned':>10} {'Pipeline':>9} {'Cost':>10}"
    sep = "-" * len(header)
    lines = [f"\nCross-Track Comparison: {dataset.upper()}", sep, header, sep]

    for tid, data in results.items():
        name = {
            "A": "A: Baseline",
            "B": "B: RAG Variants",
            "C": "C: Iterative",
            "D": "D: Arbitrated",
        }.get(tid, tid)

        if data is None:
            lines.append(f"{name:<22} {'--':>7} {'--':>7} {'--':>10} {'--':>9} {'--':>10}")
            continue

        metrics = data.get("metrics", data.get("summary", {}).get("metrics", {}))
        top1 = metrics.get("top1_accuracy", -1)
        top3 = metrics.get("top3_accuracy", -1)
        mentioned = metrics.get("mentioned_accuracy", -1)
        pipeline = metrics.get("parse_success", metrics.get("pipeline_success", -1))
        cost = data.get("total_cost_usd", data.get("cost", {}).get("total_cost_usd", -1))

        def fmt(v: float) -> str:
            return f"{v:.1%}" if v >= 0 else "--"

        cost_str = f"${cost:.4f}" if cost >= 0 else "--"
        lines.append(
            f"{name:<22} {fmt(top1):>7} {fmt(top3):>7} {fmt(mentioned):>10} "
            f"{fmt(pipeline):>9} {cost_str:>10}"
        )

    lines.append(sep)
    return "\n".join(lines)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare results across experimental tracks")
    parser.add_argument("--dataset", default="medqa", help="Dataset to compare (default: medqa)")
    parser.add_argument("--json", action="store_true", help="Output as JSON instead of table")
    args = parser.parse_args()

    if args.json:
        results = load_all_results(args.dataset)
        # Filter out None values for clean JSON
        clean = {k: v for k, v in results.items() if v is not None}
        print(json.dumps(clean, indent=2))
    else:
        print(compare_tracks(args.dataset))


if __name__ == "__main__":
    main()
