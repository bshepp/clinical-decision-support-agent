"""
RAG Retrieval Quality Test

Directly tests the ChromaDB vector search to validate that the expanded
clinical guidelines corpus returns relevant results for diverse clinical queries.

This bypasses the full agent pipeline and tests the retrieval layer in isolation.

Usage:
    cd src/backend
    python test_rag_quality.py                   # Run all queries
    python test_rag_quality.py --rebuild          # Delete ChromaDB and rebuild from scratch
    python test_rag_quality.py --stats            # Show collection statistics only
    python test_rag_quality.py --query "chest pain evaluation"  # Test a single query
"""
import asyncio
import argparse
import json
import shutil
import sys
import os
from pathlib import Path

# Ensure the app package is importable
sys.path.insert(0, str(Path(__file__).parent))

os.environ.setdefault("MEDGEMMA_API_KEY", "dummy")  # Prevent settings validation error


# ─────────────────────────────────────────────────
# Test Queries: (query, expected_specialty, expected_title_substring)
# Each query simulates what the orchestrator would send to the RAG tool.
# ─────────────────────────────────────────────────

RAG_TEST_QUERIES = [
    # Cardiology
    {
        "query": "Acute chest pain evaluation and troponin testing guidelines",
        "expected_specialties": ["Cardiology"],
        "expected_title_keywords": ["chest pain", "ACS", "STEMI"],
        "min_relevance": 0.3,
    },
    {
        "query": "Heart failure management with reduced ejection fraction HFrEF",
        "expected_specialties": ["Cardiology"],
        "expected_title_keywords": ["heart failure"],
        "min_relevance": 0.3,
    },
    {
        "query": "Atrial fibrillation anticoagulation and rate control",
        "expected_specialties": ["Cardiology"],
        "expected_title_keywords": ["atrial fibrillation", "AFib"],
        "min_relevance": 0.3,
    },
    {
        "query": "Management of acute pulmonary embolism with hemodynamic instability",
        "expected_specialties": ["Cardiology", "Pulmonology", "Hematology"],
        "expected_title_keywords": ["pulmonary embolism", "PE", "VTE"],
        "min_relevance": 0.25,
    },
    # Emergency Medicine
    {
        "query": "Acute ischemic stroke tPA thrombolysis eligibility criteria",
        "expected_specialties": ["Emergency Medicine", "Neurology"],
        "expected_title_keywords": ["stroke", "CVA"],
        "min_relevance": 0.3,
    },
    {
        "query": "Sepsis hour-1 bundle treatment with IV fluids and antibiotics",
        "expected_specialties": ["Emergency Medicine"],
        "expected_title_keywords": ["sepsis"],
        "min_relevance": 0.3,
    },
    {
        "query": "Anaphylaxis emergency treatment with epinephrine",
        "expected_specialties": ["Emergency Medicine"],
        "expected_title_keywords": ["anaphylaxis"],
        "min_relevance": 0.3,
    },
    {
        "query": "Trauma ATLS assessment primary survey hemorrhagic shock",
        "expected_specialties": ["Emergency Medicine"],
        "expected_title_keywords": ["trauma"],
        "min_relevance": 0.25,
    },
    {
        "query": "Seizure management status epilepticus benzodiazepine protocol",
        "expected_specialties": ["Emergency Medicine", "Neurology"],
        "expected_title_keywords": ["seizure"],
        "min_relevance": 0.25,
    },
    # Endocrinology
    {
        "query": "Diabetic ketoacidosis DKA insulin drip and fluid management",
        "expected_specialties": ["Endocrinology"],
        "expected_title_keywords": ["DKA", "diabetic ketoacidosis"],
        "min_relevance": 0.3,
    },
    {
        "query": "Type 2 diabetes management metformin A1C targets",
        "expected_specialties": ["Endocrinology"],
        "expected_title_keywords": ["diabetes", "DM"],
        "min_relevance": 0.3,
    },
    {
        "query": "Thyroid disease hyperthyroidism Graves disease treatment",
        "expected_specialties": ["Endocrinology"],
        "expected_title_keywords": ["thyroid"],
        "min_relevance": 0.3,
    },
    # Pulmonology
    {
        "query": "COPD exacerbation treatment bronchodilators steroids antibiotics",
        "expected_specialties": ["Pulmonology"],
        "expected_title_keywords": ["COPD"],
        "min_relevance": 0.3,
    },
    {
        "query": "Acute asthma exacerbation treatment albuterol magnesium",
        "expected_specialties": ["Pulmonology", "Pediatrics"],
        "expected_title_keywords": ["asthma"],
        "min_relevance": 0.3,
    },
    {
        "query": "Community acquired pneumonia antibiotic selection CURB-65",
        "expected_specialties": ["Pulmonology", "Infectious Disease"],
        "expected_title_keywords": ["pneumonia"],
        "min_relevance": 0.3,
    },
    # Gastroenterology
    {
        "query": "Upper GI bleeding management endoscopy PPI transfusion",
        "expected_specialties": ["Gastroenterology"],
        "expected_title_keywords": ["GI bleed", "upper GI"],
        "min_relevance": 0.3,
    },
    {
        "query": "Acute pancreatitis management fluid resuscitation pain control",
        "expected_specialties": ["Gastroenterology"],
        "expected_title_keywords": ["pancreatitis"],
        "min_relevance": 0.3,
    },
    # Neurology
    {
        "query": "Epilepsy seizure medication selection antiepileptic drugs",
        "expected_specialties": ["Neurology"],
        "expected_title_keywords": ["epilepsy"],
        "min_relevance": 0.3,
    },
    {
        "query": "Bacterial meningitis empiric antibiotics lumbar puncture",
        "expected_specialties": ["Neurology", "Infectious Disease"],
        "expected_title_keywords": ["meningitis"],
        "min_relevance": 0.3,
    },
    # Psychiatry
    {
        "query": "Suicide risk assessment safety planning lethal means counseling",
        "expected_specialties": ["Psychiatry"],
        "expected_title_keywords": ["suicide", "suicid"],
        "min_relevance": 0.3,
    },
    {
        "query": "Major depressive disorder SSRI treatment algorithm",
        "expected_specialties": ["Psychiatry"],
        "expected_title_keywords": ["depression", "depressive"],
        "min_relevance": 0.3,
    },
    # Pediatrics
    {
        "query": "Neonatal fever sepsis workup guidelines for infants under 60 days",
        "expected_specialties": ["Pediatrics"],
        "expected_title_keywords": ["fever", "neonate", "neonatal"],
        "min_relevance": 0.25,
    },
    {
        "query": "Pediatric dehydration oral rehydration IV fluid bolus",
        "expected_specialties": ["Pediatrics"],
        "expected_title_keywords": ["dehydration"],
        "min_relevance": 0.25,
    },
    # Nephrology
    {
        "query": "Hyperkalemia emergency management calcium insulin kayexalate dialysis",
        "expected_specialties": ["Nephrology", "Emergency Medicine"],
        "expected_title_keywords": ["hyperkalemia"],
        "min_relevance": 0.25,
    },
    {
        "query": "Acute kidney injury management and staging KDIGO",
        "expected_specialties": ["Nephrology"],
        "expected_title_keywords": ["AKI", "kidney injury"],
        "min_relevance": 0.25,
    },
    # Hematology
    {
        "query": "Venous thromboembolism DVT PE anticoagulation treatment duration",
        "expected_specialties": ["Hematology", "Cardiology"],
        "expected_title_keywords": ["VTE", "thromboembolism"],
        "min_relevance": 0.25,
    },
    # Infectious Disease
    {
        "query": "HIV antiretroviral therapy guidelines initial regimen",
        "expected_specialties": ["Infectious Disease"],
        "expected_title_keywords": ["HIV"],
        "min_relevance": 0.3,
    },
    {
        "query": "Urinary tract infection treatment pyelonephritis uncomplicated cystitis",
        "expected_specialties": ["Infectious Disease"],
        "expected_title_keywords": ["UTI", "urinary tract"],
        "min_relevance": 0.25,
    },
    # OB/GYN
    {
        "query": "Preeclampsia management magnesium sulfate antihypertensives",
        "expected_specialties": ["OB/GYN"],
        "expected_title_keywords": ["preeclampsia", "hypertensive"],
        "min_relevance": 0.3,
    },
    # Rheumatology
    {
        "query": "Acute gout treatment colchicine NSAIDs corticosteroids",
        "expected_specialties": ["Rheumatology"],
        "expected_title_keywords": ["gout"],
        "min_relevance": 0.3,
    },
]


async def rebuild_chroma(persist_dir: str):
    """Delete and recreate the ChromaDB collection."""
    p = Path(persist_dir)
    if p.exists():
        shutil.rmtree(p)
        print(f"  Deleted ChromaDB directory: {p}")
    else:
        print(f"  ChromaDB directory does not exist: {p}")

    # Re-init by creating a new tool instance and triggering load
    from app.tools.guideline_retrieval import GuidelineRetrievalTool
    tool = GuidelineRetrievalTool()
    await tool._ensure_initialized()
    assert tool._collection is not None, "Collection failed to initialize"
    count = tool._collection.count()
    print(f"  Rebuilt collection with {count} guidelines")
    return tool


async def show_stats(persist_dir: str):
    """Show ChromaDB collection statistics."""
    from app.tools.guideline_retrieval import GuidelineRetrievalTool
    tool = GuidelineRetrievalTool()
    await tool._ensure_initialized()
    assert tool._collection is not None, "Collection failed to initialize"

    count = tool._collection.count()
    print(f"\n  Collection: clinical_guidelines")
    print(f"  Documents: {count}")
    print(f"  Persist dir: {persist_dir}")

    if count > 0:
        # Get all metadata to show specialties
        all_data = tool._collection.get(include=["metadatas"])
        specialties = {}
        for meta in all_data["metadatas"]:
            spec = meta.get("specialty", "Unknown")
            specialties[spec] = specialties.get(spec, 0) + 1

        print(f"\n  Guidelines by specialty:")
        for spec, cnt in sorted(specialties.items()):
            print(f"    {spec:30s} {cnt}")

    return tool


async def test_single_query(tool, query_text: str, n_results: int = 5):
    """Test a single query and display results."""
    result = await tool.run(query_text, n_results=n_results)
    print(f"\n  Query: \"{query_text}\"")
    print(f"  Results: {len(result.excerpts)}")
    for i, exc in enumerate(result.excerpts):
        print(f"\n    [{i+1}] {exc.title}")
        print(f"        Source: {exc.source}")
        print(f"        Relevance: {exc.relevance_score:.4f}")
        print(f"        Excerpt: {exc.excerpt[:150]}...")


async def run_quality_tests(tool, test_queries):
    """Run all quality test queries and score results."""
    results = []

    for tq in test_queries:
        query = tq["query"]
        expected_specs = tq["expected_specialties"]
        expected_keywords = tq["expected_title_keywords"]
        min_rel = tq["min_relevance"]

        result = await tool.run(query, n_results=5)

        # Get top result info
        top_excerpt = result.excerpts[0] if result.excerpts else None
        top_title = top_excerpt.title if top_excerpt else "N/A"
        top_relevance = top_excerpt.relevance_score if top_excerpt else 0
        top_source = top_excerpt.source if top_excerpt else "N/A"

        # Check if any of the top-3 results match expected specialty
        specialty_match = False
        keyword_match = False
        matched_result_idx = -1

        for idx, exc in enumerate(result.excerpts[:3]):
            # Check source text or title for specialty/keyword matches
            title_lower = exc.title.lower()
            source_lower = exc.source.lower()
            combined = title_lower + " " + source_lower + " " + exc.excerpt.lower()

            for kw in expected_keywords:
                if kw.lower() in combined:
                    keyword_match = True
                    if matched_result_idx == -1:
                        matched_result_idx = idx
                    break

        # Relevance check
        relevance_ok = top_relevance >= min_rel

        # Overall pass: keyword match in top-3 AND minimum relevance
        passed = keyword_match and relevance_ok

        test_result = {
            "query": query[:60] + ("..." if len(query) > 60 else ""),
            "expected_specialties": expected_specs,
            "expected_keywords": expected_keywords,
            "top_title": top_title,
            "top_relevance": top_relevance,
            "keyword_match": keyword_match,
            "keyword_match_position": matched_result_idx + 1 if matched_result_idx >= 0 else 0,
            "relevance_ok": relevance_ok,
            "passed": passed,
            "all_titles": [e.title for e in result.excerpts[:5]],
            "all_relevances": [e.relevance_score for e in result.excerpts[:5]],
        }
        results.append(test_result)

    return results


async def main():
    parser = argparse.ArgumentParser(description="RAG Retrieval Quality Test")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild ChromaDB from scratch")
    parser.add_argument("--stats", action="store_true", help="Show collection statistics only")
    parser.add_argument("--query", help="Test a single query")
    parser.add_argument("--verbose", action="store_true", help="Show detailed results for each query")
    args = parser.parse_args()

    from app.config import settings
    persist_dir = settings.chroma_persist_dir

    print(f"\n{'='*70}")
    print(f"  RAG Retrieval Quality Test")
    print(f"  Persist dir: {persist_dir}")
    print(f"  Embedding model: {settings.embedding_model}")
    print(f"{'='*70}")

    if args.rebuild:
        tool = await rebuild_chroma(persist_dir)
    elif args.stats:
        await show_stats(persist_dir)
        return
    else:
        from app.tools.guideline_retrieval import GuidelineRetrievalTool
        tool = GuidelineRetrievalTool()
        await tool._ensure_initialized()
        assert tool._collection is not None, "Collection failed to initialize"
        count = tool._collection.count()
        print(f"\n  Collection has {count} documents")
        if count == 0:
            print("  ⚠ Collection is empty! Run with --rebuild to load guidelines.")
            return

    if args.query:
        await test_single_query(tool, args.query)
        return

    # Run all quality tests
    print(f"\n  Running {len(RAG_TEST_QUERIES)} retrieval quality tests...\n")
    results = await run_quality_tests(tool, RAG_TEST_QUERIES)

    # Display results
    passed_count = 0
    for r in results:
        icon = "✓" if r["passed"] else "✗"
        pos = f"@{r['keyword_match_position']}" if r["keyword_match"] else "  "
        rel = f"{r['top_relevance']:.3f}"
        print(f"  {icon} [{rel}] {pos:>3} {r['query']}")
        if not r["passed"] or args.verbose:
            print(f"       → Top: {r['top_title']}")
            if not r["keyword_match"]:
                print(f"       ✗ Expected keywords not found in top-3: {r['expected_keywords']}")
            if not r["relevance_ok"]:
                print(f"       ✗ Relevance {r['top_relevance']:.3f} below threshold")
            if args.verbose:
                for i, (t, s) in enumerate(zip(r["all_titles"], r["all_relevances"])):
                    print(f"         {i+1}. [{s:.3f}] {t}")
        if r["passed"]:
            passed_count += 1

    # Summary
    total = len(results)
    pct = (passed_count / total * 100) if total else 0
    print(f"\n{'='*70}")
    print(f"  RESULTS: {passed_count}/{total} passed ({pct:.0f}%)")
    print(f"{'='*70}")

    # By-specialty breakdown
    spec_results = {}
    for r in results:
        for spec in r.get("expected_specialties", ["Unknown"]):
            if spec not in spec_results:
                spec_results[spec] = {"passed": 0, "total": 0}
            spec_results[spec]["total"] += 1
            if r["passed"]:
                spec_results[spec]["passed"] += 1

    print(f"\n  By specialty:")
    for spec, counts in sorted(spec_results.items()):
        p = counts["passed"]
        t = counts["total"]
        bar = "█" * p + "░" * (t - p)
        print(f"    {spec:25s} {p}/{t} {bar}")

    # Relevance distribution
    all_rels = [r["top_relevance"] for r in results]
    if all_rels:
        avg_rel = sum(all_rels) / len(all_rels)
        min_rel_val = min(all_rels)
        max_rel_val = max(all_rels)
        print(f"\n  Relevance: avg={avg_rel:.3f}  min={min_rel_val:.3f}  max={max_rel_val:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
