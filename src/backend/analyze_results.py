"""
Post-analysis of MedQA validation results.

Categorizes questions by type and reports accuracy for each category.
This is important because the CDS pipeline focuses on DIAGNOSIS, while
MedQA includes many non-diagnostic questions (pharmacology, management,
biostatistics, pathophysiology).
"""
import json
import re
from collections import defaultdict
from pathlib import Path

CHECKPOINT = Path("validation/results/medqa_checkpoint.jsonl")
DATA_FILE = Path("validation/data/medqa_test.jsonl")


def classify_answer(correct_answer: str, full_question: str = "") -> str:
    """Classify the MedQA answer type.
    
    Categories:
    - diagnosis: Answer is a disease, condition, or syndrome
    - treatment: Answer is a drug, procedure, or intervention
    - management: Answer is a management strategy (reassurance, referral, etc.)
    - pathophysiology: Answer is a mechanism, pathway, or biochemical entity
    - statistics: Answer is about study design, statistics, or epidemiology
    - anatomy: Answer is about anatomy/location
    - other: Everything else
    """
    answer = correct_answer.lower().strip()
    question = full_question.lower()
    
    # Statistics / study design
    stats_patterns = [
        r"type [12] error", r"null hypothesis", r"p.value", r"confidence interval",
        r"odds ratio", r"relative risk", r"sensitivity", r"specificity",
        r"positive predictive", r"negative predictive", r"number needed",
        r"standard deviation", r"study design", r"randomized", r"case.control",
        r"cohort study", r"cross.sectional", r"meta.analysis", r"selection bias",
        r"recall bias", r"confounding", r"blinding", r"power of",
    ]
    for p in stats_patterns:
        if re.search(p, answer) or re.search(p, question):
            return "statistics"
    
    # Treatment / pharmacology (drugs, procedures, interventions)
    treatment_patterns = [
        r"^start\b", r"^administer\b", r"^give\b", r"^prescribe\b",
        r"^begin\b", r"^initiate\b", r"surgery", r"laparotomy",
        r"laparoscop", r"analgesia", r"^reassurance", r"^observation",
        r"^follow.up", r"^refer", r"^discharge",
        r"corticosteroid", r"hydrocortisone", r"fludrocortisone",
        r"prednisone", r"methylprednisolone", r"dexamethasone",
        r"amitriptyline", r"fluoxetine", r"sertraline", r"metformin",
        r"insulin", r"heparin", r"warfarin", r"aspirin",
        r"amoxicillin", r"azithromycin", r"ceftriaxone",
        r"exploratory", r"endoscop",
    ]
    for p in treatment_patterns:
        if re.search(p, answer):
            return "treatment"
    
    # Management strategies
    management_patterns = [
        r"reassurance", r"watchful waiting", r"follow.up", r"counseling",
        r"lifestyle", r"observation", r"monitor", r"admit",
        r"discharge", r"consult",
    ]
    for p in management_patterns:
        if re.search(p, answer):
            return "management"
    
    # Pathophysiology / biochemistry
    patho_patterns = [
        r"prostaglandin", r"acetaldehyde", r"histamine", r"serotonin",
        r"dopamine", r"cytokine", r"interleukin", r"antibod",
        r"complement", r"release of", r"synthesis of", r"inhibition of",
        r"degradation of", r"mutation in", r"deficiency of",
        r"mechanism", r"pathway", r"receptor", r"kinase",
        r"affective symptoms", r"diagnosis of exclusion",
    ]
    for p in patho_patterns:
        if re.search(p, answer):
            return "pathophysiology"
    
    # Anatomy
    anatomy_patterns = [
        r"lytic lesions", r"fracture", r"artery", r"vein",
        r"nerve", r"muscle", r"bone", r"ligament",
        r"right.sided", r"left.sided", r"posterior", r"anterior",
    ]
    for p in anatomy_patterns:
        if re.search(p, answer):
            return "anatomy"
    
    # Default: assume it's a diagnosis
    return "diagnosis"


def analyze():
    if not CHECKPOINT.exists():
        print("No checkpoint file found. Run validation first.")
        return
    
    # Load results
    results = []
    for line in CHECKPOINT.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            results.append(json.loads(line))
    
    # Load original questions for classification
    questions = {}
    if DATA_FILE.exists():
        raw = DATA_FILE.read_text(encoding="utf-8").strip().split("\n")
        for item_str in raw:
            item = json.loads(item_str)
            questions[item.get("question", "")] = item
    
    # Classify and categorize
    categories = defaultdict(list)
    
    for r in results:
        det = r.get("details", {})
        correct = det.get("correct_answer", "")
        full_q = det.get("full_question", "")
        
        # Try to get the full question from the ground truth
        if not full_q:
            for case_key in r.get("ground_truth", {}).keys():
                pass  # Fallback
        
        cat = classify_answer(correct, full_q)
        categories[cat].append(r)
    
    # Print summary
    print("=" * 70)
    print("  MedQA RESULTS BY QUESTION CATEGORY")
    print("=" * 70)
    
    total_cases = len(results)
    total_mentioned = sum(1 for r in results if r.get("details", {}).get("match_location", "not_found") != "not_found")
    total_diff = sum(1 for r in results if r.get("details", {}).get("match_location") == "differential")
    
    print(f"\n  OVERALL: {total_cases} cases | Mentioned: {total_mentioned}/{total_cases} ({100*total_mentioned/total_cases:.0f}%) | Differential: {total_diff}/{total_cases} ({100*total_diff/total_cases:.0f}%)")
    
    print(f"\n  {'Category':<20} {'Count':>6} {'Mentioned':>10} {'Differential':>13} {'Pipeline OK':>12}")
    print(f"  {'-'*20} {'-'*6} {'-'*10} {'-'*13} {'-'*12}")
    
    for cat in sorted(categories.keys()):
        items = categories[cat]
        n = len(items)
        mentioned = sum(1 for r in items if r.get("details", {}).get("match_location", "not_found") != "not_found")
        differential = sum(1 for r in items if r.get("details", {}).get("match_location") == "differential")
        success = sum(1 for r in items if r.get("success"))
        
        mentioned_pct = f"{100*mentioned/n:.0f}%" if n > 0 else "N/A"
        diff_pct = f"{100*differential/n:.0f}%" if n > 0 else "N/A"
        success_pct = f"{100*success/n:.0f}%" if n > 0 else "N/A"
        
        print(f"  {cat:<20} {n:>6} {mentioned:>5} ({mentioned_pct:>4}) {differential:>7} ({diff_pct:>4}) {success:>6} ({success_pct:>4})")
    
    # Detailed per-case
    print(f"\n  DETAILED PER-CASE RESULTS:")
    print(f"  {'Case':<14} {'Cat':<15} {'Location':<14} {'Correct':<35} {'Top Dx':<35}")
    print(f"  {'-'*14} {'-'*15} {'-'*14} {'-'*35} {'-'*35}")
    
    for r in results:
        det = r.get("details", {})
        correct = det.get("correct_answer", "?")[:34]
        top = det.get("top_diagnosis", "?")[:34]
        loc = det.get("match_location", "not_found")
        cat = classify_answer(det.get("correct_answer", ""))
        
        print(f"  {r['case_id']:<14} {cat:<15} {loc:<14} {correct:<35} {top:<35}")
    
    # Key insight
    diag_items = categories.get("diagnosis", [])
    if diag_items:
        d_mentioned = sum(1 for r in diag_items if r.get("details", {}).get("match_location", "not_found") != "not_found")
        d_diff = sum(1 for r in diag_items if r.get("details", {}).get("match_location") == "differential")
        d_n = len(diag_items)
        print(f"\n  KEY INSIGHT:")
        print(f"  On DIAGNOSTIC questions only: Mentioned {d_mentioned}/{d_n} ({100*d_mentioned/d_n:.0f}%), Differential {d_diff}/{d_n} ({100*d_diff/d_n:.0f}%)")
        print(f"  The CDS pipeline is designed for diagnosis support; non-diagnostic questions")
        print(f"  (treatment, stats, pathophysiology) are outside its intended scope.")


if __name__ == "__main__":
    analyze()
