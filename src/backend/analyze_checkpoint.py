"""Quick analysis of MedQA checkpoint data."""
import json

path = "validation/results/medqa_checkpoint.jsonl"
with open(path) as f:
    results = [json.loads(l) for l in f]

print(f"Cases completed: {len(results)}\n")

# ── Table view ──
fmt = "{:<12} {:>3} {:>3} {:>4} {:>7} {:>3} {:>4}  {:<15} {:<42} {}"
print(fmt.format("ID", "t1", "t3", "diff", "ms", "#dx", "rnk", "match_loc", "correct_answer", "top_diagnosis"))
print("-" * 145)

for r in results:
    d = r["details"]
    t1 = "Y" if r["scores"]["top1_accuracy"] else "N"
    t3 = "Y" if r["scores"]["top3_accuracy"] else "N"
    da = "Y" if r["scores"].get("differential_accuracy") else "N"
    rank = d.get("found_at_rank", -1)
    loc = d.get("match_location", "?")
    ca = d["correct_answer"][:42]
    td = d.get("top_diagnosis", "?")[:45]
    print(fmt.format(r["case_id"], t1, t3, da, r["pipeline_time_ms"], d.get("num_diagnoses", 0), rank, loc, ca, td))

print()

# ── Timing analysis ──
correct = [r for r in results if r["scores"]["top1_accuracy"]]
wrong = [r for r in results if not r["scores"]["top1_accuracy"]]
mentioned = [r for r in results if r["scores"].get("mentioned_accuracy")]
top3 = [r for r in results if r["scores"]["top3_accuracy"]]
diff_only = [r for r in results if r["scores"].get("differential_accuracy")]

if correct:
    avg = sum(r["pipeline_time_ms"] for r in correct) / len(correct)
    print(f"Correct (top1) avg time: {avg:.0f}ms  ({len(correct)}/{len(results)} = {len(correct)/len(results)*100:.0f}%)")
if top3:
    avg = sum(r["pipeline_time_ms"] for r in top3) / len(top3)
    print(f"Correct (top3) avg time: {avg:.0f}ms  ({len(top3)}/{len(results)} = {len(top3)/len(results)*100:.0f}%)")
if diff_only:
    avg = sum(r["pipeline_time_ms"] for r in diff_only) / len(diff_only)
    print(f"Differential only:       {avg:.0f}ms  ({len(diff_only)}/{len(results)} = {len(diff_only)/len(results)*100:.0f}%)")
if wrong:
    avg = sum(r["pipeline_time_ms"] for r in wrong) / len(wrong)
    print(f"Wrong   (top1) avg time: {avg:.0f}ms  ({len(wrong)}/{len(results)} = {len(wrong)/len(results)*100:.0f}%)")
if mentioned:
    print(f"Mentioned anywhere:      {len(mentioned)}/{len(results)}")

# ── Match location breakdown ──
print("\n=== MATCH LOCATION BREAKDOWN ===")
loc_counts = {}
for r in results:
    loc = r["details"].get("match_location", "not_found")
    loc_counts[loc] = loc_counts.get(loc, 0) + 1
for loc, count in sorted(loc_counts.items()):
    print(f"  {loc:<20} {count:>3} ({count/len(results)*100:.0f}%)")

# ── Detailed per-case (new fields if available) ──
print("\n=== PER-CASE DETAIL ===")
for r in results:
    d = r["details"]
    cid = r["case_id"]
    loc = d.get("match_location", "?")
    ca = d["correct_answer"]
    td = d.get("top_diagnosis", "?")
    all_dx = d.get("all_diagnoses", [td])
    all_next = d.get("all_next_steps", [])
    all_recs = d.get("all_recommendations", [])
    t1 = "Y" if r["scores"]["top1_accuracy"] else "N"

    print(f"\n  {cid} [t1={t1}, loc={loc}]")
    print(f"    Expected: {ca}")
    print(f"    Differential: {', '.join(all_dx)}")
    if all_next:
        print(f"    Next steps: {'; '.join(all_next[:3])}")
    if all_recs:
        print(f"    Recommendations: {'; '.join(str(r)[:60] for r in all_recs[:3])}")

# ── Answer type vs accuracy ──
print("\n=== ANSWER TYPE vs ACCURACY ===")
dx_correct = dx_total = mgmt_correct = mgmt_total = 0
action_words = ["start", "stop", "give", "prescribe", "perform", "order", "refer",
                "increase", "decrease", "switch", "add", "monitor", "observation",
                "reassure", "discharge", "admit", "excess", "adaptation", "exclusion",
                "it is", "right-sided", "affective", "exploratory", "lytic"]
for r in results:
    ca = r["details"]["correct_answer"]
    is_dx = not any(w.lower() in ca.lower() for w in action_words)
    if is_dx:
        dx_total += 1
        if r["scores"]["top1_accuracy"]:
            dx_correct += 1
    else:
        mgmt_total += 1
        if r["scores"]["top1_accuracy"]:
            mgmt_correct += 1

if dx_total:
    print(f"  Diagnosis questions:    {dx_correct}/{dx_total} = {dx_correct/dx_total*100:.0f}%")
if mgmt_total:
    print(f"  Mgmt/concept questions: {mgmt_correct}/{mgmt_total} = {mgmt_correct/mgmt_total*100:.0f}%")

dx_counts = [r["details"].get("num_diagnoses", 0) for r in results]
print(f"\nDiagnoses generated: min={min(dx_counts)}, max={max(dx_counts)}, avg={sum(dx_counts)/len(dx_counts):.1f}")
