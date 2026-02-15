"""Quick progress checker for validation run."""
import json
from pathlib import Path

checkpoint = Path("validation/results/medqa_checkpoint.jsonl")
if not checkpoint.exists():
    print("No checkpoint file found")
    exit()

lines = checkpoint.read_text(encoding="utf-8").strip().split("\n")
print(f"Completed: {len(lines)}/50")

matches = 0
diff_matches = 0
top3_matches = 0
failures = 0

for line in lines:
    d = json.loads(line)
    det = d.get("details", {})
    scores = d.get("scores", {})
    loc = det.get("match_location", "not_found")
    
    if not d.get("success"):
        failures += 1
    if loc != "not_found":
        matches += 1
    if loc == "differential":
        diff_matches += 1
    if scores.get("top3_accuracy", 0) > 0:
        top3_matches += 1

print(f"Pipeline success: {len(lines) - failures}/{len(lines)}")
print(f"Mentioned matches: {matches}/{len(lines)} ({100*matches/len(lines):.0f}%)")
print(f"Differential matches: {diff_matches}/{len(lines)} ({100*diff_matches/len(lines):.0f}%)")
print(f"Top-3 matches: {top3_matches}/{len(lines)} ({100*top3_matches/len(lines):.0f}%)")

# Show last 5 cases
print("\nRecent cases:")
for line in lines[-5:]:
    d = json.loads(line)
    det = d.get("details", {})
    correct = det.get("correct_answer", "?")[:45]
    top = det.get("top_diagnosis", "?")[:45]
    loc = det.get("match_location", "not_found")
    t = d.get("pipeline_time_ms", 0)
    print(f"  {d['case_id']}: [{loc}] {t/1000:.0f}s | correct={correct} | top={top}")
