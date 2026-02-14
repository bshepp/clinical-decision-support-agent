# TODO — Next Session Action Items

> **Last updated:** End of validation framework + documentation audit session.
> **Read this first** if you're a new AI instance picking up this project.

---

## High Priority (Do Next)

### 1. Run Full-Scale Validation (~2 hours total)

The validation framework is built and tested with a 3-case smoke test. It needs a proper run:

```bash
cd src/backend

# MedQA — 50 cases, ~45 min
python -m validation.run_validation --medqa --max-cases 50 --seed 42 --delay 2

# MTSamples — 50 cases, ~45 min
python -m validation.run_validation --mtsamples --max-cases 50 --seed 42 --delay 2

# PMC Case Reports — 10-20 cases (smaller pool), ~15-30 min
python -m validation.run_validation --pmc --max-cases 20 --seed 42 --delay 2
```

Results save to `validation/results/`. After running, update:
- `docs/test_results.md` Section 6 with real numbers (replace smoke test placeholder)
- `docs/writeup_draft.md` validation methodology section with actual metrics
- `README.md` "External Dataset Validation" section

### 2. Update Writeup with Actual Validation Metrics

`docs/writeup_draft.md` currently says "initial smoke test" and "in progress." Once full validation is done, replace with actual numbers (top-1 accuracy, parse success rates, etc.).

### 3. Record a Demo Video

The writeup says "Video: [To be recorded]". Record a ~3 min screencast showing:
1. Pasting a patient case
2. Watching the 6-step pipeline execute live
3. Reviewing the CDS report (especially conflicts section)
4. Showing validation results

---

## Medium Priority

### 4. CI Gating on Validation Scores

Add a GitHub Action or pre-commit check that runs a small validation suite (e.g., 5 MedQA cases) and fails if top-1 accuracy drops below a threshold. This prevents regressions.

### 5. PMC Harness Improvements

The PMC case fetcher currently gets ~5 cases per run. The limiting factor is title-based diagnosis extraction — many PubMed case report titles don't follow parseable patterns. Options:
- Use the full-text XML API (not just abstracts) to extract "final diagnosis" from structured sections
- Add more title regex patterns
- Use the LLM to extract the diagnosis from the abstract itself (meta, but effective)

### 6. Calibrated Uncertainty Indicators

We deliberately removed numeric confidence scores (see Phase 8 in DEVELOPMENT_LOG.md). If revisiting uncertainty communication:
- Consider evidence-strength indicators per recommendation instead of a single composite score
- Look at conformal prediction or test-time compute approaches if fine-tuning
- Do NOT add back uncalibrated float scores — the anchoring bias risk is real

---

## Low Priority / Future

### 7. Model Upgrade Path

Currently using `gemma-3-27b-it`. When available, evaluate:
- MedGemma (medical-specific Gemma fine-tune) if released
- Smaller/distilled models for latency reduction
- Specialized models for individual pipeline steps (e.g., a parse-only model)

### 8. EHR Integration Prototype

Current input is manual text paste. A FHIR client could auto-populate patient data. This is a significant scope expansion but would dramatically increase real-world usability.

### 9. Frontend Polish

- Loading skeletons during pipeline execution
- Dark mode
- Export report as PDF
- Mobile-responsive layout

---

## Project State Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Backend (6-step pipeline) | ✅ Complete | All steps working, conflict detection added |
| Frontend (Next.js) | ✅ Complete | Real-time pipeline viz, CDS report with conflicts |
| RAG (62 guidelines) | ✅ Complete | 30/30 quality test, 100% top-1 accuracy |
| Conflict Detection | ✅ Complete | Integrated into pipeline, frontend, and docs |
| Validation Framework | ✅ Built | Smoke-tested only — needs full-scale runs |
| Documentation (5 files) | ✅ Audited | All docs updated and cross-checked |
| test_e2e.py | ✅ Fixed | Now asserts 6 steps + conflict_detection |
| GitHub | ✅ Pushed | `bshepp/clinical-decision-support-agent` (master) |

**Key files:**
- Backend entry: `src/backend/app/main.py`
- Orchestrator: `src/backend/app/agent/orchestrator.py`
- Validation CLI: `src/backend/validation/run_validation.py`
- All docs: `README.md`, `docs/architecture.md`, `docs/test_results.md`, `docs/writeup_draft.md`, `DEVELOPMENT_LOG.md`

**Dev ports:** Backend = 8002 (not 8000 — zombie process issue), Frontend = 3000
