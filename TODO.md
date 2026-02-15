# TODO — Next Session Action Items

> **Last updated:** After 50-case MedQA validation, MedGemma HF Endpoint deployment, and documentation audit.  
> **Read this first** if you're a new AI instance picking up this project.

---

## High Priority (Do Next)

### 1. Record a Demo Video

The writeup says "Video: [To be recorded]". Record a ~3 min screencast showing:
1. Pasting a patient case
2. Watching the 6-step pipeline execute live
3. Reviewing the CDS report (especially conflicts section)
4. Showing validation results

**Note:** Resume the HF Endpoint first (`medgemma-27b-cds` on HuggingFace). It costs ~$2.50/hr and is currently **paused**. Allow 5–15 min for cold start.

### 2. Finalize Submission Writeup

`docs/writeup_draft.md` has been updated with 50-case MedQA results. Still needs:
- Team name / member info filled in
- Final polish for 3-page limit
- Links to video and live demo (once recorded/deployed)

### 3. Improve Diagnostic Accuracy (Optional)

Current 50-case MedQA accuracy: 36% top-1, 38% mentioned. Potential improvements:
- **Specialist agents (Option B):** Route to domain-specific reasoning agents for cardiology, neurology, etc.
- **Better prompting:** Further refine `clinical_reasoning.py` system prompt
- **Multi-turn reasoning:** Add a self-critique / verification step before synthesis
- **Run MTSamples + PMC validation** for additional metrics

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

### 7. Model Optimization

Currently using `google/medgemma-27b-text-it` on 1× A100 80 GB. Options:
- Smaller/quantized models for latency reduction (medgemma-4b-it for lighter steps)
- Specialized models for individual pipeline steps (e.g., a parse-only model)
- Batch inference optimizations

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
| MedGemma HF Endpoint | ✅ Deployed | `medgemma-27b-cds`, 1× A100 80 GB, scale-to-zero, **currently paused** |
| MedQA Validation (50 cases) | ✅ Complete | 36% top-1, 38% mentioned, 94% pipeline success |
| Validation Framework | ✅ Complete | MedQA done; MTSamples + PMC harnesses built but not yet run at scale |
| Documentation (8+ files) | ✅ Audited | All docs updated and cross-checked |
| test_e2e.py | ✅ Fixed | Now asserts 6 steps + conflict_detection |
| GitHub | ✅ Pushed | `bshepp/clinical-decision-support-agent` (master) |
| Demo Video | ⬜ Not started | Required for submission |
| Submission Writeup | 🔄 In progress | Template filled, needs final polish |

**Key files:**
- Backend entry: `src/backend/app/main.py`
- Orchestrator: `src/backend/app/agent/orchestrator.py`
- MedGemma service: `src/backend/app/services/medgemma.py`
- Validation CLI: `src/backend/validation/run_validation.py`
- HF Endpoint guide: `docs/deploy_medgemma_hf.md`
- All docs: `README.md`, `docs/architecture.md`, `docs/test_results.md`, `docs/writeup_draft.md`, `DEVELOPMENT_LOG.md`

**Infrastructure:**
- HF Endpoint: `medgemma-27b-cds` at `https://lisvpf8if1yhgxn2.us-east-1.aws.endpoints.huggingface.cloud`
- Dev ports: Backend = 8002 (not 8000 — zombie process issue), Frontend = 3000
- Virtual env: `src/backend/venv/`
