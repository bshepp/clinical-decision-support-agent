# TODO — Next Session Action Items

> **Last updated:** Feb 15, 2026 — Experimental track system built.  
> **Read this first** if you're a new AI instance picking up this project.  
> **See also:** `CLAUDE.md` (project intelligence) and `TRACKS.md` (track registry).

---

## High Priority (Do Next)

### 1. Run Experimental Tracks

Three experimental tracks are built and ready to test. See `TRACKS.md` for full details.

**Track B — RAG Variants** (`src/backend/tracks/rag_variants/`)
```bash
cd src/backend
python -m tracks.rag_variants.run_variants --max-cases 10   # smoke test
python -m tracks.rag_variants.run_variants                    # full sweep
```
Tests 10 configurations: chunking strategies (none, fixed-256, fixed-512, sentence, overlap), embedding models (MiniLM-L6, MiniLM-L12, MPNet, MedCPT), top-k sweep (3, 5, 10), and reranking.

**Track C — Iterative Refinement** (`src/backend/tracks/iterative/`)
```bash
python -m tracks.iterative.run_iterative --max-cases 10
python -m tracks.iterative.run_iterative
```
Tests 4 configurations: 2-round, 3-round, 5-round, and aggressive-critic. Produces cost/benefit data per iteration.

**Track D — Arbitrated Parallel** (`src/backend/tracks/arbitrated/`)
```bash
python -m tracks.arbitrated.run_arbitrated --max-cases 10
python -m tracks.arbitrated.run_arbitrated
```
Tests 4 configurations: 3-specialist/1-round, 5-specialist/1-round, 3-specialist/2-round, 5-specialist/2-round. Specialists: Cardiologist, Neurologist, ID, General IM, Emergency Medicine.

**Prerequisites:**
- Resume HF Endpoint (`medgemma-27b-cds`) — allow 5–15 min cold start (~$2.50/hr)
- Activate venv: `src/backend/venv/`
- May need: `pip install sentence-transformers` for MedCPT/MPNet/reranking variants

### 2. Record the Demo Video

Video script is ready: `docs/video_script.md`. Need to actually record:
1. Resume HF Endpoint
2. Start backend + frontend locally
3. Record ~3 min screencast following the script
4. Upload to YouTube/Loom and get the link

### 3. Submit on Kaggle

Kaggle writeup content is ready: `docs/kaggle_writeup.md`. Steps:
1. Go to competition page → "New Writeup"
2. Paste writeup content (fill in team name/member info first)
3. Select tracks: Main Track + Agentic Workflow Prize
4. Add links: video URL, GitHub repo, (optional) live demo
5. Click Submit
6. **Fill in [Your Name] placeholder** in the team table

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
| **Track System** | ✅ **Scaffolded** | **4 tracks (A/B/C/D), shared utils, all runners built — needs experimentation** |
| Track B — RAG Variants | ✅ Built | 10 variants (chunking × embedding × rerank), ready to run |
| Track C — Iterative Refinement | ✅ Built | 4 configs (2/3/5-round + aggressive), ready to run |
| Track D — Arbitrated Parallel | ✅ Built | 4 configs (3/5 specialists × 1/2 rounds), ready to run |
| Documentation (8+ files) | ✅ Audited | All docs updated and cross-checked |
| test_e2e.py | ✅ Fixed | Now asserts 6 steps + conflict_detection |
| GitHub | ✅ Pushed | `bshepp/clinical-decision-support-agent` (master) |
| Kaggle Writeup | ✅ Draft ready | `docs/kaggle_writeup.md` — paste into Kaggle |
| Video Script | ✅ Ready | `docs/video_script.md` — 3 min narration |
| Demo Video | ⬜ Not started | Required for submission |

**Key files:**
- Backend entry: `src/backend/app/main.py`
- Orchestrator: `src/backend/app/agent/orchestrator.py`
- MedGemma service: `src/backend/app/services/medgemma.py`
- Validation CLI: `src/backend/validation/run_validation.py`
- **Track registry: `TRACKS.md`**
- **Project intelligence: `CLAUDE.md`**
- HF Endpoint guide: `docs/deploy_medgemma_hf.md`
- All docs: `README.md`, `docs/architecture.md`, `docs/test_results.md`, `docs/writeup_draft.md`, `DEVELOPMENT_LOG.md`

**Infrastructure:**
- HF Endpoint: `medgemma-27b-cds` at `https://lisvpf8if1yhgxn2.us-east-1.aws.endpoints.huggingface.cloud`
- Dev ports: Backend = 8002 (not 8000 — zombie process issue), Frontend = 3000
- Virtual env: `src/backend/venv/`
