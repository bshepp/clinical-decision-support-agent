# Development Log — CDS Agent

> Chronological record of the build process, problems encountered, and solutions applied.

---

## Phase 1: Project Scaffolding

### Decision: Track Selection

Chose the **Agentic Workflow Prize** track ($10K) for the MedGemma Impact Challenge. The clinical decision support use case maps perfectly to an agentic architecture — multiple specialized tools orchestrated by a central agent.

### Architecture Design

Designed a 5-step sequential pipeline:

1. Parse patient data (LLM)
2. Clinical reasoning / differential diagnosis (LLM)
3. Drug interaction check (external APIs)
4. Guideline retrieval (RAG)
5. Synthesis into CDS report (LLM)

**Key design choices:**
- **Custom orchestrator** instead of LangChain — simpler, more transparent, no framework overhead
- **WebSocket streaming** — clinician sees each step execute in real time (critical for trust)
- **Pydantic v2 everywhere** — all inter-step data is strongly typed

### Backend Scaffold

Built the FastAPI backend from scratch:

- `app/main.py` — FastAPI app with CORS, router includes, lifespan
- `app/config.py` — Pydantic Settings from `.env`
- `app/models/schemas.py` — All domain models (~238 lines, 10+ Pydantic models)
- `app/agent/orchestrator.py` — 5-step pipeline (267 lines)
- `app/services/medgemma.py` — LLM service wrapping OpenAI SDK
- `app/tools/` — 5 tool modules (one per pipeline step)
- `app/api/` — 3 route modules (health, cases, WebSocket)

### Frontend Scaffold

Built the Next.js 14 frontend:

- `PatientInput.tsx` — Text area + 3 pre-loaded sample cases
- `AgentPipeline.tsx` — Real-time 5-step status visualization
- `CDSReport.tsx` — Final report renderer
- `useAgentWebSocket.ts` — WebSocket hook for real-time updates
- `next.config.js` — API proxy to backend

---

## Phase 2: Integration & Bug Fixes

### Bug: Gemma System Prompt 400 Error

**Problem:** The first LLM call failed with HTTP 400. Gemma models via the Google AI Studio OpenAI-compatible endpoint do not support `role: "system"` messages — a fundamental difference from OpenAI's API.

**Solution:** Modified `medgemma.py` to detect system messages and fold them into the first user message with a `[System Instructions]` prefix. All pipeline steps now work correctly.

**File changed:** `src/backend/app/services/medgemma.py`

### Bug: RxNorm API — `rxnormId` Is a List

**Problem:** The drug interaction checker crashed when querying RxNorm. The NLM API returns `rxnormId` as a **list** (e.g., `["12345"]`), not a scalar string. The code assumed a string.

**Solution:** Added type checking — if `rxnormId` is a list, take the first element; if it's a string, use directly.

**File changed:** `src/backend/app/tools/drug_interactions.py`

### Bug: OpenAI SDK Version Mismatch

**Problem:** `openai==1.0.0` had breaking API changes compared to the code written for the older API pattern.

**Solution:** Pinned to `openai==1.51.0` in `requirements.txt`, which is compatible with both the modern SDK API and the Google AI Studio OpenAI-compatible endpoint.

**File changed:** `src/backend/requirements.txt`

### Bug: Port 8000 Zombie Processes

**Problem:** Previous server instances left zombie processes holding port 8000. New `uvicorn` instances couldn't bind.

**Solution:** Switched to port 8002 for development. Updated `next.config.js` and `useAgentWebSocket.ts` to proxy to 8002.

**Files changed:** `src/frontend/next.config.js`, `src/frontend/src/hooks/useAgentWebSocket.ts`

---

## Phase 3: First Successful E2E Test

### Test Case: Chest Pain / ACS

Submitted a 62-year-old male with crushing substernal chest pain, diaphoresis, HTN, on lisinopril + metformin + atorvastatin.

**Results — all 5 steps passed:**

| Step | Duration | Outcome |
|------|----------|---------|
| Parse | 7.8 s | Correct structured extraction |
| Reason | 21.2 s | ACS as top differential (correct) |
| Drug Check | 11.3 s | Queried all 3 medications |
| Guidelines | 9.6 s | Retrieved ACS/chest pain guidelines |
| Synthesis | 25.3 s | Comprehensive report with recommendations |

This was the first end-to-end success. Total pipeline: ~75 seconds.

---

## Phase 4: Project Direction Shift

### Decision: From Competition to Real Application

After achieving the first successful E2E test, made the decision to shift focus from "winning a competition" to "building a genuinely important medical application." The clinical decision support problem is real and impactful regardless of competition outcomes.

This shift influenced subsequent work — emphasis on:
- Comprehensive clinical coverage (more specialties, more guidelines)
- Thorough testing (not just demos)
- Proper documentation

---

## Phase 5: RAG Expansion

### Guideline Corpus: 2 → 62

The initial RAG system had only 2 minimal fallback guidelines. Expanded to a comprehensive corpus:

- **Created:** `app/data/clinical_guidelines.json` — 62 guidelines across 14 specialties
- **Updated:** `guideline_retrieval.py` — loads from JSON, stores specialty/ID metadata in ChromaDB
- **Sources:** ACC/AHA, ADA, GOLD, GINA, IDSA, ACOG, AAN, APA, AAP, ACR, ASH, KDIGO, WHO, USPSTF

### ChromaDB Rebuild

Had to kill locking processes holding the ChromaDB files before rebuilding. After clearing locks, ChromaDB successfully indexed all 62 guidelines with `all-MiniLM-L6-v2` embeddings (384 dimensions).

---

## Phase 6: Comprehensive Test Suite

### RAG Quality Tests (30 queries)

Created `test_rag_quality.py` with 30 clinical queries, each mapped to an expected guideline ID:

- **Result: 30/30 passed (100%)**
- Average relevance score: 0.639
- Every query returned the correct guideline as the #1 result
- All 14 specialty categories achieved 100% pass rate

### Clinical Test Cases (22 scenarios)

Created `test_clinical_cases.py` with 22 diverse clinical scenarios:

- Covers 14+ specialties (Cardiology, EM, Endocrinology, Neurology, Pulmonology, GI, ID, Psych, Peds, Nephrology, Toxicology, Geriatrics)
- Each case has: clinical vignette, expected specialty, validation keywords
- Supports CLI flags: `--case`, `--specialty`, `--list`, `--report`, `--quiet`

---

## Phase 7: Documentation

Performed comprehensive documentation audit. Found:
- README was outdated (wrong port, missing test info, incomplete structure tree)
- Architecture doc lacked implementation specifics (RAG details, Gemma workaround, timing)
- Writeup draft was 100% TODO placeholders
- No test results documentation existed
- No development log existed

Rewrote/created all documentation:
- **README.md** — Complete rewrite with results, RAG corpus info, updated structure, corrected setup
- **docs/architecture.md** — Updated with actual implementation details, timing, config, limitations
- **docs/test_results.md** — New file documenting all test results and reproduction steps
- **DEVELOPMENT_LOG.md** — This file
- **docs/writeup_draft.md** — Filled in with actual project information

---

## Phase 8: Conflict Detection Feature

### Design Decision: Drop Confidence Scores, Add Conflict Detection

During review, identified that the system's "confidence" was just the LLM picking a label (LOW/MODERATE/HIGH) — not a calibrated score. Composite numeric confidence scores were considered and **rejected** because:
- Uncalibrated confidence values are dangerous (clinician anchoring bias)
- No training data exists to calibrate outputs
- A single number hides more than it reveals

**Instead, added Conflict Detection** — a new pipeline step that compares guideline recommendations against the patient's actual data to identify specific, actionable gaps. This provides direct patient safety value without requiring calibration.

### Implementation

**New models added to `schemas.py`:**
- `ConflictType` enum — 6 categories: omission, contradiction, dosage, monitoring, allergy_risk, interaction_gap
- `ClinicalConflict` model — Each conflict has: type, severity, guideline_source, guideline_text, patient_data, description, suggested_resolution
- `ConflictDetectionResult` — List of conflicts + summary + guidelines_checked count
- `conflicts` field added to `CDSReport`
- `conflict_detection` field added to `AgentState`

**New tool: `conflict_detection.py`:**
- Takes patient profile, clinical reasoning, drug interactions, and guidelines
- Uses MedGemma at low temperature (0.1) for safety-critical analysis
- Returns structured `ConflictDetectionResult` with specific, actionable conflicts
- Graceful degradation: returns empty if no guidelines available

**Pipeline changes (`orchestrator.py`):**
- Pipeline expanded from 5 to 6 steps
- New Step 5: Conflict Detection (between guideline retrieval and synthesis)
- Synthesis (now Step 6) receives conflict data and prominently includes it in the report

**Synthesis changes (`synthesis.py`):**
- Accepts `conflict_detection` parameter
- New "Conflicts & Gaps" section in synthesis prompt
- Fallback: copies detected conflicts directly into report if LLM doesn't populate the structured field

**Frontend changes (`CDSReport.tsx`):**
- New "Conflicts & Gaps Detected" section with high visual prominence
- Red border container, severity-coded left-accent cards (critical=red, high=orange, moderate=yellow, low=blue)
- Side-by-side "Guideline says" vs "Patient data" comparison
- Green-highlighted suggested resolutions
- Positioned immediately after drug interactions for maximum visibility

**Files created:** `src/backend/app/tools/conflict_detection.py` (1 new file)
**Files modified:** `schemas.py`, `orchestrator.py`, `synthesis.py`, `CDSReport.tsx` (4 files)

---

## Dependency Inventory

### Python Backend (`requirements.txt`)

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | 0.115.0 | Web framework |
| uvicorn | 0.30.6 | ASGI server |
| openai | 1.51.0 | LLM API client (OpenAI-compatible) |
| chromadb | 0.5.7 | Vector database for RAG |
| sentence-transformers | 3.1.1 | Embedding model |
| httpx | 0.27.2 | Async HTTP client (API calls) |
| torch | 2.4.1 | PyTorch (sentence-transformers dependency) |
| transformers | 4.45.0 | HuggingFace transformers |
| pydantic-settings | 2.5.2 | Settings management |
| pydantic | 2.9.2 | Data validation |
| websockets | 13.1 | WebSocket support |
| python-dotenv | 1.0.1 | .env file loading |
| numpy | 1.26.4 | Numerical computing |

### Frontend (`package.json`)

| Package | Purpose |
|---------|---------|
| next 14.x | React framework |
| react 18.x | UI library |
| typescript | Type safety |
| tailwindcss | Styling |

---

## Environment Configuration

All config via `.env` (template in `.env.template`):

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MEDGEMMA_API_KEY` | Yes | — | HuggingFace API token or Google AI Studio API key |
| `MEDGEMMA_BASE_URL` | No | `""` (empty) | LLM endpoint (HF Endpoint URL/v1 or Google AI Studio URL) |
| `MEDGEMMA_MODEL_ID` | No | `google/medgemma` | Model identifier (`tgi` for HF Endpoints, or full model name) |
| `HF_TOKEN` | No | `""` | HuggingFace token for dataset downloads |
| `CHROMA_PERSIST_DIR` | No | `./data/chroma` | ChromaDB storage |
| `EMBEDDING_MODEL` | No | `sentence-transformers/all-MiniLM-L6-v2` | RAG embeddings |
| `MAX_GUIDELINES` | No | `5` | Guidelines per RAG query |
| `AGENT_TIMEOUT` | No | `120` | Pipeline timeout (seconds) |

---

## Phase 9: External Dataset Validation Framework

### Motivation

Internal tests (RAG quality, clinical cases) are useful but don't measure diagnostic accuracy against ground truth. Added a validation framework to test the full pipeline against real-world clinical datasets with known correct answers.

### Datasets Evaluated

| Dataset | Source | What It Tests |
|---------|--------|---------------|
| **MedQA (USMLE)** | HuggingFace — `GBaker/MedQA-USMLE-4-options` | Diagnostic accuracy (1,273 USMLE-style questions with verified answers) |
| **MTSamples** | GitHub — `socd06/medical-nlp` | Parse quality & field completeness on real medical transcription notes |
| **PMC Case Reports** | PubMed E-utilities (esearch + efetch) | Diagnostic accuracy on published case reports with known diagnoses |

### Architecture

Created `src/backend/validation/` package:

- **`base.py`** — Core framework: `ValidationCase`, `ValidationResult`, `ValidationSummary` dataclasses. `run_cds_pipeline()` invokes the Orchestrator directly (no HTTP server needed). Includes `fuzzy_match()` token-overlap scorer and `diagnosis_in_differential()` checker.
- **`harness_medqa.py`** — Downloads JSONL from HuggingFace, extracts clinical vignettes (strips question stems), scores top-1/top-3/mentioned diagnostic accuracy.
- **`harness_mtsamples.py`** — Downloads CSV, filters to relevant specialties, stratified sampling. Scores parse success, field completeness, specialty alignment, has_differential, has_recommendations.
- **`harness_pmc.py`** — Uses NCBI E-utilities with 20 curated queries across specialties. Extracts diagnosis from article titles via regex patterns. Scores diagnostic accuracy.
- **`run_validation.py`** — Unified CLI: `python -m validation.run_validation --all --max-cases 10`. Supports `--fetch-only`, `--no-drugs`, `--no-guidelines`, `--seed`, `--delay`.

### Problems Solved

1. **MedQA URL 404:** Original GitHub raw URL was stale. Fixed to HuggingFace direct download.
2. **MTSamples URL 404:** Original mirror was down. Found working mirror at `socd06/medical-nlp`.
3. **PMC fetcher returned 0 cases:** PubMed API worked, but title regex patterns didn't match common formats like "X: A Case Report." Added 3 new title patterns and fixed query-based fallback extraction.
4. **`datetime.utcnow()` deprecation:** Replaced with `datetime.now(timezone.utc)` throughout.
5. **Pipeline time display bug:** `print_summary` showed time metrics as percentages. Fixed by reordering type checks.

### Initial Results (Smoke Test)

Ran 3 MedQA cases through the full pipeline:
- **Parse success:** 100% (3/3)
- **Top-1 diagnostic accuracy:** 66.7% (2/3)
- **Avg pipeline time:** ~94 seconds per case

Full validation runs (50–100+ cases) are planned for the next session.

**Files created:** `validation/__init__.py`, `validation/base.py`, `validation/harness_medqa.py`, `validation/harness_mtsamples.py`, `validation/harness_pmc.py`, `validation/run_validation.py`  
**Files modified:** `.gitignore` (added `validation/data/` and `validation/results/`)

---

## Phase 11: MedGemma HuggingFace Dedicated Endpoint

### Motivation

The competition requires using HAI-DEF models (MedGemma). Google AI Studio served `gemma-3-27b-it` for development, but for the final submission we needed the actual `google/medgemma-27b-text-it` model. HuggingFace Dedicated Endpoints provide an OpenAI-compatible TGI server with scale-to-zero billing.

### Deployment

- **Endpoint name:** `medgemma-27b-cds`
- **Model:** `google/medgemma-27b-text-it`
- **Instance:** 1× NVIDIA A100 80 GB (AWS `us-east-1`)
- **Container:** Text Generation Inference (TGI) with `DTYPE=bfloat16`
- **Scale-to-zero:** Enabled (15 min idle timeout)
- **Cost:** ~$2.50/hr when running

### Key Configuration

After initial deployment, the default TGI token limits (`MAX_INPUT_TOKENS=4096`) caused 422 errors on longer synthesis prompts. Updated endpoint environment:

- `MAX_INPUT_TOKENS=12288`
- `MAX_TOTAL_TOKENS=16384`

Also reduced per-step `max_tokens` to stay within limits:
- `patient_parser.py`: 1500
- `clinical_reasoning.py`: 3072
- `conflict_detection.py`: 2000
- `synthesis.py`: 3000

### Code Changes

- **`medgemma.py`:** Updated to send `role: "system"` natively (TGI supports it), with automatic fallback to folding system prompt into user message for Google AI Studio compatibility.
- **`.env`:** Updated `MEDGEMMA_BASE_URL` to HF endpoint URL, `MEDGEMMA_API_KEY` to HF token, `MEDGEMMA_MODEL_ID=tgi`.
- **`.env.template`:** Updated with MedGemma model name and HF Endpoint instructions.

### Verification

Single-case test: Chikungunya question → correct diagnosis appeared at rank 5 in differential. All 6 pipeline steps completed in 281s.

**Deployment guide:** `docs/deploy_medgemma_hf.md`

---

## Phase 12: 50-Case MedQA Validation

### Setup

Ran 50 MedQA (USMLE) cases through the full pipeline using the MedGemma HF Endpoint:

```bash
cd src/backend
python -m validation.run_validation --medqa --max-cases 50 --seed 42 --delay 2
```

### Results

| Metric | Value |
|--------|-------|
| Cases run | 50 |
| Pipeline success | 94% (47/50) |
| Top-1 diagnostic accuracy | 36% |
| Top-3 diagnostic accuracy | 38% |
| Differential accuracy | 10% |
| Mentioned in report | 38% |
| Avg pipeline time | 204 s/case |
| Total run time | ~60 min |

### Question Type Breakdown

Used `analyze_results.py` to categorize the 50 cases:

| Type | Count | Mentioned | Differential |
|------|-------|-----------|-------------|
| Diagnostic | 36 | 14 (39%) | 5 (14%) |
| Treatment | 6 | — | — |
| Pathophysiology | 6 | — | — |
| Statistics | 1 | — | — |
| Anatomy | 1 | — | — |

### Key Observations

1. **MedQA includes many non-diagnostic questions** (treatment, mechanism, stats) that the CDS pipeline is not designed to answer — it generates differential diagnoses, not multiple-choice answers.
2. **On diagnostic questions specifically**, 39% mentioned accuracy is reasonable for a pipeline that wasn't optimized for exam-style questions.
3. **Pipeline failures (3/50)** were caused by the HF endpoint scaling to zero mid-run. The `--resume` flag successfully continued from the checkpoint.
4. **Improved clinical reasoning prompt** to demand disease-level diagnoses rather than symptom categories (e.g., "Chikungunya" not "viral arthritis").

### Infrastructure Improvements

- **Incremental JSONL checkpoints:** Each case result is appended to `medqa_checkpoint.jsonl` as it completes.
- **`--resume` flag:** Skips already-completed cases, enabling graceful recovery from endpoint failures.
- **`check_progress.py`:** Utility to monitor checkpoint progress during long runs.
- **`analyze_results.py`:** Categorizes MedQA results by question type for more meaningful accuracy analysis.
- **Unicode fixes:** Replaced box-drawing characters (`╔═╗║╚╝`) and symbols (`✓✗─`) with ASCII equivalents for Windows console compatibility.

**Files created:** `validation/analyze_results.py`, `validation/check_progress.py`  
**Files modified:** `validation/base.py`, `validation/harness_medqa.py`, `validation/run_validation.py`, `app/tools/clinical_reasoning.py`, `app/tools/synthesis.py`, `app/tools/conflict_detection.py`, `app/tools/patient_parser.py`

---

## Phase 10: Final Documentation Audit & Cleanup

Performed a full accuracy audit of all 5 documentation files and `test_e2e.py`.

**Issues found and fixed:**
- README.md: step count said "5" in E2E table (fixed to 6), missing Conflict Detection row, missing `validation/` in project structure, missing validation section and test commands
- architecture.md: Design Decision #1 said "5-step" (fixed to 6), Decision #4 said "Gemma in two roles" (fixed to four), no validation framework section
- test_results.md: no external validation section, stale line count for test_e2e.py
- DEVELOPMENT_LOG.md: Phase 7 said "(Current)", missing Phase 9 for validation framework
- writeup_draft.md: referenced "confidence levels" (removed earlier), placeholder links, no validation methodology
- test_e2e.py: no assertions on step count or conflict_detection step

**Created:** `TODO.md` in project root with next-session action items for easy pickup by future contributors or AI instances.
