# CDS Agent — Project Writeup

> Competition writeup template filled in with actual project details.  
> Also serves as the primary project summary document.

---

### Project name

**CDS Agent** — Agentic Clinical Decision Support System

### Your team

| Name | Specialty | Role |
|------|-----------|------|
| (Developer) | Software Engineering / AI | Full-stack development, agent architecture, RAG system, testing |

### Problem statement

**The Problem:**

Clinical decision-making is one of the most cognitively demanding tasks in medicine. A clinician seeing a patient must simultaneously: review the patient's history and current presentation, mentally generate a differential diagnosis, recall drug interactions for current and proposed medications, remember relevant clinical guidelines, and synthesize all of this into a coherent care plan — often while fatigued, time-pressured, and managing multiple patients.

Medical errors remain a leading cause of patient harm. Studies estimate that diagnostic errors affect approximately 12 million Americans annually, and medication errors harm over 1.5 million people per year. Many of these errors stem not from lack of knowledge, but from the cognitive burden of integrating information from multiple sources under time pressure.

**Who is affected:**

- **Clinicians** (primary users) — physicians, nurse practitioners, physician assistants in emergency departments, urgent care, and inpatient settings where rapid, comprehensive decision-making is critical
- **Patients** — who benefit from more thorough, evidence-based care with fewer diagnostic and medication errors
- **Health systems** — which bear the cost of medical errors, readmissions, and liability

**Why AI is the right solution:**

This problem cannot be solved with traditional rule-based systems because:
1. Clinical reasoning requires understanding free-text narratives, not just coded data
2. Differential diagnosis generation requires probabilistic reasoning over thousands of conditions
3. Guideline retrieval requires semantic understanding of clinical context
4. Synthesis requires integrating heterogeneous data (structured labs, free-text guidelines, API-sourced drug data) into coherent recommendations

Large language models — specifically medical-domain models like Gemma — can perform all of these tasks. But a single LLM call is insufficient. The agent architecture orchestrates the LLM across multiple specialized steps, augmented with external tools (drug APIs, RAG) to produce a result that no single component could achieve alone.

**Impact potential:**

If deployed, this system could:
- Reduce diagnostic error rates by providing systematic differential diagnosis generation for every patient encounter
- Catch drug interactions that clinicians might miss, especially in polypharmacy patients
- Ensure guideline-concordant care by surfacing relevant, current clinical guidelines at the point of care
- Save clinician time by automating the information-gathering and synthesis steps of clinical reasoning

Estimated reach: There are approximately 140 million ED visits per year in the US alone. Even a modest improvement in diagnostic accuracy or medication safety across a fraction of these encounters would represent significant impact.

### Overall solution

**HAI-DEF models used:**

- **MedGemma** (`google/medgemma-27b-text-it`) — Google's medical-domain model from the Health AI Developer Foundations (HAI-DEF) collection
- Development/validation also performed with **Gemma 3 27B IT** (`gemma-3-27b-it`) via Google AI Studio for rapid iteration

**Why MedGemma:**

MedGemma is purpose-built for medical applications and is part of Google's HAI-DEF collection:
- Trained specifically for health and biomedical tasks, providing stronger clinical reasoning than general-purpose models
- Open-weight model that can be self-hosted for HIPAA compliance in production
- Large enough (27B parameters) for complex chain-of-thought clinical reasoning
- Designed to be the foundation for healthcare AI applications — exactly what this competition demands

**How the model is used:**

The model serves as the reasoning engine in a 6-step agentic pipeline:

1. **Patient Data Parsing** (LLM) — Extracts structured patient data from free-text clinical narratives
2. **Clinical Reasoning** (LLM) — Generates ranked differential diagnoses with chain-of-thought reasoning
3. **Drug Interaction Check** (External APIs) — Queries OpenFDA and RxNorm for medication safety
4. **Guideline Retrieval** (RAG) — Retrieves relevant clinical guidelines from a 62-guideline corpus using ChromaDB
5. **Conflict Detection** (LLM) — Compares guideline recommendations against patient data to identify omissions, contradictions, dosage concerns, monitoring gaps, allergy risks, and interaction gaps
6. **Synthesis** (LLM) — Integrates all outputs into a comprehensive CDS report with conflicts prominently featured

The model is used in Steps 1, 2, 5, and 6 — parsing, reasoning, conflict detection, and synthesis. This demonstrates the model used "to its fullest potential" across multiple distinct clinical tasks within a single workflow.

### Technical details

**Architecture:**

```
Frontend (Next.js 14)  ←→  Backend (FastAPI + Python 3.10)
                              │
                    Orchestrator (6-step pipeline)
                    ├── Step 1: Patient Parser (LLM)
                    ├── Step 2: Clinical Reasoning (LLM)
                    ├── Step 3: Drug Check (OpenFDA + RxNorm APIs)
                    ├── Step 4: Guideline Retrieval (ChromaDB RAG)
                    ├── Step 5: Conflict Detection (LLM)
                    └── Step 6: Synthesis (LLM)
```

All inter-step data is strongly typed with Pydantic v2 models. The pipeline streams each step's progress to the frontend via WebSocket for real-time visibility.

**Fine-tuning:**

No fine-tuning was performed in the current version. The base MedGemma model (`medgemma-27b-text-it`) was used with carefully crafted prompt engineering for each pipeline step. Fine-tuning on clinical reasoning datasets is a planned improvement.

**Performance analysis:**

| Test | Result |
|------|--------|
| E2E pipeline (chest pain / ACS) | All 6 steps passed, ~75–85 s total |
| RAG retrieval quality | 30/30 queries passed (100%), avg relevance 0.639 |
| Clinical test suite | 22 scenarios across 14 specialties |
| Top-1 RAG accuracy | 100% — correct guideline ranked #1 for all queries |
| **MedQA 50-case validation** | **36% top-1, 38% top-3, 38% mentioned, 94% pipeline success** |
| MedQA diagnostic-only (36 cases) | 39% mentioned, 14% differential |

**Application stack:**

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 14, React 18, TypeScript, Tailwind CSS |
| Backend | FastAPI, Python 3.10, Pydantic v2, WebSocket |
| LLM | MedGemma 27B Text IT (HAI-DEF) + Gemma 3 27B IT for dev |
| RAG | ChromaDB + sentence-transformers (all-MiniLM-L6-v2) |
| Drug Data | OpenFDA API, RxNorm / NLM API |

**Deployment considerations:**

- **HIPAA compliance:** MedGemma is an open-weight model that can be self-hosted on-premises, eliminating the need to send patient data to external APIs. This is critical for healthcare deployment.
- **Latency:** Current pipeline takes ~75 s for a single E2E case (local), or ~204 s avg on the HuggingFace Dedicated Endpoint (50-case MedQA validation). For production, this could be reduced with: smaller/distilled models, parallel LLM calls, or GPU-accelerated inference with higher throughput.
- **Scalability:** FastAPI + uvicorn supports async request handling. For high-throughput deployment, add worker processes and a task queue (e.g., Celery).
- **EHR integration:** Current input is manual text paste. A production system would integrate with EHR systems via FHIR APIs for automatic patient data extraction.

### Validation methodology

The project includes an external dataset validation framework (`src/backend/validation/`) that tests the full pipeline against real-world clinical data:

| Dataset | Source | What It Tests |
|---------|--------|---------------|
| **MedQA (USMLE)** | HuggingFace (1,273 test cases) | Diagnostic accuracy — does the pipeline's top differential match the USMLE correct answer? |
| **MTSamples** | GitHub (~5,000 medical transcriptions) | Parse quality, field completeness, specialty alignment on real clinical notes |
| **PMC Case Reports** | PubMed E-utilities (dynamic) | Diagnostic accuracy on published case reports with known diagnoses |

The validation harness calls the `Orchestrator` directly (no HTTP server), enabling rapid batch testing. Each dataset has a dedicated harness that fetches data, converts it to patient narratives, runs the pipeline, and scores the output against ground truth.

**Initial smoke test (3 MedQA cases):** 100% parse success, 66.7% top-1 diagnostic accuracy, ~94 s avg per case.

**50-case MedQA validation (MedGemma 27B via HF Endpoint):** 94% pipeline success, 36% top-1 diagnostic accuracy, 38% mentioned in report, 204 s avg per case. On diagnostic-only questions (36/50), 39% mentioned the correct diagnosis. Full results in [docs/test_results.md](docs/test_results.md).

**Practical usage:**

In a real clinical setting, the system would be used at the point of care:
1. Clinician opens the CDS Agent interface (embedded in the EHR or as a standalone app)
2. Patient data is automatically pulled from the EHR (or pasted manually)
3. The agent pipeline runs in ~60–90 seconds, during which the clinician can continue other tasks
4. The CDS report appears with:
   - Ranked differential diagnoses with reasoning chains (transparent AI)
   - Drug interaction warnings with severity levels
   - **Conflicts & gaps** between guideline recommendations and the patient's actual data — prominently displayed with specific guideline citations, patient data comparisons, and suggested resolutions
   - Relevant clinical guideline excerpts with citations to authoritative sources
   - Suggested next steps (immediate, short-term, long-term)
5. The clinician reviews the recommendations and incorporates them into their clinical judgment

The system is explicitly designed as a **decision support** tool, not a decision-making tool. All recommendations include caveats and limitations. The clinician retains full authority over patient care.

---

**Links:**

- Video: [To be recorded]
- Code Repository: [github.com/bshepp/clinical-decision-support-agent](https://github.com/bshepp/clinical-decision-support-agent)
- Live Demo: [To be deployed]
- Hugging Face Model: [google/medgemma-27b-text-it](https://huggingface.co/google/medgemma-27b-text-it)
