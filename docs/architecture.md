# Clinical Decision Support Agent — Architecture

## The Problem

**Current workflow (painful, error-prone):**
A clinician sees a patient → manually reviews the chart, labs, medications → searches
UpToDate or reference materials → checks drug interactions → mentally synthesizes all
information → makes clinical decisions. This is slow, cognitively taxing, and mistakes
happen when clinicians are fatigued or overloaded.

**Agent-reimagined workflow:**
Patient data goes in → an orchestrated agent pipeline automatically gathers context,
reasons about the case, checks interactions, retrieves guidelines, and produces a
structured clinical decision support report — all in seconds.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FRONTEND (Next.js 14 + React)                │
│  PatientInput.tsx  │  AgentPipeline.tsx  │  CDSReport.tsx       │
│  3 sample cases    │  Real-time step viz │  Full report render  │
└──────────────────────────┬──────────────────────────────────────┘
                           │ REST API (port 3000 → proxy)
                           │ WebSocket (direct to backend)
┌──────────────────────────▼──────────────────────────────────────┐
│                  BACKEND (FastAPI + Python 3.10)                 │
│  Port 8000 (default) / 8002 (dev)                               │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │            ORCHESTRATOR (orchestrator.py, ~300 lines)              │  │
│  │  Sequential 6-step pipeline with structured state passing         │  │
│  └──┬──────────┬──────────┬──────────┬──────────┬──────────┬────────┘  │
│     │          │          │          │          │          │            │
│  ┌──▼───┐ ┌───▼────┐ ┌──▼───┐ ┌───▼────┐ ┌───▼─────┐ ┌──▼──────┐   │
│  │Step 1│ │Step 2  │ │Step 3│ │Step 4  │ │Step 5   │ │Step 6   │   │
│  │Pati- │ │Clini-  │ │Drug  │ │Guide-  │ │Conflict │ │Synthe-  │   │
│  │ent   │ │cal     │ │Inter-│ │line    │ │Detect-  │ │sis      │   │
│  │Parser│ │Reason- │ │action│ │Retriev-│ │ion      │ │Agent    │   │
│  │(LLM) │ │ing     │ │(APIs)│ │al(RAG) │ │(LLM)    │ │(LLM)    │   │
│  └──────┘ │(LLM)   │ └──┬──┘ └──┬─────┘ └─────────┘ └─────────┘   │
│           └────────┘    │       │                                    │
│                        ┌────▼────┐ ┌─▼──────────────┐           │
│                        │OpenFDA  │ │ChromaDB         │           │
│                        │RxNorm   │ │62 guidelines    │           │
│                        │NLM API  │ │14 specialties   │           │
│                        └─────────┘ │MiniLM-L6-v2     │           │
│                                    └─────────────────┘           │
└──────────────────────────────────────────────────────────────────┘

LLM: google/medgemma-27b-text-it via HuggingFace Dedicated Endpoint
     (OpenAI-compatible TGI, 1× A100 80 GB, bfloat16)
```

---

## Agent Pipeline — Step-by-Step

### Step 1: Patient Data Parser (`patient_parser.py`)

- **Input:** Raw patient case free-text
- **Output:** `PatientProfile` (Pydantic model)
- **Method:** LLM extraction with structured JSON output
- **Fields extracted:** Demographics, chief complaint, HPI, vital signs, labs, medications, allergies, past medical history, social history
- **Timing:** ~7.8 s (observed)

### Step 2: Clinical Reasoning Agent (`clinical_reasoning.py`)

- **Input:** `PatientProfile` from Step 1
- **Output:** `ClinicalReasoningResult` with ranked differential diagnosis
- **Method:** Chain-of-thought prompting for transparent reasoning
- **Key outputs:** Ranked `DiagnosisCandidate` list (name, likelihood, key findings for/against), risk assessment, recommended workup
- **Timing:** ~21.2 s (observed)

### Step 3: Drug Interaction Check (`drug_interactions.py`)

- **Input:** Medication list from Step 1 + any proposed medications from Step 2
- **Output:** `DrugInteractionResult` with interaction warnings
- **Method:** Two-API approach:
  1. **RxNorm / NLM API** — Normalize medication names to RxCUI identifiers, check pairwise interactions
  2. **OpenFDA API** — Query drug adverse event reports for additional safety data
- **Bug fix applied:** RxNorm API returns `rxnormId` as a list, not a scalar — code handles both formats
- **Timing:** ~11.3 s (observed)

### Step 4: Guideline Retrieval — RAG (`guideline_retrieval.py`)

- **Input:** Primary diagnosis/conditions from Step 2
- **Output:** `GuidelineRetrievalResult` with relevant guideline excerpts and citations
- **Method:** Retrieval-Augmented Generation over a curated guideline corpus
- **RAG details:**
  - **Vector store:** ChromaDB `PersistentClient` (persist dir: `./data/chroma`)
  - **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
  - **Corpus:** 62 clinical guidelines from `clinical_guidelines.json`
  - **Specialties:** 14 (Cardiology, EM, Endocrinology, Pulmonology, Neurology, GI, ID, Psychiatry, Pediatrics, Nephrology, Hematology, Rheumatology, OB/GYN, Preventive/Other)
  - **Metadata:** `specialty`, `guideline_id` stored per document in ChromaDB
  - **Similarity:** Cosine similarity, top-k retrieval (k=5 default)
  - **Sources:** ACC/AHA, ADA, GOLD, GINA, IDSA, ACOG, AAN, APA, AAP, ACR, ASH, KDIGO, WHO, USPSTF, and others
  - **Fallback:** If `clinical_guidelines.json` is missing, falls back to 2 minimal embedded guidelines
- **Timing:** ~9.6 s (observed)

### Step 5: Conflict Detection (`conflict_detection.py`)

- **Input:** Patient profile, clinical reasoning, drug interactions, and retrieved guidelines from Steps 1–4
- **Output:** `ConflictDetectionResult` with specific `ClinicalConflict` items
- **Method:** LLM-based comparison of guideline recommendations against the patient's actual data
- **Conflict types detected:**
  - **Omission** — Guideline recommends something the patient is not receiving
  - **Contradiction** — Patient's current treatment conflicts with guideline advice
  - **Dosage** — Guideline specifies dose adjustments that apply to this patient (age, renal function, etc.)
  - **Monitoring** — Guideline requires monitoring that is not documented as ordered
  - **Allergy Risk** — Guideline-recommended treatment involves a medication the patient is allergic to
  - **Interaction Gap** — Known drug interaction is not addressed in the care plan
- **Each conflict includes:** severity (critical/high/moderate/low), guideline source, guideline text, patient data, description, and suggested resolution
- **Temperature:** 0.1 (low, for safety-critical analysis)
- **Graceful degradation:** Returns empty result if no guidelines were retrieved (Step 4 skipped/failed)

### Step 6: Synthesis Agent (`synthesis.py`)

- **Input:** All outputs from Steps 1–4 plus conflict detection results
- **Output:** `CDSReport` (comprehensive structured report)
- **Report sections:**
  - Patient summary
  - Differential diagnosis with reasoning chains
  - Drug interaction warnings with severity
  - **Conflicts & gaps** — prominently featured with guideline vs patient data comparison
  - Guideline-concordant recommendations with citations
  - Suggested next steps (immediate, short-term, long-term)
  - Caveats and limitations
- **Timing:** ~25.3 s (observed)

**Total pipeline time:** ~75–85 s for a complex case (6 steps, with Steps 3–4 parallel).

---

## LLM Integration — Implementation Details

### Model Configuration

- **Model:** `google/medgemma-27b-text-it` (MedGemma from HAI-DEF)
- **API:** HuggingFace Dedicated Endpoint (TGI), with Google AI Studio as fallback
- **Base URL:** `https://lisvpf8if1yhgxn2.us-east-1.aws.endpoints.huggingface.cloud/v1` (HF Endpoint)
- **Client:** OpenAI Python SDK (`openai==1.51.0`)
- **Service:** `medgemma.py` wraps all LLM calls
- **Endpoint config:** `MAX_INPUT_TOKENS=12288`, `MAX_TOTAL_TOKENS=16384`, `DTYPE=bfloat16`

### Gemma System Prompt Handling

**MedGemma via TGI** natively supports `role: "system"` messages, so we send system/user messages properly.

**Fallback for Google AI Studio:** If the backend happens to be plain Gemma on Google AI Studio (which rejects the system role), the code automatically catches the error and falls back to folding the system prompt into the first user message:

```python
# If system message exists, fold it into the first user message
if messages[0]["role"] == "system":
    system_content = messages[0]["content"]
    messages = messages[1:]
    if messages and messages[0]["role"] == "user":
        messages[0]["content"] = f"[System Instructions]\n{system_content}\n\n{messages[0]['content']}"
```

This preserves the intended behavior while staying compatible with Gemma's API constraints.

---

## Data Models (Pydantic v2)

All pipeline data is strongly typed via Pydantic models in `schemas.py` (~280 lines):

| Model | Purpose |
|-------|---------|
| `CaseSubmission` | Input: patient text + feature flags |
| `PatientProfile` | Step 1 output: demographics, vitals, labs, meds, history |
| `DiagnosisCandidate` | Individual diagnosis with likelihood + evidence |
| `ClinicalReasoningResult` | Step 2 output: ranked differentials + workup |
| `DrugInteraction` | Individual drug interaction warning |
| `DrugInteractionResult` | Step 3 output: all interaction data |
| `GuidelineExcerpt` | Individual guideline citation |
| `GuidelineRetrievalResult` | Step 4 output: relevant guidelines |
| `ConflictType` | Enum: omission, contradiction, dosage, monitoring, allergy_risk, interaction_gap |
| `ClinicalConflict` | Individual conflict: guideline_text vs patient_data + suggested resolution |
| `ConflictDetectionResult` | Step 5 output: all detected conflicts |
| `CDSReport` | Step 6 output: full synthesized report (now includes conflicts) |
| `AgentStep` | WebSocket message: step name, status, data, timing |

---

## Frontend Architecture

### Technology

- **Framework:** Next.js 14 (App Router)
- **UI:** React 18 + TypeScript + Tailwind CSS
- **State:** React hooks + WebSocket for real-time updates

### Components

| Component | Role |
|-----------|------|
| `PatientInput.tsx` | Text area for patient case + 3 pre-loaded sample cases (chest pain, DKA, pediatric fever) |
| `AgentPipeline.tsx` | Visualizes the 6-step pipeline in real time — shows status (pending / running / complete / error) for each step as WebSocket messages arrive |
| `CDSReport.tsx` | Renders the final CDS report: patient summary, differentials, drug warnings, **conflicts & gaps** (prominently styled), guidelines, next steps |

### Communication

- **REST API:** `POST /api/cases/submit` (submit case), `GET /api/cases/{id}` (poll results)
- **WebSocket:** `ws://localhost:8000/ws/agent` — receives `AgentStep` messages as each pipeline step starts/completes
- **Proxy:** `next.config.js` proxies `/api/` requests to the backend

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check — returns backend status |
| `/api/cases/submit` | POST | Submit a `CaseSubmission` for analysis |
| `/api/cases/{case_id}` | GET | Poll for case results |
| `/api/cases` | GET | List all submitted cases |
| `/ws/agent` | WebSocket | Real-time pipeline step streaming |

---

## External API Dependencies

| API | Purpose | Authentication | Rate Limits |
|-----|---------|---------------|-------------|
| HuggingFace Dedicated Endpoint | MedGemma 27B Text IT LLM inference | HF API token | Dedicated GPU (no shared limits) |
| Google AI Studio (fallback) | Gemma 3 27B IT LLM inference | API key | Per-key quota |
| OpenFDA | Drug adverse event data | None (public) | 240 req/min (with key), 40/min (without) |
| RxNorm / NLM | Drug normalization (name → RxCUI), pairwise interactions | None (public) | 20 req/sec |

---

## Why This Is Agentic (Not Just a Chatbot)

| Characteristic | Chatbot | This Agent System |
|----------------|---------|-------------------|
| Tool use | None | 5+ specialized tools (parser, drug API, RAG, conflict detection, synthesis) |
| Planning | None | Orchestrator executes a defined 6-step plan |
| State management | Stateless | Patient context flows through all steps |
| Error handling | Generic | Tool-specific fallbacks, graceful degradation |
| Output structure | Free text | Pydantic-validated, structured, cited |
| Transparency | Black box | Shows each reasoning step + tool outputs in real time |
| External data | None | Queries 3 external data sources (OpenFDA, RxNorm, ChromaDB) |

---

## Key Design Decisions

1. **Custom orchestrator over LangChain/LlamaIndex** — Simpler, more transparent, easier to debug. We control the pipeline loop explicitly. No framework overhead for a sequential 6-step pipeline.

2. **WebSocket for agent activity** — The frontend shows each step as it happens (parsing → reasoning → checking → retrieving → synthesizing). This real-time visibility is critical for clinician trust.

3. **Structured outputs everywhere** — Every tool returns a Pydantic model. The synthesis agent receives structured data, not messy text. This ensures consistency and enables frontend rendering.

4. **Gemma in four roles** — As the patient parser (Step 1), clinical reasoning engine (Step 2), conflict detector (Step 5), and synthesis engine (Step 6). The same model extracts structured data, reasons about the case, identifies guideline-vs-patient conflicts, and integrates all tool outputs into a coherent report.

5. **Graceful degradation** — If a tool fails (e.g., OpenFDA is down), the agent continues with available information and notes the gap in the final report.

6. **Curated guideline corpus over general web search** — 62 hand-selected guidelines from authoritative sources (ACC/AHA, ADA, etc.) ensure quality and citability. Better than scraping the web.

7. **ChromaDB for simplicity** — Embedded vector DB that persists locally. No external database service to manage. Rebuilds in seconds from the JSON source.

---

## Configuration

All configuration lives in `config.py` (Pydantic Settings) and `.env`:

| Setting | Default | Description |
|---------|---------|-------------|
| `MEDGEMMA_API_KEY` | (required) | HuggingFace API token or Google AI Studio API key |
| `MEDGEMMA_BASE_URL` | `""` (empty) | LLM API endpoint (HF Endpoint URL with /v1, or Google AI Studio URL) |
| `MEDGEMMA_MODEL_ID` | `google/medgemma` | Model identifier (`tgi` for HF Endpoints, or full model name) |
| `HF_TOKEN` | `""` | HuggingFace token for dataset downloads |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | ChromaDB storage directory |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model for RAG |
| `MAX_GUIDELINES` | `5` | Number of guidelines to retrieve per query |
| `AGENT_TIMEOUT` | `120` | Max seconds for full pipeline execution |

---

## Known Limitations

- **LLM latency:** Full pipeline takes ~75 s due to multiple sequential LLM calls. Could be improved with smaller models or parallel LLM calls.
- **No authentication:** No user auth — designed as a local demo / research tool.
- **Single-model:** Uses only MedGemma 27B Text IT. Could benefit from specialized models for different steps.
- **Guideline currency:** Guidelines are a static snapshot. A production system would need automated updates.
- **No EHR integration:** Input is manual text paste. A production system would integrate with EHR FHIR APIs.

---

## Validation Framework

The project includes an external dataset validation framework that tests the full pipeline against real-world clinical data — bypassing the HTTP server and calling the `Orchestrator` directly.

### Datasets

| Dataset | Source | Cases | What It Measures |
|---------|--------|-------|------------------|
| **MedQA (USMLE)** | HuggingFace (`GBaker/MedQA-USMLE-4-options`) | 1,273 | Diagnostic accuracy — top-1, top-3, mentioned |
| **MTSamples** | GitHub (`socd06/medical-nlp`) | ~5,000 | Parse quality, field completeness, specialty alignment |
| **PMC Case Reports** | PubMed E-utilities (esearch + efetch) | Dynamic | Diagnostic accuracy on published cases with known diagnoses |

### Architecture

```
validation/
├── base.py               # ValidationCase, ValidationResult, ValidationSummary
│                         # run_cds_pipeline() — direct Orchestrator invocation
│                         # fuzzy_match(), diagnosis_in_differential()
├── harness_medqa.py      # Fetch from HuggingFace, extract vignettes, score diagnostics
├── harness_mtsamples.py  # Fetch CSV, stratified sampling, score parse quality
├── harness_pmc.py        # PubMed E-utilities, title-based diagnosis extraction
└── run_validation.py     # Unified CLI: --medqa --mtsamples --pmc --all --max-cases N
```

All datasets are cached locally in `validation/data/` (gitignored). Results are saved to `validation/results/` (also gitignored).
