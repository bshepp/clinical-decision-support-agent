# CDS Agent — Clinical Decision Support System

> An agentic clinical decision support application that orchestrates medical AI with specialized tools to assist clinicians in real time.

**Origin:** [MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge) (Kaggle / Google Research)  
**Focus:** Building a genuinely impactful medical application — not just a competition entry.

---

## What It Does

A clinician pastes a patient case. The system automatically:

1. **Parses** the free-text into structured patient data (demographics, vitals, labs, medications, history)
2. **Reasons** about the case to generate a ranked differential diagnosis with chain-of-thought transparency
3. **Checks drug interactions** against OpenFDA and RxNorm databases
4. **Retrieves clinical guidelines** from a 62-guideline RAG corpus spanning 14 medical specialties
5. **Detects conflicts** between guideline recommendations and the patient's actual data — surfacing omissions, contradictions, dosage concerns, and monitoring gaps
6. **Synthesizes** everything into a structured CDS report with recommendations, warnings, conflicts, and citations

All six steps stream to the frontend in real time via WebSocket — the clinician sees each step execute live.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FRONTEND (Next.js 14 + React)                    │
│  Patient Case Input  │  Agent Activity Feed  │  CDS Report View    │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ REST API + WebSocket
┌──────────────────────────▼──────────────────────────────────────────┐
│                     BACKEND (FastAPI + Python 3.10)                  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                ORCHESTRATOR (6-Step Pipeline)                  │  │
│  └──┬──────────┬──────────┬──────────┬──────────┬──────────┬─────┘  │
│  ┌──▼───┐ ┌───▼────┐ ┌──▼───┐ ┌───▼────┐ ┌───▼─────┐ ┌──▼────┐  │
│  │Parse │ │Reason  │ │ Drug │ │  RAG   │ │Conflict │ │Synth- │  │
│  │Pati- │ │(LLM)   │ │Check │ │Guide-  │ │Detect-  │ │esize  │  │
│  │ent   │ │Differ- │ │OpenFDA│ │lines   │ │ion      │ │(LLM)  │  │
│  │Data  │ │ential  │ │RxNorm │ │ChromaDB│ │(LLM)    │ │Report │  │
│  └──────┘ └────────┘ └──────┘ └────────┘ └─────────┘ └───────┘  │
│                                                                      │
│  External: OpenFDA API │ RxNorm/NLM API │ ChromaDB (local)          │
└──────────────────────────────────────────────────────────────────────┘
```

See [docs/architecture.md](docs/architecture.md) for the full design document.

---

## Verified Test Results

### Full Pipeline E2E Test (Chest Pain / ACS Case)

All 5 pipeline steps completed successfully:

| Step | Duration | Result |
|------|----------|--------|
| Parse Patient Data | 7.8 s | Structured profile extracted |
| Clinical Reasoning | 21.2 s | ACS correctly identified as top differential |
| Drug Interaction Check | 11.3 s | Interactions queried against OpenFDA / RxNorm |
| Guideline Retrieval (RAG) | 9.6 s | Relevant cardiology guidelines retrieved |
| Synthesis | 25.3 s | Comprehensive CDS report generated |

### RAG Retrieval Quality Test

**30 / 30 queries passed (100%)** across all 14 specialties:

| Metric | Value |
|--------|-------|
| Queries tested | 30 |
| Pass rate | 100% (30/30) |
| Avg relevance score | 0.639 |
| Min relevance score | 0.519 |
| Max relevance score | 0.765 |
| Top-1 accuracy | 100% (correct guideline ranked #1 for every query) |

Full results: [docs/test_results.md](docs/test_results.md)

### Clinical Test Suite

22 comprehensive clinical scenarios covering: ACS, AFib, heart failure, stroke, sepsis, anaphylaxis, polytrauma, DKA, thyroid storm, adrenal crisis, massive PE, status asthmaticus, GI bleeding, pancreatitis, status epilepticus, meningitis, suicidal ideation, neonatal fever, pediatric dehydration, hyperkalemia, acetaminophen overdose, and elderly polypharmacy with falls.

---

## RAG Clinical Guidelines Corpus

**62 clinical guidelines** across **14 medical specialties**, stored in ChromaDB with sentence-transformer embeddings (`all-MiniLM-L6-v2`):

| Specialty | Count | Key Topics |
|-----------|-------|------------|
| Cardiology | 8 | HTN, chest pain / ACS, HF, AFib, lipids, NSTEMI, PE, valvular disease |
| Emergency Medicine | 10 | Stroke, sepsis, trauma, anaphylaxis, burns, ACLS, seizures, toxicology, hyperkalemia, acute abdomen |
| Endocrinology | 7 | DM management, DKA, thyroid, adrenal insufficiency, osteoporosis, hypoglycemia, hypercalcemia |
| Pulmonology | 4 | COPD, asthma, CAP, pleural effusion |
| Neurology | 4 | Epilepsy, migraine, MS, meningitis |
| Gastroenterology | 5 | Upper GI bleed, pancreatitis, cirrhosis, IBD, CRC screening |
| Infectious Disease | 5 | STIs, UTI, HIV, SSTIs, COVID-19 |
| Psychiatry | 4 | MDD, suicide risk, GAD, substance use |
| Pediatrics | 4 | Fever without source, asthma, dehydration, neonatal jaundice |
| Nephrology | 2 | CKD, AKI |
| Hematology | 2 | VTE, sickle cell |
| Rheumatology | 2 | RA, gout |
| OB/GYN | 2 | Hypertensive disorders of pregnancy, postpartum hemorrhage |
| Other | 3+ | Preventive medicine (USPSTF), perioperative cardiac risk, dermatology (melanoma) |

Sources include ACC/AHA, ADA, GOLD, GINA, IDSA, ACOG, AAN, APA, AAP, ACR, ASH, KDIGO, WHO, and other major guideline organizations.

---

## Project Structure

```
medgemma_impact_challenge/
├── README.md                           # This file
├── DEVELOPMENT_LOG.md                  # Chronological build history & decisions
├── SUBMISSION_GUIDE.md                 # Competition submission strategy
├── RULES_SUMMARY.md                    # Competition rules checklist
├── docs/
│   ├── architecture.md                 # System architecture & design decisions
│   ├── test_results.md                 # Detailed test results & benchmarks
│   └── writeup_draft.md               # Project writeup / summary
├── src/
│   ├── backend/                        # Python FastAPI backend
│   │   ├── .env.template              # Environment config template
│   │   ├── .env                       # Local config (not committed)
│   │   ├── requirements.txt           # Python dependencies (28 packages)
│   │   ├── test_e2e.py               # End-to-end pipeline test
│   │   ├── test_clinical_cases.py    # 22 clinical scenario test suite
│   │   ├── test_rag_quality.py       # RAG retrieval quality tests (30 queries)
│   │   ├── test_poll.py              # Simple case poller utility
│   │   └── app/
│   │       ├── main.py               # FastAPI entry (CORS, routers, lifespan)
│   │       ├── config.py             # Pydantic Settings (ports, models, dirs)
│   │       ├── __init__.py
│   │       ├── models/
│   │       │   └── schemas.py        # All Pydantic models (~280 lines)
│   │       ├── agent/
│   │       │   └── orchestrator.py   # 6-step pipeline orchestrator (~300 lines)
│   │       ├── services/
│   │       │   └── medgemma.py       # LLM service (OpenAI-compatible API)
│   │       ├── tools/
│   │       │   ├── patient_parser.py      # Step 1: Free-text → structured data
│   │       │   ├── clinical_reasoning.py  # Step 2: Differential diagnosis
│   │       │   ├── drug_interactions.py   # Step 3: OpenFDA + RxNorm
│   │       │   ├── guideline_retrieval.py # Step 4: RAG over ChromaDB
│   │       │   ├── conflict_detection.py  # Step 5: Guideline vs patient conflicts
│   │       │   └── synthesis.py           # Step 6: CDS report generation
│   │       ├── data/
│   │       │   └── clinical_guidelines.json  # 62 guidelines, 14 specialties
│   │       └── api/
│   │           ├── health.py         # GET /api/health
│   │           ├── cases.py          # POST /api/cases/submit, GET /api/cases/{id}
│   │           └── ws.py            # WebSocket /ws/agent
│   └── frontend/                     # Next.js 14 + React 18 + TypeScript
│       ├── package.json
│       ├── next.config.js            # API proxy → backend
│       ├── tailwind.config.js
│       └── src/
│           ├── app/
│           │   ├── layout.tsx
│           │   ├── page.tsx          # Main CDS interface
│           │   └── globals.css
│           ├── components/
│           │   ├── PatientInput.tsx   # Patient case input + 3 sample cases
│           │   ├── AgentPipeline.tsx  # Real-time step visualization
│           │   └── CDSReport.tsx     # Final report renderer
│           └── hooks/
│               └── useAgentWebSocket.ts  # WebSocket state management
├── notebooks/                        # Experiment notebooks
├── models/                           # Fine-tuned models (future)
└── demo/                             # Video & demo assets
```

---

## Quick Start

### Prerequisites

- **Python 3.10+** (tested with Python 3.10)
- **Node.js 18+** (tested with Node.js 18)
- **API Key:** Google AI Studio API key for Gemma model access

### Backend Setup

```bash
cd src/backend

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.template .env        # Windows (or: cp .env.template .env)
# Edit .env — set MEDGEMMA_API_KEY to your Google AI Studio key

# Start the backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
cd src/frontend

npm install
npm run dev
# Open http://localhost:3000
```

> **Note:** The frontend proxies API requests to the backend. If using a non-default port, update `next.config.js` and `src/hooks/useAgentWebSocket.ts` accordingly.

### Running Tests

```bash
cd src/backend

# RAG retrieval quality test (no backend needed)
python test_rag_quality.py --rebuild --verbose

# Full pipeline E2E test (requires running backend)
python test_e2e.py

# Comprehensive clinical test suite (requires running backend)
python test_clinical_cases.py --list              # See all 22 cases
python test_clinical_cases.py --case em_sepsis    # Run one case
python test_clinical_cases.py --specialty Cardio   # Run by specialty
python test_clinical_cases.py                      # Run all cases
python test_clinical_cases.py --report results.json  # Save results
```

### Usage

1. Open `http://localhost:3000`
2. Paste a patient case description (or click a sample case)
3. Click **"Analyze Patient Case"**
4. Watch the 6-step agent pipeline execute in real time
5. Review the CDS report: differential diagnosis, drug warnings, **conflicts & gaps**, guideline recommendations, next steps

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | Next.js 14, React 18, TypeScript, Tailwind CSS | Patient input, pipeline visualization, report display |
| API | FastAPI, WebSocket, Pydantic v2 | REST endpoints + real-time streaming |
| LLM | Gemma 3 27B IT (via Google AI Studio) | Clinical reasoning + synthesis |
| RAG | ChromaDB, sentence-transformers (all-MiniLM-L6-v2) | Clinical guideline retrieval |
| Drug Data | OpenFDA API, RxNorm / NLM API | Drug interactions, medication normalization |
| Validation | Pydantic | Structured output validation across all pipeline steps |

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/cases/submit` | POST | Submit a patient case for analysis |
| `/api/cases/{case_id}` | GET | Get case results (poll for completion) |
| `/api/cases` | GET | List all cases |
| `/ws/agent` | WebSocket | Real-time pipeline step streaming |

### Submit a Case (REST)

```bash
curl -X POST http://localhost:8000/api/cases/submit \
  -H "Content-Type: application/json" \
  -d '{
    "patient_text": "62yo male with crushing chest pain radiating to left arm...",
    "include_drug_check": true,
    "include_guidelines": true
  }'
```

---

## Documentation Index

| Document | Description |
|----------|-------------|
| [README.md](README.md) | This file — overview, setup, results |
| [docs/architecture.md](docs/architecture.md) | System architecture, pipeline design, design decisions |
| [docs/test_results.md](docs/test_results.md) | Detailed test results, RAG benchmarks, pipeline timing |
| [DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md) | Chronological build history, problems solved, decisions made |
| [docs/writeup_draft.md](docs/writeup_draft.md) | Project writeup / summary |
| [SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md) | Competition submission strategy |
| [RULES_SUMMARY.md](RULES_SUMMARY.md) | Competition rules checklist |

---

## License

Subject to [HAI-DEF Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms) for model usage.

> **Disclaimer:** This is a research / demonstration system. It is NOT a substitute for professional medical judgment. All clinical decisions must be made by qualified healthcare professionals.
