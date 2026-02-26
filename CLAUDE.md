# CLAUDE.md — Project Intelligence for AI Assistants

> **Read this file first.** It contains essential project context, conventions, and navigation for any AI instance working on this codebase.

---

## Project Overview

**CDS Agent** is an agentic clinical decision support system that orchestrates MedGemma 27B across a 6-step pipeline to produce clinical decision support reports from free-text patient cases. Originally built for the MedGemma Impact Challenge (Kaggle / Google Research).

**App demo:** Available upon request

---

## Key Files

| File | Purpose |
|------|---------|
| [DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md) | Chronological build history and decisions |
| [docs/architecture.md](docs/architecture.md) | System architecture and design decisions |
| [docs/test_results.md](docs/test_results.md) | Detailed test results and benchmarks |
| [docs/deploy_medgemma_hf.md](docs/deploy_medgemma_hf.md) | MedGemma HF Endpoint deployment guide |

---

## Codebase Layout

```
medgemma_impact_challenge/
├── CLAUDE.md                  <- You are here
├── DEVELOPMENT_LOG.md         <- Build history
├── src/backend/
│   ├── app/                   <- Production pipeline
│   │   ├── agent/orchestrator.py   <- 6-step pipeline orchestrator
│   │   ├── services/medgemma.py    <- LLM service (OpenAI-compatible)
│   │   ├── tools/                  <- 6 pipeline tools
│   │   │   ├── patient_parser.py        Step 1: Free-text -> structured data
│   │   │   ├── clinical_reasoning.py    Step 2: Differential diagnosis
│   │   │   ├── drug_interactions.py     Step 3: OpenFDA + RxNorm APIs
│   │   │   ├── guideline_retrieval.py   Step 4: RAG over ChromaDB
│   │   │   ├── conflict_detection.py    Step 5: Guideline vs patient gaps
│   │   │   └── synthesis.py             Step 6: CDS report generation
│   │   ├── models/schemas.py       <- Pydantic data models
│   │   ├── data/clinical_guidelines.json  <- 62 guidelines, 14 specialties
│   │   └── api/                    <- REST + WebSocket endpoints
│   ├── tracks/                <- Experimental pipeline variants
│   │   ├── shared/            <- Cross-track utilities
│   │   ├── rag_variants/      <- Chunking & embedding experiments
│   │   ├── iterative/         <- Serial iterative refinement
│   │   └── arbitrated/        <- Parallel specialists + arbiter
│   └── validation/            <- External dataset validation framework
│       ├── harness_medqa.py   <- MedQA (USMLE) diagnostic accuracy
│       ├── harness_mtsamples.py <- MTSamples parse quality
│       └── harness_pmc.py     <- PMC Case Reports diagnostic accuracy
└── src/frontend/              <- Next.js 14 + React 18 + TypeScript
    └── src/
        ├── components/        <- PatientInput, AgentPipeline, CDSReport
        └── hooks/             <- WebSocket state management
```

---

## Conventions

- **Python style:** Pydantic v2 for all data models, async throughout, type hints everywhere
- **LLM calls:** Always go through `app/services/medgemma.py` — never instantiate the OpenAI SDK directly
- **Structured output:** Use `medgemma.generate_structured(prompt, response_model)` with Pydantic models
- **Temperature conventions:** 0.1 for safety-critical/extraction, 0.2-0.3 for reasoning/synthesis
- **Error handling:** Graceful degradation — return partial results rather than crashing
- **No framework dependencies:** Custom orchestrator, no LangChain/LlamaIndex
