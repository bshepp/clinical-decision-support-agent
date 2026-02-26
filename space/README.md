---
title: CDS Agent
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
fullWidth: true
---

# CDS Agent — Agentic Clinical Decision Support

An agentic pipeline that orchestrates **MedGemma** (HAI-DEF) across six specialized clinical reasoning steps, augmented with drug safety APIs and guideline RAG, to produce comprehensive decision support reports in real time.

## Architecture

```
Frontend (Next.js 14) ←WebSocket→ Backend (FastAPI)
                                      │
                            Orchestrator (6-step pipeline)
                            ├── Step 1: Parse Patient Data (MedGemma)
                            ├── Step 2: Clinical Reasoning (MedGemma)
                            ├── Step 3: Drug Interaction Check (OpenFDA + RxNorm)
                            ├── Step 4: Guideline Retrieval (ChromaDB RAG)
                            ├── Step 5: Conflict Detection (MedGemma)
                            └── Step 6: Synthesis (MedGemma)
```

## Links

- **Code:** [github.com/bshepp/clinical-decision-support-agent](https://github.com/bshepp/clinical-decision-support-agent)
- **Model:** [google/medgemma-27b-text-it](https://huggingface.co/google/medgemma-27b-text-it)
- **Competition:** [MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge)
