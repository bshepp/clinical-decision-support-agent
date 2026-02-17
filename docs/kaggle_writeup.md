# CDS Agent — Agentic Clinical Decision Support System

### Project name

**CDS Agent** — An agentic pipeline that orchestrates MedGemma across six specialized clinical reasoning steps, augmented with drug safety APIs and guideline RAG, to produce comprehensive decision support reports in real time.

### Your team

| Name | Specialty | Role |
|------|-----------|------|
| [Your Name] | Software Engineering / AI | Architecture, full-stack development, agent pipeline, RAG system, validation framework |

### Problem statement

**The problem:** Clinical decision-making is among the most cognitively demanding tasks in medicine. For every patient encounter, a clinician must simultaneously parse the clinical narrative, generate a differential diagnosis, recall drug interactions across the medication list, remember relevant clinical guidelines, and synthesize all of this into a care plan — often while fatigued and managing multiple patients.

This cognitive burden has real consequences. Diagnostic errors affect approximately 12 million Americans annually. Medication errors harm over 1.5 million people per year. Many of these errors are not from lack of knowledge, but from the difficulty of integrating information from multiple sources under time pressure.

**Who benefits:** Emergency physicians, hospitalists, and primary care clinicians — anyone making complex diagnostic and treatment decisions at the point of care. Patients benefit from more thorough, evidence-based care with fewer diagnostic and medication errors.

**Impact potential:** The U.S. alone sees ~140 million ED visits per year. Even a modest improvement in diagnostic completeness or medication safety across a fraction of these encounters represents significant harm reduction. Our system surfaces specific, actionable conflicts between clinical guidelines and patient data — the kind of gap that leads to missed diagnoses, omitted treatments, and monitoring failures. By automating the information-gathering and synthesis steps of clinical reasoning, CDS Agent gives clinicians back cognitive bandwidth for the parts of medicine that require human judgment.

### Overall solution

**HAI-DEF model:** MedGemma (`google/medgemma-27b-text-it`) — Google's medical-domain model from the Health AI Developer Foundations collection, deployed on a HuggingFace Dedicated Endpoint (1× A100 80 GB, TGI, bfloat16).

**Why MedGemma is essential, not bolted on:** MedGemma is the reasoning engine in four of six pipeline steps. It is not a wrapper around a general-purpose model — it leverages MedGemma's medical training to:

1. **Parse** free-text clinical narratives into structured patient profiles (demographics, vitals, labs, medications, allergies, history)
2. **Reason** about the case via chain-of-thought to produce a ranked differential diagnosis with explicit evidence for/against each candidate
3. **Detect conflicts** between guideline recommendations and the patient's actual data — identifying omissions, contradictions, dosage concerns, and monitoring gaps
4. **Synthesize** all pipeline outputs into a comprehensive CDS report with recommendations, warnings, and citations

Steps 3 and 4 augment MedGemma with external tools: **OpenFDA + RxNorm APIs** for drug interaction data, and **ChromaDB RAG** over 62 curated clinical guidelines spanning 14 specialties (sourced from ACC/AHA, ADA, GOLD, GINA, IDSA, ACOG, AAN, and others).

The agentic architecture is critical: no single LLM call can parse patient data, check drug interactions against federal databases, retrieve specialty-specific guidelines, AND cross-reference those guidelines against the patient's profile. The orchestrated pipeline produces results that no individual component could achieve alone.

### Technical details

**Architecture:**

```
Frontend (Next.js 14)  ←WebSocket→  Backend (FastAPI)
                                        │
                              Orchestrator (6-step pipeline)
                              ├── Step 1: Parse Patient Data (MedGemma)
                              ├── Step 2: Clinical Reasoning (MedGemma)
                              ├── Step 3: Drug Interaction Check (OpenFDA + RxNorm)
                              ├── Step 4: Guideline Retrieval (ChromaDB RAG, 62 guidelines)
                              ├── Step 5: Conflict Detection (MedGemma)
                              └── Step 6: Synthesis (MedGemma)
```

All inter-step data is strongly typed (Pydantic v2). Each step streams its status to the frontend via WebSocket — the clinician watches the pipeline execute in real time, building trust through transparency.

**Key design decisions:**
- **Custom orchestrator** over LangChain — simpler, more transparent, no framework overhead
- **Conflict detection over confidence scores** — we deliberately rejected numeric "confidence" scores (uncalibrated LLM outputs create dangerous anchoring bias). Instead, we compare guidelines against patient data to surface specific, actionable conflicts with cited sources and suggested resolutions.
- **RAG with curated guidelines** — 62 guidelines across 14 specialties, indexed with sentence-transformer embeddings (all-MiniLM-L6-v2). 100% top-1 retrieval accuracy across 30 test queries.

**Validation results:**

| Test | Result |
|------|--------|
| RAG retrieval accuracy | 30/30 (100%) — correct guideline ranked #1 for every query |
| E2E pipeline (ACS case) | All 6 steps passed, 75 s total |
| Clinical test suite | 22 scenarios across 14 specialties |
| MedQA (50 USMLE cases) | 94% pipeline success, 36% top-1 diagnostic accuracy, 38% mentioned |
| MedQA diagnostic-only (36 cases) | 39% mentioned correct diagnosis in report |

The 36% top-1 on MedQA reflects that many questions are non-diagnostic (treatment, mechanism, statistics) — the pipeline generates differential diagnoses, not multiple-choice answers. On diagnostic questions specifically, 39% mentioned the correct diagnosis.

**Deployment:**
- **Model hosting:** HuggingFace Dedicated Endpoint (`medgemma-27b-cds`), 1× A100 80 GB, scale-to-zero billing
- **HIPAA path:** MedGemma is open-weight and can be self-hosted on-premises, eliminating external data transmission
- **Scalability:** FastAPI async + uvicorn workers; production path includes task queue and horizontal scaling
- **EHR integration:** Current input is manual text paste; production system would use FHIR APIs for automatic patient data extraction

**Stack:** Python 3.10, FastAPI, ChromaDB, sentence-transformers, Next.js 14, React 18, TypeScript, Tailwind CSS

---

**Links:**
- **Video:** [TODO — insert video link]
- **Code:** [github.com/bshepp/clinical-decision-support-agent](https://github.com/bshepp/clinical-decision-support-agent)
- **Live Demo:** [TODO — insert demo link if deployed]
- **HuggingFace Model:** [google/medgemma-27b-text-it](https://huggingface.co/google/medgemma-27b-text-it)
