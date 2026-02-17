# CLAUDE.md — Project Intelligence for AI Assistants

> **Read this file first.** It contains essential project context, conventions, and navigation for any AI instance working on this codebase.

---

## Project Overview

**CDS Agent** is an agentic clinical decision support system built for the [MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge) (Kaggle / Google Research). It orchestrates MedGemma across a multi-step pipeline to produce clinical decision support reports.

**Deadline:** February 24, 2026, 11:59 PM UTC

---

## Track System — READ THIS

This project uses an **experimental track system** to evaluate multiple diagnostic accuracy strategies in strict isolation. Each track is an independent pipeline variant with its own files, configuration, and results.

**The track registry is in [TRACKS.md](TRACKS.md).** That file is the single source of truth for:
- Which tracks exist and what they do
- Which files belong to which track
- File tagging conventions
- Isolation rules

### Track Isolation Rules (Summary)

1. **Every file owned by a track MUST have a track tag on line 1** — a comment identifying its track ID (e.g., `# [Track B: RAG Variants]`). The exact format depends on the file type.
2. **Never modify a file owned by one track to benefit another.** Shared code lives in `src/backend/tracks/shared/`.
3. **The baseline pipeline (`src/backend/app/`) is Track A.** Experimental tracks extend or wrap Track A code — they do NOT modify it.
4. **Results from each track are stored separately** under `src/backend/tracks/<track_dir>/results/`.
5. **Cross-track comparison** is performed only via shared utilities in `src/backend/tracks/shared/`.

See **[TRACKS.md](TRACKS.md)** for the complete specification.

---

## Critical Files

| File | Purpose |
|------|---------|
| **[TRACKS.md](TRACKS.md)** | Track registry, file ownership, isolation rules — **start here for experimental work** |
| **[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)** | 4-phase execution plan for accuracy optimization — **the step-by-step playbook** |
| [TODO.md](TODO.md) | Session-level action items and project status |
| [DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md) | Chronological build history and decisions |
| [SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md) | Competition rules, timeline, and submission checklist |
| [docs/kaggle_writeup.md](docs/kaggle_writeup.md) | Final writeup content for Kaggle submission |
| [docs/video_script.md](docs/video_script.md) | 3-minute demo video narration script |
| [docs/architecture.md](docs/architecture.md) | System architecture and design decisions |

---

## Codebase Layout

```
medgemma_impact_challenge/
├── CLAUDE.md                  ← You are here
├── TRACKS.md                  ← Track registry and isolation rules
├── TODO.md                    ← Next-session action items
├── DEVELOPMENT_LOG.md         ← Build history
├── src/backend/
│   ├── app/                   ← Track A (Baseline) — production pipeline
│   │   ├── agent/orchestrator.py
│   │   ├── services/medgemma.py
│   │   ├── tools/             ← 6 pipeline tools
│   │   ├── models/schemas.py
│   │   └── data/clinical_guidelines.json
│   ├── tracks/                ← Experimental tracks
│   │   ├── shared/            ← Cross-track utilities (cost tracking, comparison)
│   │   ├── rag_variants/      ← Track B: Chunking & embedding experiments
│   │   ├── iterative/         ← Track C: Serial iterative refinement
│   │   ├── arbitrated/        ← Track D: Parallel specialists + arbiter
│   │   ├── combined/          ← Track E: Composition of per-axis winners (Phase 3)
│   │   ├── prompt_arch/       ← Track F: Prompt architecture variants (Phase 2)
│   │   ├── voting/            ← Track G: Multi-sample voting (Phase 2)
│   │   └── verification/      ← Track H: Evidence verification (Phase 2)
│   └── validation/            ← Validation framework (shared across all tracks)
└── src/frontend/              ← Next.js frontend (not track-specific)
```

---

## Conventions

- **Python style:** Pydantic v2 for all data models, async throughout, type hints everywhere
- **LLM calls:** Always go through `app/services/medgemma.py` — never instantiate the OpenAI SDK directly
- **Structured output:** Use `medgemma.generate_structured(prompt, response_model)` with Pydantic models
- **Temperature conventions:** 0.1 for safety-critical/extraction, 0.2–0.3 for reasoning/synthesis
- **Error handling:** Graceful degradation — return partial results rather than crashing
- **No framework dependencies:** Custom orchestrator, no LangChain/LlamaIndex
- **Windows compatibility:** ASCII characters only in console output (no box-drawing or Unicode symbols)
- **Track tagging:** Line 1 of every track-owned file must carry the track tag comment
