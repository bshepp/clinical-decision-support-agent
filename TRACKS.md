# TRACKS.md — Experimental Track Registry

> **Single source of truth** for all experimental tracks, their file ownership, tagging conventions, and isolation rules.  
> Referenced by [CLAUDE.md](CLAUDE.md). Read that file first for general project context.

---

## Why Tracks?

The baseline pipeline (Track A) achieves 36% top-1 diagnostic accuracy on MedQA. To improve this, we are evaluating **multiple independent strategies** in parallel. Each strategy is an isolated "track" with its own code, configuration, and results — so we can compare them fairly without cross-contamination.

---

## Track Registry

| ID | Name | Directory | Strategy |
|----|------|-----------|----------|
| **A** | Baseline | `src/backend/app/` | The production 6-step pipeline. No modifications for experiments. |
| **B** | RAG Variants | `src/backend/tracks/rag_variants/` | Test different chunking sizes, segment strategies, and embedding models to optimize guideline retrieval quality and downstream diagnostic accuracy. |
| **C** | Iterative Refinement | `src/backend/tracks/iterative/` | Run the diagnosis step in a serial loop — each iteration critiques and refines the previous output. Continue until the marginal improvement drops below a cost/benefit threshold. Produces a convergence chart. |
| **D** | Arbitrated Parallel | `src/backend/tracks/arbitrated/` | Run multiple specialist reasoning agents in parallel. An arbiter agent evaluates all outputs, tailors resubmission prompts for each specialist based on their strengths/weaknesses, and repeats until the cost/benefit ratio plateaus. Produces a cost/benefit chart. |
| **E** | Combined | `src/backend/tracks/combined/` | Compose per-axis winners from B/C/D/F/G/H. Tests 3 composition patterns (breadth-then-depth, depth-within-breadth, bookend). **Phase 3 — build after Phase 1+2 data.** |
| **F** | Prompt Architecture | `src/backend/tracks/prompt_arch/` | Test how reasoning prompt structure affects accuracy: structured template, few-shot, reverse reasoning, Bayesian framing. **Phase 2.** |
| **G** | Multi-Sample Voting | `src/backend/tracks/voting/` | Self-consistency via repeated sampling + majority/weighted vote. 1/3/5 samples at varying temperatures. **Phase 2.** |
| **H** | Evidence Verification | `src/backend/tracks/verification/` | Post-hoc grounding check: verify each diagnosis against patient evidence, re-rank by grounding score. **Phase 2.** |
| **—** | Shared | `src/backend/tracks/shared/` | Cross-track utilities: cost tracking, comparison harness, chart generation. Not a track itself. |

---

## File Tagging Convention

**Every file owned by a track MUST carry a track tag on line 1.** This makes ownership unambiguous when reading any file in isolation.

### Format by file type

| File Type | Tag Format | Example |
|-----------|-----------|---------|
| Python (`.py`) | `# [Track X: Name]` | `# [Track B: RAG Variants]` |
| JSON (`.json`) | First key in object | `{"_track": "Track B: RAG Variants", ...}` |
| Markdown (`.md`) | HTML comment | `<!-- [Track B: RAG Variants] -->` |
| Config (`.env`, `.yaml`) | Comment | `# [Track B: RAG Variants]` |

### Track A exception

Track A files (`src/backend/app/`) were written before the track system existed. They are tagged with `# [Track A: Baseline]` on line 1, but their code is NOT modified for experimental purposes. Experiments extend or wrap Track A code from within their own track directory.

---

## Isolation Rules

These rules prevent cross-contamination between experimental tracks:

### 1. File Ownership

- Each file belongs to exactly **one track** (identified by its line-1 tag and directory).
- Files in `src/backend/app/` belong to **Track A**.
- Files in `src/backend/tracks/<dir>/` belong to the corresponding track.
- Files in `src/backend/tracks/shared/` are shared utilities, not owned by any single track.

### 2. No Cross-Modification

- **Never modify a Track A file to serve an experiment.** Instead, import and extend from your track's directory.
- **Never modify a Track B file from Track C code**, and so forth.
- If two tracks need the same utility, put it in `shared/`.

### 3. Import Direction

```
Track B/C/D code  →  may import from  →  Track A (app/) and shared/
Track A code      →  NEVER imports    →  Track B/C/D
shared/ code      →  may import from  →  Track A (app/) only
```

### 4. Results Isolation

- Each track stores results in `src/backend/tracks/<dir>/results/`.
- Result filenames include the track ID prefix (e.g., `trackB_medqa_20260215.json`).
- Cross-track comparison is done **only** via `src/backend/tracks/shared/compare.py`.

### 5. Configuration Isolation

- Track-specific parameters live in each track's own config or constants — not in `app/config.py`.
- The shared `app/config.py` provides only baseline/global settings (API keys, endpoints, etc.).

---

## Track Details

### Track A: Baseline

**Purpose:** The production-ready pipeline. The control group for all experiments.

**Pipeline:** Parse → Reason → Drug Check → Guideline Retrieval → Conflict Detection → Synthesis

**Key parameters:**
- Embedding: `all-MiniLM-L6-v2` (384 dims)
- RAG top-k: 5
- No guideline chunking (each guideline = 1 document)
- Clinical reasoning temperature: 0.3
- Synthesis temperature: 0.2
- Single-pass reasoning (no iteration)

**Baseline accuracy (50-case MedQA):** 36% top-1, 38% mentioned

---

### Track B: RAG Variants

**Purpose:** Determine whether retrieval quality improvements translate to better diagnostic accuracy.

**Experiments:**
1. **Chunking strategies** — Split each guideline into smaller segments (100-word chunks, 200-word chunks, sentence-level) with configurable overlap
2. **Embedding models** — Compare `all-MiniLM-L6-v2` (384d) vs `all-mpnet-base-v2` (768d) vs `bge-base-en-v1.5` (768d) vs `medcpt` (medical-specific)
3. **Top-k variation** — Test k=3, k=5, k=8, k=10 to find optimal retrieval breadth
4. **Re-ranking** — Add a cross-encoder re-ranking step after initial retrieval

**Measured outcomes:**
- RAG retrieval accuracy (30-query test suite)
- MedQA diagnostic accuracy (same 50-case seed=42)
- Retrieval latency per query

**Key files:**
- `src/backend/tracks/rag_variants/config.py` — Variant definitions
- `src/backend/tracks/rag_variants/chunker.py` — Guideline chunking strategies
- `src/backend/tracks/rag_variants/retriever.py` — Modified retrieval with configurable embedding/chunking
- `src/backend/tracks/rag_variants/run_variants.py` — Runner that tests all configurations
- `src/backend/tracks/rag_variants/results/` — Per-variant results

---

### Track C: Iterative Refinement

**Purpose:** Determine whether repeated self-critique improves diagnostic accuracy, and find the point of diminishing returns.

**Method:**
1. Run baseline clinical reasoning (iteration 0)
2. Feed the output back along with the patient data and a critique prompt
3. The model reviews its own differential, identifies weaknesses, and produces a refined version
4. Repeat until: (a) max iterations reached, or (b) the differential stops changing meaningfully
5. Track accuracy and LLM cost at each iteration to produce a convergence/cost-benefit chart

**Measured outcomes:**
- Accuracy at each iteration (top-1, top-3, mentioned)
- LLM token cost at each iteration
- Convergence curve: accuracy vs. cumulative cost
- Iteration at which improvement drops below threshold

**Key files:**
- `src/backend/tracks/iterative/config.py` — Max iterations, convergence threshold
- `src/backend/tracks/iterative/refiner.py` — Iterative reasoning loop with self-critique
- `src/backend/tracks/iterative/run_iterative.py` — Runner with per-iteration scoring
- `src/backend/tracks/iterative/results/` — Per-iteration results and charts

---

### Track D: Arbitrated Parallel

**Purpose:** Determine whether multiple specialist agents, coordinated by an arbiter, outperform a single-pass generalist — and at what cost.

**Method:**
1. Run N specialist reasoning agents **in parallel**, each with a domain-specific system prompt (e.g., cardiologist, neurologist, infectious disease specialist)
2. An **arbiter agent** receives all N specialist outputs plus the patient data
3. The arbiter evaluates each specialist's differential, identifies agreements and disagreements
4. The arbiter generates **tailored resubmission prompts** for each specialist — telling the cardiologist "the neurologist raised X, reconsider Y" and vice versa
5. Specialists run again with the arbiter's feedback
6. Repeat until: (a) consensus reached, (b) max rounds, or (c) cost/benefit drops below threshold
7. The arbiter produces the final merged differential
8. Track accuracy and cost at each round to produce a cost/benefit chart

**Measured outcomes:**
- Accuracy at each arbitration round (top-1, top-3, mentioned)
- Per-specialist accuracy contribution
- LLM token cost per round (N specialists + 1 arbiter)
- Cost/benefit convergence chart
- Consensus rate across rounds

**Key files:**
- `src/backend/tracks/arbitrated/config.py` — Specialist definitions, max rounds, threshold
- `src/backend/tracks/arbitrated/specialists.py` — Domain-specific reasoning agents
- `src/backend/tracks/arbitrated/arbiter.py` — Arbiter agent that evaluates and coordinates
- `src/backend/tracks/arbitrated/run_arbitrated.py` — Runner with per-round scoring
- `src/backend/tracks/arbitrated/results/` — Per-round results and charts

---

## Adding a New Track

1. Choose an unused letter ID (E, F, ...).
2. Create `src/backend/tracks/<dir_name>/` with `__init__.py`.
3. Add the track to the **Track Registry** table above.
4. Tag every new file on line 1 with `# [Track X: Name]`.
5. Store results in `src/backend/tracks/<dir_name>/results/`.
6. Add a comparison entry in `src/backend/tracks/shared/compare.py`.
7. Never import from another track's directory — only from `app/` and `shared/`.
