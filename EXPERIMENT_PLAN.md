# EXPERIMENT_PLAN.md — 4-Phase Accuracy Optimization Plan

> **Purpose:** Step-by-step execution plan for an AI agent or human to follow.
> Each step is atomic, has clear inputs/outputs, and explicit success criteria.
>
> **Context:** Baseline accuracy is 36% top-1 on 50-case MedQA (seed=42). Our
> goal is to find the best composite strategy before the Feb 24, 2026 deadline.
>
> **Prerequisite reading:** `CLAUDE.md` → `TRACKS.md` → this file.

---

## Infrastructure Prerequisites

Before ANY phase, ensure:

1. **HF Endpoint is running.**
   - Go to https://ui.endpoints.huggingface.co → `medgemma-27b-cds` → Resume
   - Wait until status shows "Running" (5–15 min cold start)
   - Cost: ~$2.50/hr — **pause when done**

2. **Virtual environment is active.**
   ```powershell
   cd f:\kaggle\medgemma_impact_challenge\src\backend
   .\venv\Scripts\Activate.ps1
   ```

3. **Dependencies installed.**
   ```powershell
   pip install -r requirements.txt
   pip install sentence-transformers   # Needed for Track B embedding variants
   ```

4. **Environment variables set.**
   - `.env` file in `src/backend/` must have `HF_TOKEN`, `MEDGEMMA_API_KEY`, `MEDGEMMA_BASE_URL`
   - Verify: `python -c "from app.config import Settings; s = Settings(); print(s.medgemma_base_url)"`

5. **Quick health check.** Run 1 case through baseline to confirm the endpoint responds:
   ```powershell
   python -m validation.run_validation --medqa --max-cases 1
   ```
   **Success:** Pipeline returns a `CDSReport` without timeout errors.

---

## Phase 1 — Independent Axis Sweeps

**Goal:** Find the best single-axis configuration for B, C, and D independently.
**Estimated cost:** ~$15–25 of endpoint time (6–10 hours)
**Estimated cases:** 50 per config × (10 + 4 + 4) = 900 total pipeline runs

### Phase 1A — Track B: RAG Variants

**What we're testing:** Which retrieval configuration gets the best documents in front of the model?

#### Step 1A.1: Smoke Test (3 cases × 10 variants = 30 runs)

```powershell
cd f:\kaggle\medgemma_impact_challenge\src\backend
python -m tracks.rag_variants.run_variants --max-cases 3
```

**Check for:**
- [ ] All 10 variants complete without errors
- [ ] Each variant produces a result JSON in `tracks/rag_variants/results/`
- [ ] MedCPT and MPNet embedding models download successfully
- [ ] Reranking variant (B9) loads the cross-encoder model
- [ ] Output shows a comparison table with per-variant scores

**If any variant fails:** Fix the error, then re-run with `--variant <id>` to test just that one:
```powershell
python -m tracks.rag_variants.run_variants --variant B6_medcpt --max-cases 3
```

**Common failure modes:**
- `sentence-transformers` not installed → `pip install sentence-transformers`
- MedCPT download fails → check `HF_TOKEN` is set
- ChromaDB lock → delete `tracks/rag_variants/data/chroma/` and retry

#### Step 1A.2: Full Sweep (50 cases × 10 variants = 500 runs)

```powershell
python -m tracks.rag_variants.run_variants
```

**Expected runtime:** 3–5 hours (50 cases × 10 variants, ~2 min/case with API latency)

**Output:** Results in `tracks/rag_variants/results/` — one JSON per variant.

#### Step 1A.3: Identify B*

Read the comparison table printed at the end, or run:
```powershell
python -m tracks.shared.compare --tracks B --dataset medqa
```

**Record the winner:**
```
B* = ____________ (variant_id)
B* top-1 accuracy = _____%
B* improvement over B0_baseline = +_____%
```

**Decision rules:**
- If the best variant beats B0 by <2%, retrieval isn't the bottleneck. Note this, but still carry B* forward.
- If multiple variants tie within 1%, prefer the one with lower latency/complexity.
- If reranking (B9) wins, note the added latency cost.

---

### Phase 1B — Track C: Iterative Refinement

**What we're testing:** Does repeated self-critique improve diagnostic accuracy? At what point do returns diminish?

#### Step 1B.1: Smoke Test (3 cases × 4 configs = 12 runs)

```powershell
python -m tracks.iterative.run_iterative --max-cases 3
```

**Check for:**
- [ ] All 4 configs complete without errors
- [ ] Per-iteration accuracy and cost data is printed
- [ ] Convergence detection works (C0_2rounds should always run all 2 iterations; C2_5rounds might converge early)
- [ ] Cost ledger populates correctly

**If a config hangs:** Likely an LLM timeout. Check that the endpoint is warm. The iterative track makes 2-10× more LLM calls per case than baseline.

#### Step 1B.2: Full Sweep (50 cases × 4 configs)

```powershell
python -m tracks.iterative.run_iterative
```

**Expected runtime:** 2–4 hours (C0 fastest, C3 slowest)

**Output:** Results in `tracks/iterative/results/`

#### Step 1B.3: Identify C*

```powershell
python -m tracks.shared.compare --tracks C --dataset medqa
```

**Record the winner:**
```
C* = ____________ (config_id)
C* top-1 accuracy = _____%
C* avg iterations used = _____
C* cost per case = $_____
C* improvement over baseline = +_____%
```

**Key data to extract:** The per-iteration accuracy curve. Plot or record:
```
Iteration 0 (baseline): ___% top-1
Iteration 1 (first critique): ___% top-1
Iteration 2: ___% top-1
Iteration 3: ___% top-1 (if applicable)
...
```

**Decision rules:**
- The winning config is the one with the best accuracy/cost ratio, not necessarily the one with the highest absolute accuracy.
- If C2_5rounds converges at iteration 2 in most cases, the extra rounds aren't helping — C1_3rounds is probably enough.
- If C3_aggressive loses accuracy (the critic is too harsh), note this as a failure mode.

---

### Phase 1C — Track D: Arbitrated Parallel

**What we're testing:** Do multiple specialist perspectives, coordinated by an arbiter, find diagnoses a generalist misses?

#### Step 1C.1: Smoke Test (3 cases × 4 configs = 12 runs)

```powershell
python -m tracks.arbitrated.run_arbitrated --max-cases 3
```

**Check for:**
- [ ] All 4 configs complete without errors
- [ ] Specialist outputs show domain-specific reasoning (cardiologist emphasizes cardiac, etc.)
- [ ] Arbiter merge output is a coherent consensus differential, not just concatenation
- [ ] For multi-round configs (D2, D3): tailored resubmission prompts are generated
- [ ] For multi-round configs: second-round specialist outputs differ from first round
- [ ] Cost tracking shows escalating cost with more specialists/rounds

**If the arbiter produces garbage:** The merge prompt may need tuning. Check `tracks/arbitrated/arbiter.py` ARBITER_MERGE_PROMPT.

#### Step 1C.2: Full Sweep (50 cases × 4 configs)

```powershell
python -m tracks.arbitrated.run_arbitrated
```

**Expected runtime:** 3–6 hours (D0 fastest, D3 slowest — D3 runs 5 specialists × 2 rounds = 12 LLM calls/case)

**Output:** Results in `tracks/arbitrated/results/`

#### Step 1C.3: Identify D*

```powershell
python -m tracks.shared.compare --tracks D --dataset medqa
```

**Record the winner:**
```
D* = ____________ (config_id)
D* top-1 accuracy = _____%
D* cost per case = $_____
D* improvement over baseline = +_____%
```

**Additional data to record:**
```
Per-specialist contribution analysis:
  Cardiologist: Contributed unique correct dx in ___% of cases
  Neurologist: ____%
  ID Specialist: ____%
  General Internist: ____%
  Emergency Med: ____%
Arbitration consensus rate: ____% of cases where >3 specialists agreed on top-1
Round 2 lift (if applicable): +____% over round 1
```

**Decision rules:**
- If D0 (3-spec, 1-round) matches D3 (5-spec, 2-rounds), the extra cost isn't justified.
- If specialists all agree in round 1, round 2 is wasted computation — future configs can drop it.
- If one specialist consistently disagrees with the correct answer, consider removing it from the ensemble.

---

### Phase 1D — Cross-Track Comparison

After all three tracks complete, run the unified comparison:

```powershell
python -m tracks.shared.compare --dataset medqa
```

**Expected output:**
```
Cross-Track Comparison: MEDQA
-------------------------------------------------------------
Track                   Top-1   Top-3  Mentioned  Pipeline       Cost
-------------------------------------------------------------
A: Baseline              36.0%  --       38.0%     94.0%        $X.XX
B: RAG Variants          ___%   --       ___%      ___%         $X.XX
C: Iterative             ___%   --       ___%      ___%         $X.XX
D: Arbitrated            ___%   --       ___%      ___%         $X.XX
-------------------------------------------------------------
```

**Record Phase 1 summary:**
```
B* = __________, accuracy = ____%, delta = +____%
C* = __________, accuracy = ____%, delta = +____%
D* = __________, accuracy = ____%, delta = +____%
Best single axis: Track ___
```

**Go/No-Go for Phase 2:**
- If ALL tracks are within 2% of baseline → the model itself may be the bottleneck,
  not the pipeline. Consider investigating prompt architecture (Phase 2) more aggressively.
- If ANY single track shows ≥5% lift → strong signal, proceed to Phase 2 and Phase 3.
- If results are noisy (high variance) → increase to 100 cases or use a different seed
  to get more statistical power.

---

## Phase 2 — New Axes (F, G, H)

**Goal:** Test 3 lightweight axes that are cheap to implement and orthogonal to B/C/D.
**Build these ONLY after Phase 1 data is in.** Phase 1 results inform which axes matter most.

### Phase 2A — Track F: Prompt Architecture

**Axis:** *How* the model is asked to reason, independent of depth (C) or breadth (D).

**Why:** This is the cheapest axis to test — same token count, different structure. If prompt architecture matters more than retrieval or iteration, we want to know early.

#### Step 2A.1: Build Track F

Create `src/backend/tracks/prompt_arch/` with the track system conventions (see TRACKS.md "Adding a New Track").

**Files to create:**
```
tracks/prompt_arch/
  __init__.py           # Track tag, package init
  config.py             # PromptVariant dataclass + 5 variants
  reasoner.py           # Modified clinical_reasoning that accepts prompt templates
  run_prompt_arch.py    # Runner following same pattern as other tracks
  results/              # Output directory
```

**Variant definitions:**
| ID | Name | Strategy | Prompt Change |
|----|------|----------|---------------|
| F0 | Baseline | Current free-form | No change (control) |
| F1 | Structured Template | Force structured output | System prompt: "For each symptom, list 3 possible causes. Identify diagnoses appearing in ≥2 symptom lists. Rank by frequency of appearance." |
| F2 | Few-Shot | 2 worked examples | Add 2 solved MedQA cases (NOT from test set) to the system prompt as worked examples with reasoning chains |
| F3 | Reverse Reasoning | Falsification | After initial differential: "For each of your top 5 diagnoses, list the findings you would EXPECT. Mark which are present, absent, or unknown in this patient. Re-rank based on match percentage." |
| F4 | Bayesian | Prior updating | "Assign a prior probability to each diagnosis based on prevalence. For each finding, update posterior probability. Show the Bayesian reasoning chain. Final differential ordered by posterior." |

**Implementation notes:**
- `reasoner.py` should accept a `prompt_template: str` parameter and inject it into the system prompt or user prompt of the clinical reasoning call.
- F0 uses the exact same system prompt as `app/tools/clinical_reasoning.py` — this is the control.
- Few-shot examples (F2) need to come from MedQA TRAIN set, not the 50-case test set. Pick 2 from `validation/data/medqa_test.jsonl` that are NOT in the seed=42 sample, or create synthetic examples from textbook cases.
- F3 and F4 require TWO LLM calls: first the initial differential, then the structured verification/update. This makes them comparable to C in cost but different in mechanism (structured verification vs. open-ended critique).

#### Step 2A.2: Run Track F

```powershell
# Smoke test
python -m tracks.prompt_arch.run_prompt_arch --max-cases 3

# Full sweep
python -m tracks.prompt_arch.run_prompt_arch
```

#### Step 2A.3: Identify F*

```
F* = ____________
F* top-1 accuracy = _____%
F* improvement over F0 = +_____%
```

---

### Phase 2B — Track G: Multi-Sample Voting (Self-Consistency)

**Axis:** Statistical diversity via repeated sampling at higher temperature.

**Why:** Self-consistency is one of the most reliable accuracy boosters in the CoT literature. It's embarrassingly parallel and requires no new prompts — just `asyncio.gather()` over N samples.

#### Step 2B.1: Build Track G

Create `src/backend/tracks/voting/`.

**Files:**
```
tracks/voting/
  __init__.py
  config.py             # VotingConfig: n_samples, temperature, aggregation_method
  voter.py              # Generate N reasoning outputs, extract top-k diagnoses, vote
  run_voting.py
  results/
```

**Variant definitions:**
| ID | Samples | Temp | Aggregation | Description |
|----|---------|------|-------------|-------------|
| G0 | 1 | 0.3 | N/A | Control (identical to baseline) |
| G1 | 3 | 0.5 | Majority vote | 3 samples, majority wins |
| G2 | 5 | 0.5 | Majority vote | 5 samples, majority wins |
| G3 | 5 | 0.7 | Weighted vote | 5 samples at higher diversity, weighted by internal consistency |
| G4 | 3 | 0.5 | Best-of-N | 3 samples, pick the one whose differential best matches retrieved guidelines |

**Implementation notes:**
- `voter.py` calls `medgemma.generate()` N times in parallel with `asyncio.gather()`.
- Temperature must be high enough to get diversity (≥0.5), otherwise all N samples will be nearly identical.
- **Majority vote aggregation:** Extract top-1 diagnosis from each sample. The diagnosis appearing most frequently wins. If tied, use the one from the sample with the longest reasoning (proxy for confidence).
- **Weighted vote (G3):** For each sample, check how many of its diagnoses are mentioned in the retrieved guidelines. Weight = number of guideline-grounded diagnoses. This penalizes hallucinated differentials.
- **Best-of-N (G4):** Score each sample's differential against the retrieved guidelines using fuzzy_match overlap. Pick the highest-scoring sample wholesale.
- Cost scales linearly: G2 costs 5× baseline reasoning per case.

#### Step 2B.2: Run Track G

```powershell
python -m tracks.voting.run_voting --max-cases 3   # smoke
python -m tracks.voting.run_voting                   # full
```

#### Step 2B.3: Identify G*

```
G* = ____________
G* top-1 accuracy = _____%
G* cost multiplier vs baseline = _____×
```

---

### Phase 2C — Track H: Evidence Verification (Post-Hoc Grounding)

**Axis:** A structured fact-check pass that re-ranks the differential based on evidence alignment.

**Why:** The model might rank a diagnosis #1 that isn't actually supported by the evidence. H catches this. It's different from C (which is open-ended self-critique) — H is specifically checking "does the evidence support this ranking?"

#### Step 2C.1: Build Track H

Create `src/backend/tracks/verification/`.

**Files:**
```
tracks/verification/
  __init__.py
  config.py             # VerificationConfig
  verifier.py           # Post-hoc evidence grounding check
  run_verification.py
  results/
```

**Method for each case:**
1. Run baseline pipeline → get differential with top-5 diagnoses
2. For EACH diagnosis in the differential, make ONE LLM call:
   ```
   Patient findings: {summary}
   Retrieved guidelines: {relevant_guidelines}
   Diagnosis under review: {diagnosis_name}

   Task: List the specific findings from this patient that SUPPORT this diagnosis,
   the findings that ARGUE AGAINST it, and the findings that are NEUTRAL.
   Give a grounding score from 0-10 based on evidence alignment.
   ```
3. Re-rank the differential by grounding score (descending)
4. Use the re-ranked differential for scoring

**Variant definitions:**
| ID | Method | LLM Calls | Description |
|----|--------|-----------|-------------|
| H0 | None | 0 extra | Control |
| H1 | Top-5 re-rank | 5 extra | Verify and re-rank all 5 diagnoses |
| H2 | Top-3 re-rank | 3 extra | Verify only top 3 (cheaper) |
| H3 | Eliminate-only | 5 extra | Don't re-rank — just DROP any diagnosis with score ≤3 and promote the rest |

**Implementation notes:**
- Use `medgemma.generate_structured()` with a Pydantic model for the grounding output:
  ```python
  class GroundingResult(BaseModel):
      diagnosis: str
      supporting_findings: List[str]
      opposing_findings: List[str]
      neutral_findings: List[str]
      grounding_score: int  # 0-10
  ```
- Temperature: 0.1 (this is extraction/evaluation, not generation)
- Each verification call is independent → run all 5 in parallel with `asyncio.gather()`

#### Step 2C.2: Run Track H

```powershell
python -m tracks.verification.run_verification --max-cases 3
python -m tracks.verification.run_verification
```

#### Step 2C.3: Identify H*

```
H* = ____________
H* top-1 accuracy = _____%
H* improvement over baseline = +_____%
```

---

### Phase 2D — Phase 2 Cross-Comparison

After F, G, H are done:

```powershell
python -m tracks.shared.compare --dataset medqa
```

Update the shared compare.py to include tracks E/F/G/H before running (add entries to `TRACK_DIRS`).

**Record Phase 2 summary:**
```
F* = __________, accuracy = ____%, delta = +____%
G* = __________, accuracy = ____%, delta = +____%, cost = _____×
H* = __________, accuracy = ____%, delta = +____%
```

**Rank all 6 axes by accuracy lift:**
```
1. Track ___ : +____% (cost: ___×)
2. Track ___ : +____% (cost: ___×)
3. Track ___ : +____% (cost: ___×)
4. Track ___ : +____% (cost: ___×)
5. Track ___ : +____% (cost: ___×)
6. Track ___ : +____% (cost: ___×)
```

---

## Phase 3 — Composition (Track E: Combined)

**Goal:** Wire the per-axis winners together and test whether gains are additive.
**Only start this after Phase 1 and Phase 2 data is in hand.**

### Step 3.1: Build Track E

Create `src/backend/tracks/combined/`.

**Files:**
```
tracks/combined/
  __init__.py
  config.py           # CombinedConfig: which B*/C*/D*/F*/G*/H* to compose
  pipeline.py         # The composite pipeline that wires winners together
  run_combined.py
  results/
```

**CombinedConfig should reference winner IDs from Phase 1 and 2:**
```python
@dataclass
class CombinedConfig:
    config_id: str
    rag_variant_id: Optional[str]         # B* winner (or None = baseline retrieval)
    iterative_config_id: Optional[str]    # C* winner (or None = no iteration)
    arbitrated_config_id: Optional[str]   # D* winner (or None = single generalist)
    prompt_variant_id: Optional[str]      # F* winner (or None = default prompt)
    voting_config_id: Optional[str]       # G* winner (or None = single sample)
    verification_config_id: Optional[str] # H* winner (or None = no verification)
    composition_pattern: str              # "E1", "E2", or "E3"
    description: str = ""
```

### Step 3.2: Implement 3 Composition Patterns

**Pattern E1: Breadth-then-Depth** (recommended starting point)
```
Parse
  → B* retriever (swap guideline retrieval)
  → F* prompt template (swap reasoning prompt)
  → D* specialists in parallel (each uses F* prompt)
  → D* arbiter merge → consensus differential
  → C* iterative refinement on consensus
  → H* evidence verification on refined output
  → G* voting: run the above N times and vote (if G* ≠ G0)
  → Drug Check + Conflict Detection
  → Synthesis
```

**Pattern E2: Depth-within-Breadth**
```
Parse
  → B* retriever
  → D* specialists, each with F* prompt, each running C* internal iteration
  → D* arbiter merge over refined specialist outputs
  → H* evidence verification
  → G* voting over the above
  → Drug Check + Conflict Detection
  → Synthesis
```

**Pattern E3: Bookend (full loop)**
```
Parse
  → B* retriever
  → D* specialists (round 1, F* prompt)
  → D* arbiter merge → rough consensus
  → C* iterative refinement on consensus
  → D* specialists again (round 2, with refined consensus as additional context)
  → D* arbiter re-merge → final differential
  → H* evidence verification
  → G* voting
  → Drug Check + Conflict Detection
  → Synthesis
```

**Implementation guidance:**
- Import existing track modules — do NOT duplicate code
  ```python
  from tracks.rag_variants.retriever import VariantRetriever
  from tracks.rag_variants.config import VARIANTS
  from tracks.iterative.refiner import IterativeRefiner
  from tracks.iterative.config import CONFIGS as ITERATIVE_CONFIGS
  from tracks.arbitrated.specialists import run_specialists_parallel
  from tracks.arbitrated.arbiter import Arbiter
  from tracks.arbitrated.config import CONFIGS as ARBITRATED_CONFIGS
  ```
- The orchestrator's tools are swappable: `orchestrator.guideline_retrieval = variant_retriever`
- Use a single `CostLedger` that spans ALL stages so the total cost is tracked

### Step 3.3: Run Compositions

```powershell
# Start with E1 (simplest)
python -m tracks.combined.run_combined --pattern E1 --max-cases 3   # smoke
python -m tracks.combined.run_combined --pattern E1                  # full 50 cases

# Then E2 and E3 if E1 shows promise
python -m tracks.combined.run_combined --pattern E2 --max-cases 10
python -m tracks.combined.run_combined --pattern E3 --max-cases 10
```

### Step 3.4: Evaluate Composition

**Record:**
```
E1 top-1 accuracy = _____% | cost/case = $_____ | runtime/case = ___s
E2 top-1 accuracy = _____% | cost/case = $_____ | runtime/case = ___s
E3 top-1 accuracy = _____% | cost/case = $_____ | runtime/case = ___s

Best single track: Track ___ at ____%
Best composition: Pattern ___ at ____%
Composition lift vs best single track: +____%
```

**Key questions to answer:**
1. Are the gains from B/C/D/F/G/H additive when composed? (If E1 ≈ best single track, they're not.)
2. Which pattern gives the best accuracy/cost ratio?
3. Is there a simpler 2-axis composition (e.g., B+C only) that gets 80% of the E1 benefit at 30% of the cost?

### Step 3.5: Test Partial Compositions

Based on the Phase 1+2 ranking, test 2-axis combos of the top 3 axes:

```
E_BC:  B* + C* only (better retrieval + iteration)
E_BD:  B* + D* only (better retrieval + specialists)
E_BF:  B* + F* only (better retrieval + prompt architecture)
E_CD:  C* + D* only (iteration + specialists)
E_BH:  B* + H* only (better retrieval + verification)
```

This tells us which pairs compose well and which interfere. Run each at 50 cases.

**Record pair interaction matrix:**
```
         B*      C*      D*      F*      G*      H*
B*        -     ____%   ____%   ____%   ____%   ____%
C*              -       ____%   ____%   ____%   ____%
D*                      -       ____%   ____%   ____%
F*                              -       ____%   ____%
G*                                      -       ____%
H*                                              -
```
(Each cell = top-1 accuracy of that 2-axis composition)

---

## Phase 4 — Cherry-Pick and Finalize

**Goal:** Take the best composition from Phase 3 and apply any remaining optimizations.

### Step 4.1: Lock the Winner

Based on Phase 3 data, select the final pipeline configuration:

```
FINAL CONFIG:
  Retrieval: ____________ (B variant or baseline)
  Prompt: ____________ (F variant or baseline)
  Reasoning: ____________ (D config, or single generalist)
  Iteration: ____________ (C config, or none)
  Verification: ____________ (H config, or none)
  Voting: ____________ (G config, or single sample)
  Composition: ____________ (E pattern)
  Top-1 accuracy: ____%
  Cost per case: $____
  Runtime per case: ____s
```

### Step 4.2: 100-Case Validation

Run the final config against an expanded dataset to confirm the result isn't a fluke:

```powershell
# If possible, run 100 MedQA cases (load more from the JSONL)
python -m tracks.combined.run_combined --pattern <winner> --max-cases 100
```

**If 100-case accuracy is within ±3% of 50-case accuracy:** The result is stable.
**If it drops by >5%:** We overfit to the 50-case sample. Re-evaluate.

### Step 4.3: Run Complementary Benchmarks

Run the winner through MTSamples and PMC harnesses (if available) to show generalization:

```powershell
# These may need adaptation to work with the combined pipeline
python -m validation.run_validation --mtsamples --max-cases 20
python -m validation.run_validation --pmc --max-cases 10
```

### Step 4.4: Update Submission Materials

1. **Update `docs/kaggle_writeup.md`** with final accuracy numbers, the winning configuration,
   and the experimental journey (which axes mattered, which didn't, composition effects).

2. **Update `docs/video_script.md`** if the demo pipeline changed significantly (e.g., if the
   best config uses specialists, the video should show the specialist pipeline).

3. **Update `docs/architecture.md`** with the final pipeline diagram.

4. **Push to GitHub:**
   ```powershell
   git add -A
   git commit -m "Phase 4: Final pipeline configuration - XX% top-1 accuracy"
   git push
   ```

### Step 4.5: Record Demo Video

Follow `docs/video_script.md` with the FINAL pipeline configuration running live.

### Step 4.6: Submit on Kaggle

Follow `docs/kaggle_writeup.md` submission steps. Include:
- Final writeup with experimental results
- Video link
- GitHub repo link
- (Optional) Live demo URL if deployed

---

## Decision Log

Use this section to record key decisions as you execute the plan.

### Phase 1 Results
```
Date: ___________

B* = ___________    accuracy: ____%    delta: +____%    latency: ____ms
C* = ___________    accuracy: ____%    delta: +____%    avg_iters: ____
D* = ___________    accuracy: ____%    delta: +____%    cost/case: $____

Best single axis: Track ___
Notes:
```

### Phase 2 Results
```
Date: ___________

F* = ___________    accuracy: ____%    delta: +____%
G* = ___________    accuracy: ____%    delta: +____%    cost: ____×
H* = ___________    accuracy: ____%    delta: +____%

Ranked axes (by lift):
1. ___  2. ___  3. ___  4. ___  5. ___  6. ___

Notes:
```

### Phase 3 Results
```
Date: ___________

E1 accuracy: ____%    cost/case: $____
E2 accuracy: ____%    cost/case: $____
E3 accuracy: ____%    cost/case: $____

Best pair: ___ + ___  accuracy: ____%
Best triple: ___ + ___ + ___  accuracy: ____%

Notes:
```

### Phase 4 Final
```
Date: ___________

Final config: ___________________________
Final accuracy (50-case): ____%
Final accuracy (100-case): ____%
Cost per case: $____
Runtime per case: ____s

Submitted: [ ] Yes  [ ] No
Video recorded: [ ] Yes  [ ] No
```

---

## Time Budget

| Phase | Estimated Endpoint Hours | Estimated Wall Clock | Estimated Cost |
|-------|-------------------------|---------------------|---------------|
| Phase 1 (B+C+D) | 8–12 hrs | 1–2 days | $20–30 |
| Phase 2 (F+G+H) | 6–10 hrs | 1–2 days | $15–25 |
| Phase 3 (Compositions) | 4–8 hrs | 1 day | $10–20 |
| Phase 4 (Finalize) | 2–3 hrs | 1 day | $5–8 |
| **Total** | **20–33 hrs** | **4–7 days** | **$50–83** |

**Deadline:** February 24, 2026, 11:59 PM UTC
**Today:** February 15, 2026
**Available:** ~9 days

**Suggested schedule:**
- Feb 15–16: Phase 1 (run overnight, collect in morning)
- Feb 17–18: Phase 2 (build F/G/H, run overnight)
- Feb 19–20: Phase 3 (compositions)
- Feb 21–22: Phase 4 (finalize, video, writeup update)
- Feb 23: Buffer day + final submission
- Feb 24: Deadline

---

## Abort Conditions

Stop and re-evaluate the strategy if:

1. **Endpoint costs exceed $100 total** — we're overspending for marginal gains
2. **All Phase 1 tracks show <2% lift** — the model, not the pipeline, is the bottleneck. Consider:
   - Switching to `medgemma-4b-it` for faster iteration on prompts
   - Focusing entirely on prompt architecture (Track F)
   - Reducing scope to best-effort with current accuracy + strong writeup
3. **Phase 3 compositions LOSE accuracy vs single tracks** — negative interaction effects. Simplify back to best single track.
4. **Consistent pipeline failures (>10% error rate)** — endpoint stability issue. Fix infrastructure before continuing experiments.
5. **February 22 reached without Phase 3 complete** — lock whatever is best so far and move directly to Phase 4 (finalize + submit). Do not risk missing the deadline for marginal gains.
