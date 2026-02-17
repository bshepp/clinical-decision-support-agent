# VALIDATION_PIPELINE_PLAN.md — Validation Pipeline Fix Plan

> **Purpose:** Step-by-step implementation plan for fixing the validation/scoring
> pipeline so accuracy metrics actually reflect the system's capabilities.
>
> **Root cause:** The pipeline forces every MedQA question through differential
> diagnosis generation, but only 7/50 sampled questions are diagnostic. The other
> 43 are treatment, mechanism, lab-finding, ethics, etc. — producing near-zero
> accuracy on questions the pipeline was never designed to answer.
>
> **Expected impact:** Fixes P5+P3+P6 alone should raise measured MedQA accuracy
> from ~36% to 60-70%+. Full implementation of all 7 fixes gives honest,
> stratified metrics and unlocks multi-mode pipeline expansion.
>
> **Implementation order:** Bottom-up through the data flow. Each step locks down
> its interface before the next layer builds on it. No rewrites needed.

---

## Step 1: P5 — Fix `fuzzy_match()` for Short Answers

**File:** `src/backend/validation/base.py`  
**Functions:** `fuzzy_match()`, `normalize_text()`  
**Depends on:** Nothing  
**Depended on by:** P4 (type-aware scoring), P6 (MCQ selection comparison)

### Problem

`fuzzy_match()` uses `min(len(c_tokens), len(t_tokens))` as the denominator for
token overlap. For a 1-word target like "Clopidogrel", `min(1, 200) = 1`, so a
single token match gives 100% overlap. But for a 3-word target like "Cross-linking
of DNA", stop-word removal and normalization can reduce the target to 2 tokens,
and if the candidate doesn't contain those specific tokens, it fails — even if
the concept is present in different phrasing.

The substring check (`normalize_text(target) in normalize_text(candidate)`) works
for exact matches but fails for any morphological variation: "clopidogrel 75mg"
won't substring-match "Clopidogrel" because the candidate is longer.

Wait — actually the current code does `normalize_text(target) in normalize_text(candidate)`,
which WOULD match "clopidogrel" inside "clopidogrel 75mg daily". The real failure
case is when the answer uses different phrasing than the pipeline output, e.g.:
- Target: "Reassurance and continuous monitoring" 
- Pipeline says: "reassure the patient and monitor continuously"
- Neither substring contains the other, and token overlap may be low

### Changes

```python
# In base.py — replace fuzzy_match() entirely

def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, normalize whitespace."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# Medical stopwords that don't carry diagnostic meaning
_MEDICAL_STOPWORDS = frozenset({
    "the", "a", "an", "of", "in", "to", "and", "or", "is", "are", "was",
    "were", "be", "been", "with", "for", "on", "at", "by", "from", "this",
    "that", "these", "those", "it", "its", "has", "have", "had", "do",
    "does", "did", "will", "would", "could", "should", "may", "might",
    "most", "likely", "following", "which", "what", "patient", "patients",
})


def _content_tokens(text: str) -> set[str]:
    """Extract meaningful content tokens, removing medical stopwords."""
    tokens = set(normalize_text(text).split())
    return tokens - _MEDICAL_STOPWORDS


def fuzzy_match(candidate: str, target: str, threshold: float = 0.6) -> bool:
    """
    Check if candidate text is a fuzzy match for target.

    Strategy (checked in order, first match wins):
      1. Normalized substring containment (either direction)
      2. All content tokens of target appear in candidate (recall=1.0)
      3. Token overlap ratio >= threshold (using content tokens)

    Args:
        candidate: Text from the pipeline output (may be long)
        target: Ground truth text (usually short)
        threshold: Minimum token overlap ratio (0.0-1.0)
    """
    c_norm = normalize_text(candidate)
    t_norm = normalize_text(target)

    if not t_norm:
        return False

    # 1. Substring containment (either direction)
    if t_norm in c_norm or c_norm in t_norm:
        return True

    # 2. All content tokens of target present in candidate
    #    This catches "clopidogrel" in a 500-word report
    t_content = _content_tokens(target)
    c_content = _content_tokens(candidate)

    if t_content and t_content.issubset(c_content):
        return True

    # 3. Token overlap ratio
    if not t_content or not c_content:
        return False

    overlap = len(t_content & c_content)
    # Use target token count as denominator — "what fraction of
    # the target's meaning is present in the candidate?"
    recall = overlap / len(t_content)

    return recall >= threshold
```

### Key interface change

- **Signature stays the same:** `fuzzy_match(candidate, target, threshold) -> bool`
- **Behavior change:** More permissive matching for short targets (all-token-subset check),
  slightly different threshold semantics (recall-based instead of min-denominator-based).
  This is strictly better — no downstream code breaks.

### Tests to write

```python
# test_fuzzy_match.py
def test_short_target_substring():
    assert fuzzy_match("Start clopidogrel 75mg daily", "Clopidogrel") == True

def test_short_target_all_tokens():
    assert fuzzy_match("The diagnosis is cholesterol embolization syndrome", "Cholesterol embolization") == True

def test_multi_word_phrasing_variation():
    # "Reassurance and continuous monitoring" vs report text
    assert fuzzy_match(
        "reassure the patient and provide continuous cardiac monitoring",
        "Reassurance and continuous monitoring"
    ) == True  # content tokens: {reassurance, continuous, monitoring} — "reassurance" != "reassure" though

def test_no_false_positive():
    assert fuzzy_match("Acute myocardial infarction", "Pulmonary embolism") == False

def test_empty_target():
    assert fuzzy_match("some text", "") == False
```

**Note:** The "reassurance" vs "reassure" case will still fail without stemming.
Add stemming as a future enhancement (e.g., via `nltk.stem.PorterStemmer` or a
simple suffix-stripping function). For now, the all-token-subset check is the
biggest improvement.

### Validation

Run existing test suite — no existing tests should break because matching is
strictly more permissive. Verify on a few known failure cases from the 50-case
run results.

---

## Step 2: P3 — Preserve the Question Stem

**File:** `src/backend/validation/harness_medqa.py`  
**Functions:** `_extract_vignette()`, `fetch_medqa()`  
**Depends on:** Nothing (independent of P5, but listed second for logical flow)  
**Depended on by:** P1 (classifier needs the stem), P6 (MCQ step needs the stem + options)

### Problem

`_extract_vignette()` strips the question stem ("Which of the following is the
most likely diagnosis?") from the MedQA question. This means:
1. The pipeline doesn't know what's being asked — it always defaults to
   "generate a differential"
2. The question classifier (P1) can't classify without the stem
3. The MCQ step (P6) can't present the original question

### Changes

#### 2a. Refactor `_extract_vignette()` → `_split_question()`

```python
# In harness_medqa.py — replace _extract_vignette()

def _split_question(question: str) -> tuple[str, str]:
    """
    Split a USMLE question into (clinical_vignette, question_stem).

    The vignette is the clinical narrative. The stem is the actual question
    being asked ("Which of the following is the most likely diagnosis?").

    Returns:
        (vignette, stem) — stem may be empty if no recognizable stem found.
        In that case, vignette contains the full question text.
    """
    stems = [
        r"which of the following",
        r"what is the most likely",
        r"what is the best next step",
        r"what is the most appropriate",
        r"what is the diagnosis",
        r"the most likely diagnosis is",
        r"this patient most likely has",
        r"what would be the next step",
        r"what is the next best step",
        r"what is the underlying",
        r"what is the mechanism",
        r"what is the pathophysiology",
    ]

    text = question.strip()
    for stem_pattern in stems:
        pattern = re.compile(
            rf'(\.?\s*)([A-Z][^.]*{stem_pattern}[^.]*[\?\.]?\s*)$',
            re.IGNORECASE,
        )
        match = pattern.search(text)
        if match:
            vignette = text[:match.start()].strip()
            stem_text = match.group(2).strip()
            if len(vignette) > 50:  # Sanity check
                return vignette, stem_text

    # Fallback: no recognizable stem — return full text as vignette
    return text, ""
```

#### 2b. Update `fetch_medqa()` to store stem + vignette separately

```python
# In fetch_medqa(), replace the case-building loop body:

        vignette, question_stem = _split_question(question)

        cases.append(ValidationCase(
            case_id=f"medqa_{i:04d}",
            source_dataset="medqa",
            input_text=vignette,           # Pipeline still gets the vignette
            ground_truth={
                "correct_answer": answer_text,
                "answer_idx": answer_idx,
                "options": options,
                "full_question": question,
            },
            metadata={
                "question_stem": question_stem,      # NEW
                "clinical_vignette": vignette,       # NEW (same as input_text, explicit)
                "full_question_with_stem": question,  # NEW (redundant with ground_truth but cleaner access)
            },
        ))
```

### Key interface change

- `ValidationCase.metadata` now has 3 new keys: `question_stem`, `clinical_vignette`,
  `full_question_with_stem`
- `input_text` is still just the vignette (pipeline input unchanged)
- `_extract_vignette()` is renamed to `_split_question()` returning a tuple
- Old callers of `_extract_vignette()`: only `fetch_medqa()` — update in place

### Backward compatibility

- `input_text` stays the same → pipeline behavior unchanged
- `ground_truth` keeps all existing keys → scoring unchanged
- New data is in `metadata` only → nothing breaks

---

## Step 3: P1 — Question-Type Classifier

**New file:** `src/backend/validation/question_classifier.py`  
**Depends on:** P3 (needs `metadata["question_stem"]`)  
**Depended on by:** P4 (type-aware scoring), P6 (routing), P7 (stratified reporting)

### Design

Two-tier classifier:
1. **Heuristic classifier** (fast, no LLM call, used by default) — regex on question stem
2. **LLM classifier** (optional, for ambiguous cases) — ask MedGemma to classify

Start with heuristic only. It correctly classified our 50-case sample already
(7 diagnostic, 6 treatment, 1 mechanism, 2 lab, 34 other — matching manual review).

### Question type enum

```python
# In question_classifier.py

from enum import Enum

class QuestionType(str, Enum):
    DIAGNOSTIC = "diagnostic"           # "most likely diagnosis/cause/explanation"
    TREATMENT = "treatment"             # "most appropriate next step/management/treatment"
    MECHANISM = "mechanism"             # "mechanism of action", "pathophysiology"
    LAB_FINDING = "lab_finding"         # "expected finding", "characteristic on agar"
    PHARMACOLOGY = "pharmacology"       # "drug that targets...", "receptor..."
    EPIDEMIOLOGY = "epidemiology"       # "risk factor", "prevalence", "incidence"
    ETHICS = "ethics"                   # "most appropriate action" (ethical dilemmas)
    ANATOMY = "anatomy"                 # "structure most likely damaged"
    OTHER = "other"                     # Doesn't fit above categories
```

### Heuristic classifier

```python
import re
from typing import Optional
from validation.base import ValidationCase


# Pattern → QuestionType mapping (checked in order, first match wins)
_STEM_PATTERNS: list[tuple[str, QuestionType]] = [
    # Diagnostic
    (r"most likely diagnosis", QuestionType.DIAGNOSTIC),
    (r"most likely cause", QuestionType.DIAGNOSTIC),
    (r"most likely explanation", QuestionType.DIAGNOSTIC),
    (r"what is the diagnosis", QuestionType.DIAGNOSTIC),
    (r"diagnosis is", QuestionType.DIAGNOSTIC),
    (r"most likely condition", QuestionType.DIAGNOSTIC),
    (r"most likely has", QuestionType.DIAGNOSTIC),
    (r"most likely suffer", QuestionType.DIAGNOSTIC),

    # Treatment / Management
    (r"most appropriate (next step|management|treatment|intervention|therapy|pharmacotherapy)", QuestionType.TREATMENT),
    (r"best (next step|initial step|management|treatment)", QuestionType.TREATMENT),
    (r"most appropriate action", QuestionType.TREATMENT),  # Can be ethics — see below
    (r"recommended (treatment|management|therapy)", QuestionType.TREATMENT),

    # Mechanism
    (r"mechanism of action", QuestionType.MECHANISM),
    (r"pathophysiology", QuestionType.MECHANISM),
    (r"mediator.*(responsible|involved)", QuestionType.MECHANISM),
    (r"(inhibit|block|activate).*receptor", QuestionType.MECHANISM),
    (r"cross-link", QuestionType.MECHANISM),

    # Lab / Findings
    (r"most likely finding", QuestionType.LAB_FINDING),
    (r"expected (finding|result|value)", QuestionType.LAB_FINDING),
    (r"characteristic (finding|feature|appearance)", QuestionType.LAB_FINDING),
    (r"(agar|culture|stain|gram|biopsy).*show", QuestionType.LAB_FINDING),
    (r"(laboratory|lab).*(result|finding|value)", QuestionType.LAB_FINDING),

    # Pharmacology
    (r"drug.*(target|mechanism|receptor|inhibit)", QuestionType.PHARMACOLOGY),
    (r"(target|act on|bind).*(receptor|enzyme|channel)", QuestionType.PHARMACOLOGY),

    # Epidemiology
    (r"(risk factor|prevalence|incidence|odds ratio|relative risk)", QuestionType.EPIDEMIOLOGY),
    (r"most (common|frequent).*(cause|risk|complication)", QuestionType.EPIDEMIOLOGY),

    # Anatomy
    (r"(structure|nerve|artery|vein|muscle|ligament).*(damaged|injured|affected|involved)", QuestionType.ANATOMY),

    # Ethics (refine: "most appropriate action" in context of disclosure, consent, etc.)
    (r"(tell|inform|disclose|report|consent|refuse|autonomy|confidentiality)", QuestionType.ETHICS),
]


def classify_question(case: ValidationCase) -> QuestionType:
    """
    Classify a MedQA question by type using heuristics on the question stem.

    Looks at metadata["question_stem"] first, falls back to ground_truth["full_question"].

    Returns:
        QuestionType enum value
    """
    stem = case.metadata.get("question_stem", "")
    full_q = case.ground_truth.get("full_question", case.input_text)

    # Classify on stem first (more specific), then full question
    for text in [stem, full_q]:
        text_lower = text.lower()
        for pattern, qtype in _STEM_PATTERNS:
            if re.search(pattern, text_lower):
                return qtype

    return QuestionType.OTHER


def classify_question_from_text(question_text: str) -> QuestionType:
    """
    Classify a raw question string (no ValidationCase needed).
    Useful for ad-hoc classification.
    """
    text_lower = question_text.lower()
    for pattern, qtype in _STEM_PATTERNS:
        if re.search(pattern, text_lower):
            return qtype
    return QuestionType.OTHER


# Convenience: which types are "pipeline-appropriate"?
DIAGNOSTIC_TYPES = {QuestionType.DIAGNOSTIC}
PIPELINE_APPROPRIATE_TYPES = {
    QuestionType.DIAGNOSTIC,
    QuestionType.TREATMENT,
    QuestionType.LAB_FINDING,
}
```

### Integration point

In `fetch_medqa()`, after building each case, classify it:

```python
from validation.question_classifier import classify_question

# After creating the ValidationCase:
case.metadata["question_type"] = classify_question(case).value
```

### Tests

```python
def test_diagnostic_classification():
    case = make_case(question="...What is the most likely diagnosis?")
    assert classify_question(case) == QuestionType.DIAGNOSTIC

def test_treatment_classification():
    case = make_case(question="...What is the most appropriate next step in management?")
    assert classify_question(case) == QuestionType.TREATMENT

def test_mechanism_classification():
    case = make_case(question="...mechanism of action...")
    assert classify_question(case) == QuestionType.MECHANISM

def test_ethics_override():
    # "most appropriate action" + disclosure keywords → ethics, not treatment
    case = make_case(question="...Tell the attending that he cannot fail to disclose this mistake. What is the most appropriate action?")
    assert classify_question(case) == QuestionType.ETHICS
```

**Note on ethics override:** The pattern order matters. "most appropriate action"
will match TREATMENT first. To handle ethics, we need the ethics patterns to check
for disclosure/consent keywords in the *answer* or full question context. The
current design checks patterns in order — put ethics keyword patterns before the
generic "most appropriate action" treatment pattern, OR do a two-pass: first check
for ethics keywords, then fall through to treatment.

**Decision:** Use a two-pass approach. If the question contains ethics keywords
AND a treatment-like stem, classify as ETHICS. Otherwise classify as TREATMENT.
Implement this in `classify_question()` with a special-case check.

---

## Step 4: P4 — Question-Type-Aware Scoring

**File:** `src/backend/validation/base.py` (new function) + `src/backend/validation/harness_medqa.py` (refactor scoring block)  
**Depends on:** P5 (correct fuzzy_match), P1 (question_type in metadata)  
**Depended on by:** P7 (stratified reporting)

### Problem

`diagnosis_in_differential()` always searches the same fields in the same order
regardless of question type. Treatment answers get looked up in the differential
(wrong place), and mechanism answers get looked up everywhere (unlikely to match).

### Design: `score_case()` dispatcher

```python
# In base.py — new function alongside diagnosis_in_differential()

def score_case(
    target_answer: str,
    report: CDSReport,
    question_type: str = "diagnostic",
    reasoning_result: Optional[ClinicalReasoningResult] = None,
) -> dict[str, float]:
    """
    Score a case based on its question type.

    Returns a dict of metric_name → score (0.0 or 1.0).
    Always includes: "matched", "match_location", "match_rank"
    Plus type-specific metrics.
    """
    qt = question_type.lower()

    if qt == "diagnostic":
        return _score_diagnostic(target_answer, report)
    elif qt == "treatment":
        return _score_treatment(target_answer, report)
    elif qt == "mechanism":
        return _score_mechanism(target_answer, report, reasoning_result)
    elif qt == "lab_finding":
        return _score_lab_finding(target_answer, report, reasoning_result)
    else:
        return _score_generic(target_answer, report, reasoning_result)
```

### Per-type scorers

```python
def _score_diagnostic(target: str, report: CDSReport) -> dict:
    """Score a diagnostic question — primary field is differential_diagnosis."""
    found_top1, r1, l1 = diagnosis_in_differential(target, report, top_n=1)
    found_top3, r3, l3 = diagnosis_in_differential(target, report, top_n=3)
    found_any, ra, la = diagnosis_in_differential(target, report)

    return {
        "top1_accuracy": 1.0 if found_top1 else 0.0,
        "top3_accuracy": 1.0 if found_top3 else 0.0,
        "mentioned_accuracy": 1.0 if found_any else 0.0,
        "differential_accuracy": 1.0 if (found_any and la == "differential") else 0.0,
        "match_location": la,
        "match_rank": ra,
    }


def _score_treatment(target: str, report: CDSReport) -> dict:
    """Score a treatment question — primary fields are next_steps + recommendations."""
    # Check suggested_next_steps first (most specific)
    for i, action in enumerate(report.suggested_next_steps):
        if fuzzy_match(action.action, target):
            return {
                "top1_accuracy": 1.0 if i == 0 else 0.0,
                "top3_accuracy": 1.0 if i < 3 else 0.0,
                "mentioned_accuracy": 1.0,
                "match_location": "next_steps",
                "match_rank": i,
            }

    # Check guideline_recommendations
    for i, rec in enumerate(report.guideline_recommendations):
        if fuzzy_match(rec, target):
            return {
                "top1_accuracy": 0.0,  # Not in primary slot
                "top3_accuracy": 0.0,
                "mentioned_accuracy": 1.0,
                "match_location": "recommendations",
                "match_rank": i,
            }

    # Check differential reasoning text (treatment may appear in reasoning)
    for dx in report.differential_diagnosis:
        if fuzzy_match(dx.reasoning, target, threshold=0.3):
            return {
                "top1_accuracy": 0.0,
                "top3_accuracy": 0.0,
                "mentioned_accuracy": 1.0,
                "match_location": "reasoning_text",
                "match_rank": -1,
            }

    # Fulltext fallback
    full_text = _build_fulltext(report)
    if fuzzy_match(full_text, target, threshold=0.3):
        return {
            "top1_accuracy": 0.0,
            "top3_accuracy": 0.0,
            "mentioned_accuracy": 1.0,
            "match_location": "fulltext",
            "match_rank": -1,
        }

    return _not_found()


def _score_mechanism(
    target: str, report: CDSReport,
    reasoning_result: Optional[ClinicalReasoningResult] = None,
) -> dict:
    """Score a mechanism question — primary field is reasoning_chain."""
    # Check reasoning chain from clinical reasoning step
    if reasoning_result and reasoning_result.reasoning_chain:
        if fuzzy_match(reasoning_result.reasoning_chain, target, threshold=0.3):
            return {
                "top1_accuracy": 0.0,
                "top3_accuracy": 0.0,
                "mentioned_accuracy": 1.0,
                "match_location": "reasoning_chain",
                "match_rank": -1,
            }

    # Check differential reasoning text
    for dx in report.differential_diagnosis:
        if fuzzy_match(dx.reasoning, target, threshold=0.3):
            return {
                "top1_accuracy": 0.0,
                "top3_accuracy": 0.0,
                "mentioned_accuracy": 1.0,
                "match_location": "differential_reasoning",
                "match_rank": -1,
            }

    # Fulltext fallback
    full_text = _build_fulltext(report)
    if fuzzy_match(full_text, target, threshold=0.3):
        return {
            "top1_accuracy": 0.0,
            "top3_accuracy": 0.0,
            "mentioned_accuracy": 1.0,
            "match_location": "fulltext",
            "match_rank": -1,
        }

    return _not_found()


def _score_lab_finding(
    target: str, report: CDSReport,
    reasoning_result: Optional[ClinicalReasoningResult] = None,
) -> dict:
    """Score a lab/finding question — primary field is recommended_workup."""
    # Check recommended workup
    if reasoning_result:
        for i, action in enumerate(reasoning_result.recommended_workup):
            if fuzzy_match(action.action, target, threshold=0.4):
                return {
                    "top1_accuracy": 1.0 if i == 0 else 0.0,
                    "top3_accuracy": 1.0 if i < 3 else 0.0,
                    "mentioned_accuracy": 1.0,
                    "match_location": "recommended_workup",
                    "match_rank": i,
                }

    # Check next steps in final report
    for i, action in enumerate(report.suggested_next_steps):
        if fuzzy_match(action.action, target, threshold=0.4):
            return {
                "top1_accuracy": 0.0,
                "top3_accuracy": 0.0,
                "mentioned_accuracy": 1.0,
                "match_location": "next_steps",
                "match_rank": i,
            }

    # Fulltext fallback
    full_text = _build_fulltext(report)
    if fuzzy_match(full_text, target, threshold=0.3):
        return {
            "top1_accuracy": 0.0,
            "top3_accuracy": 0.0,
            "mentioned_accuracy": 1.0,
            "match_location": "fulltext",
            "match_rank": -1,
        }

    return _not_found()


def _score_generic(
    target: str, report: CDSReport,
    reasoning_result: Optional[ClinicalReasoningResult] = None,
) -> dict:
    """Score any question type — searches all fields broadly."""
    # Try all specific scorers, return first hit
    for scorer in [_score_diagnostic, _score_treatment]:
        result = scorer(target, report)
        if result.get("mentioned_accuracy", 0.0) > 0.0:
            return result

    if reasoning_result:
        result = _score_mechanism(target, report, reasoning_result)
        if result.get("mentioned_accuracy", 0.0) > 0.0:
            return result

    return _not_found()


def _build_fulltext(report: CDSReport) -> str:
    """Concatenate all report fields into a single searchable string."""
    return " ".join([
        report.patient_summary or "",
        " ".join(report.guideline_recommendations),
        " ".join(a.action for a in report.suggested_next_steps),
        " ".join(dx.diagnosis + " " + dx.reasoning for dx in report.differential_diagnosis),
        " ".join(report.sources_cited),
        " ".join(c.description for c in report.conflicts),
    ])


def _not_found() -> dict:
    return {
        "top1_accuracy": 0.0,
        "top3_accuracy": 0.0,
        "mentioned_accuracy": 0.0,
        "match_location": "not_found",
        "match_rank": -1,
    }
```

### Integration in harness_medqa.py

Replace the scoring block (lines ~242-290) in `validate_medqa()`:

```python
# OLD:
#   found_top1, rank1, loc1 = diagnosis_in_differential(correct_answer, report, top_n=1)
#   ...etc...

# NEW:
question_type = case.metadata.get("question_type", "other")
scores = score_case(
    target_answer=correct_answer,
    report=report,
    question_type=question_type,
    reasoning_result=state.clinical_reasoning if state else None,
)
# Extract individual metrics from the dict
scores["parse_success"] = 1.0
```

### Key interface

- `score_case()` returns `dict[str, float]` — always includes `top1_accuracy`,
  `top3_accuracy`, `mentioned_accuracy`, `match_location`, `match_rank`
- The harness doesn't need to know about question type internals — just passes
  the string through
- `diagnosis_in_differential()` is NOT removed — it's still used internally by
  `_score_diagnostic()` and as a utility

---

## Step 5: P6 — MCQ Answer-Selection Step

**File:** `src/backend/validation/harness_medqa.py` (new function + integration)  
**Depends on:** P3 (question stem + options stored in metadata/ground_truth)  
**Depended on by:** P7 (reporting), but can be integrated independently

### Design

After the pipeline generates its report, present MedGemma with the original
question + answer choices + the pipeline's analysis, and ask it to select
the best answer choice.

```python
# In harness_medqa.py — new function

from app.services.medgemma import MedGemmaService


MCQ_SELECTION_PROMPT = """You are a medical expert taking a USMLE-style exam.

You have already performed a thorough clinical analysis of this case.
Now, based on your analysis, select the single best answer from the choices below.

CLINICAL VIGNETTE:
{vignette}

QUESTION:
{question_stem}

YOUR CLINICAL ANALYSIS:
- Top diagnoses: {top_diagnoses}
- Key reasoning: {reasoning_summary}
- Recommended next steps: {next_steps}
- Guideline recommendations: {recommendations}

ANSWER CHOICES:
{formatted_options}

Based on your clinical analysis above, which answer choice (A, B, C, or D)
is BEST supported? Reply with ONLY the letter (A, B, C, or D) and a one-sentence justification.

Format: X) Justification"""


async def select_mcq_answer(
    case: ValidationCase,
    report: CDSReport,
    state: Optional[AgentState] = None,
) -> tuple[str, str]:
    """
    Use MedGemma to select the best MCQ answer given the pipeline's analysis.

    Args:
        case: The validation case (must have options in ground_truth)
        report: The CDS pipeline output
        state: Full agent state (for reasoning_chain access)

    Returns:
        (selected_letter, justification) — e.g. ("B", "Consistent with...")
    """
    options = case.ground_truth.get("options", {})
    if not options:
        return "", "No options available"

    # Format options
    if isinstance(options, dict):
        formatted = "\n".join(f"{k}) {v}" for k, v in sorted(options.items()))
    else:
        formatted = "\n".join(
            f"{chr(65+i)}) {v}" for i, v in enumerate(options)
        )

    # Build context from report
    top_dx = [dx.diagnosis for dx in report.differential_diagnosis[:3]]
    reasoning = ""
    if state and state.clinical_reasoning:
        reasoning = state.clinical_reasoning.reasoning_chain[:500]
    next_steps = [a.action for a in report.suggested_next_steps[:3]]
    recommendations = report.guideline_recommendations[:3]

    vignette = case.metadata.get("clinical_vignette", case.input_text)
    stem = case.metadata.get("question_stem", "")

    prompt = MCQ_SELECTION_PROMPT.format(
        vignette=vignette[:1000],
        question_stem=stem or "Based on the clinical presentation, select the best answer.",
        top_diagnoses=", ".join(top_dx) if top_dx else "None generated",
        reasoning_summary=reasoning[:500] if reasoning else "Not available",
        next_steps=", ".join(next_steps) if next_steps else "None",
        recommendations=", ".join(recommendations) if recommendations else "None",
        formatted_options=formatted,
    )

    service = MedGemmaService()
    raw = await service.generate(
        prompt=prompt,
        system_prompt="You are a medical expert. Select the single best answer.",
        max_tokens=100,
        temperature=0.1,
    )

    # Parse response — look for a letter A-D
    selected = ""
    justification = raw.strip()
    for char in raw.strip()[:5]:
        if char.upper() in "ABCD":
            selected = char.upper()
            break

    return selected, justification


def score_mcq_selection(
    selected_letter: str,
    correct_idx: str,
) -> float:
    """Return 1.0 if selected matches correct, else 0.0."""
    return 1.0 if selected_letter.upper() == correct_idx.upper() else 0.0
```

### Integration in validate_medqa()

After the existing scoring block, add:

```python
# MCQ selection (optional additional scoring)
if report and case.ground_truth.get("options"):
    try:
        selected, justification = await select_mcq_answer(case, report, state)
        scores["mcq_accuracy"] = score_mcq_selection(
            selected, case.ground_truth["answer_idx"]
        )
        details["mcq_selected"] = selected
        details["mcq_justification"] = justification
        details["mcq_correct"] = case.ground_truth["answer_idx"]
    except Exception as e:
        logger.warning(f"MCQ selection failed: {e}")
        scores["mcq_accuracy"] = 0.0
```

### Cost consideration

This adds 1 extra MedGemma call per case (~100 tokens output). For 50 cases,
that's ~5,000 extra output tokens — negligible cost (<$0.10).

### Key interface

- `select_mcq_answer()` is self-contained — can be called or skipped
- Adds `mcq_accuracy` to the scores dict
- Does NOT change any existing score calculations

---

## Step 6: P7 — Stratified Reporting

**File:** `src/backend/validation/base.py` (modify `print_summary`, `save_results`)  
+ `src/backend/validation/harness_medqa.py` (modify aggregation block)  
**Depends on:** P1 (question types), P4 (per-type scores)  
**Depended on by:** Nothing (terminal node)

### Changes to summary aggregation in validate_medqa()

```python
# In validate_medqa() — replace the aggregation block at the end

    # Aggregate — overall
    total = len(results)
    successful = sum(1 for r in results if r.success)

    metric_names = [
        "top1_accuracy", "top3_accuracy", "mentioned_accuracy",
        "differential_accuracy", "parse_success", "mcq_accuracy",
    ]
    metrics = {}
    for m in metric_names:
        values = [r.scores.get(m, 0.0) for r in results if m in r.scores]
        metrics[m] = sum(values) / len(values) if values else 0.0

    # Average pipeline time
    times = [r.pipeline_time_ms for r in results if r.success]
    metrics["avg_pipeline_time_ms"] = sum(times) / len(times) if times else 0

    # ── Stratified metrics ──
    from validation.question_classifier import QuestionType, PIPELINE_APPROPRIATE_TYPES

    # Group results by question type
    by_type: dict[str, list[ValidationResult]] = {}
    for r in results:
        qt = r.details.get("question_type", "other")
        by_type.setdefault(qt, []).append(r)

    # Per-type metrics
    for qt, type_results in by_type.items():
        n = len(type_results)
        metrics[f"count_{qt}"] = n
        for m in ["top1_accuracy", "top3_accuracy", "mentioned_accuracy", "mcq_accuracy"]:
            values = [r.scores.get(m, 0.0) for r in type_results if m in r.scores]
            if values:
                metrics[f"{m}_{qt}"] = sum(values) / len(values)

    # Pipeline-appropriate subset
    appropriate_results = [
        r for r in results
        if r.details.get("question_type", "other") in {t.value for t in PIPELINE_APPROPRIATE_TYPES}
    ]
    if appropriate_results:
        for m in ["top1_accuracy", "top3_accuracy", "mentioned_accuracy"]:
            values = [r.scores.get(m, 0.0) for r in appropriate_results]
            metrics[f"{m}_pipeline_appropriate"] = sum(values) / len(values) if values else 0.0
        metrics["count_pipeline_appropriate"] = len(appropriate_results)
```

### Changes to print_summary()

```python
# In base.py — enhanced print_summary()

def print_summary(summary: ValidationSummary):
    """Pretty-print validation results to console."""
    print(f"\n{'='*60}")
    print(f"  Validation Results: {summary.dataset.upper()}")
    print(f"{'='*60}")
    print(f"  Total cases:      {summary.total_cases}")
    print(f"  Successful:       {summary.successful_cases}")
    print(f"  Failed:           {summary.failed_cases}")
    print(f"  Duration:         {summary.run_duration_sec:.1f}s")

    # Overall metrics (exclude per-type and count metrics)
    print(f"\n  Overall Metrics:")
    for metric, value in sorted(summary.metrics.items()):
        if "_" in metric and any(metric.endswith(f"_{qt}") for qt in
            ["diagnostic", "treatment", "mechanism", "lab_finding",
             "pharmacology", "epidemiology", "ethics", "anatomy", "other",
             "pipeline_appropriate"]):
            continue  # Print these in stratified section
        if metric.startswith("count_"):
            continue
        if "time" in metric and isinstance(value, (int, float)):
            print(f"    {metric:35s} {value:.0f}ms")
        elif isinstance(value, float):
            print(f"    {metric:35s} {value:.1%}")
        else:
            print(f"    {metric:35s} {value}")

    # Stratified metrics
    type_keys = sorted(set(
        k.rsplit("_", 1)[-1] for k in summary.metrics
        if k.startswith("count_") and k != "count_pipeline_appropriate"
    ))
    if type_keys:
        print(f"\n  By Question Type:")
        print(f"    {'Type':15s} {'Count':>6s} {'Top-1':>7s} {'Top-3':>7s} {'Mentioned':>10s} {'MCQ':>7s}")
        print(f"    {'-'*15} {'-'*6} {'-'*7} {'-'*7} {'-'*10} {'-'*7}")
        for qt in type_keys:
            count = summary.metrics.get(f"count_{qt}", 0)
            t1 = summary.metrics.get(f"top1_accuracy_{qt}", None)
            t3 = summary.metrics.get(f"top3_accuracy_{qt}", None)
            ma = summary.metrics.get(f"mentioned_accuracy_{qt}", None)
            mcq = summary.metrics.get(f"mcq_accuracy_{qt}", None)
            print(f"    {qt:15s} {int(count):6d} "
                  f"{f'{t1:.0%}':>7s if t1 is not None else '   -   '} "
                  f"{f'{t3:.0%}':>7s if t3 is not None else '   -   '} "
                  f"{f'{ma:.0%}':>10s if ma is not None else '     -     '} "
                  f"{f'{mcq:.0%}':>7s if mcq is not None else '   -   '}")

    # Pipeline-appropriate subset
    pa_count = summary.metrics.get("count_pipeline_appropriate", 0)
    if pa_count > 0:
        print(f"\n  Pipeline-Appropriate Subset ({int(pa_count)} cases):")
        for m in ["top1_accuracy", "top3_accuracy", "mentioned_accuracy"]:
            v = summary.metrics.get(f"{m}_pipeline_appropriate")
            if v is not None:
                print(f"    {m:35s} {v:.1%}")

    print(f"{'='*60}\n")
```

### Key interface

- `ValidationSummary.metrics` dict gains new keys with `_{question_type}` suffixes
- `save_results()` doesn't need changes — it serializes `metrics` as-is
- Console output is richer but backward-compatible (old scripts parsing the JSON
  still see all the original keys)

---

## Step 7: P2 — Multi-Mode Pipeline (Large — Future)

**Files:** `src/backend/app/agent/orchestrator.py`, `src/backend/app/tools/clinical_reasoning.py`, `src/backend/app/models/schemas.py`  
**Depends on:** P1 (question type routing into the pipeline), P3 (question stem passed to pipeline)  
**Depended on by:** Nothing (this is the final architectural evolution)

### Overview

This is the biggest change and should be done LAST. It modifies the production
pipeline, not just the validation framework.

### 7a. Add `question_context` to `CaseSubmission`

```python
# In schemas.py — extend CaseSubmission

class CaseSubmission(BaseModel):
    patient_text: str = Field(..., min_length=10)
    include_drug_check: bool = Field(True)
    include_guidelines: bool = Field(True)
    question_context: Optional[str] = Field(
        None,
        description="The clinical question being asked (e.g., 'What is the most likely diagnosis?'). "
                    "If provided, the pipeline adapts its reasoning mode.",
    )
    question_type: Optional[str] = Field(
        None,
        description="Pre-classified question type: diagnostic, treatment, mechanism, etc.",
    )
```

### 7b. Mode-specific system prompts in clinical_reasoning.py

```python
# Replace single SYSTEM_PROMPT with a dict:

SYSTEM_PROMPTS = {
    "diagnostic": """You are an expert clinical reasoning assistant...
    [existing diagnostic prompt — mostly unchanged]""",

    "treatment": """You are an expert clinical management assistant...
    Given a structured patient profile and clinical question, recommend the
    most appropriate treatment or next step in management.
    Focus on: evidence-based treatment guidelines, patient-specific factors,
    contraindications, and prioritized management steps.
    Generate a ranked list of management options (not diagnoses)...""",

    "mechanism": """You are an expert in medical pathophysiology...
    Given a clinical scenario, explain the underlying mechanism,
    pathophysiology, or pharmacological principle being tested.
    Focus on: molecular/cellular mechanism, physiological pathways,
    drug mechanisms of action...""",

    "default": """[existing SYSTEM_PROMPT as fallback]""",
}
```

### 7c. Extend clinical reasoning output model

```python
# In schemas.py — new model for non-diagnostic reasoning

class ClinicalAnalysisResult(BaseModel):
    """Flexible clinical analysis output that adapts to question type."""
    analysis_mode: str = Field("diagnostic", description="What type of analysis was performed")
    differential_diagnosis: List[DiagnosisCandidate] = Field(default_factory=list)
    management_options: List[RecommendedAction] = Field(default_factory=list)
    mechanism_explanation: str = Field("", description="Pathophysiology/mechanism explanation")
    recommended_workup: List[RecommendedAction] = Field(default_factory=list)
    reasoning_chain: str = Field("")
    risk_assessment: Optional[str] = None
    direct_answer: Optional[str] = Field(
        None,
        description="Direct answer to the clinical question (when applicable)",
    )
```

### 7d. Orchestrator routing

```python
# In orchestrator.py — _step_reason() adapts based on question type

async def _step_reason(self):
    question_type = self._case.question_type or "diagnostic"
    result = await self.clinical_reasoning.run(
        self._state.patient_profile,
        mode=question_type,
    )
    ...
```

### Scope warning

This is a multi-file, multi-model refactor. Do it only after Steps 1-6 are
working and validated. The validation improvements (Steps 1-6) will already
give us honest metrics; Step 7 is about actually improving the pipeline's ability
to handle non-diagnostic questions.

---

## Testing Strategy

### Unit tests (no LLM calls needed)

| Test file | What it tests |
|-----------|---------------|
| `test_fuzzy_match.py` | P5: fuzzy_match with short/long targets, edge cases |
| `test_question_classifier.py` | P1: classification accuracy on known questions |
| `test_split_question.py` | P3: vignette/stem separation on real MedQA samples |
| `test_score_case.py` | P4: type-aware scoring with mock CDSReport objects |

### Integration tests (need LLM endpoint)

| Test | What it tests | Cost |
|------|---------------|------|
| 3-case smoke test with MCQ | P6: MCQ selection works | ~$0.50 |
| 10-case run with stratified reporting | P7: reporting output is correct | ~$2.00 |
| 50-case full run with all fixes | All: end-to-end accuracy comparison | ~$5.00 |

### Comparison protocol

Run 50-case MedQA (seed=42) twice:
1. **Before:** Current code (baseline: 36% top-1, 38% mentioned)
2. **After:** All fixes applied

Compare:
- Overall accuracy (should be similar or slightly higher)
- Diagnostic-only accuracy (should be similar — same pipeline, better matching)
- MCQ accuracy (expected 60-70%+ — this is the big win)
- Pipeline-appropriate accuracy (expected higher than overall)
- Stratified breakdown by question type

---

## File Change Summary

| File | Changes | Step |
|------|---------|------|
| `validation/base.py` | Rewrite `fuzzy_match()`, add `_content_tokens()`, `_MEDICAL_STOPWORDS`. Add `score_case()` and per-type scorers. Modify `print_summary()`. | P5, P4, P7 |
| `validation/harness_medqa.py` | Replace `_extract_vignette()` with `_split_question()`. Update `fetch_medqa()` metadata. Refactor scoring block to use `score_case()`. Add `select_mcq_answer()`. Update aggregation. | P3, P4, P6, P7 |
| `validation/question_classifier.py` | **NEW FILE.** `QuestionType` enum, `classify_question()`, `_STEM_PATTERNS`. | P1 |
| `app/models/schemas.py` | Add `question_context`, `question_type` to `CaseSubmission`. Add `ClinicalAnalysisResult`. | P2 (Step 7 only) |
| `app/tools/clinical_reasoning.py` | Add mode-specific system prompts. Accept `mode` param. | P2 (Step 7 only) |
| `app/agent/orchestrator.py` | Route reasoning step based on question type. | P2 (Step 7 only) |

**Steps 1-6 touch only validation code.** The production pipeline is unchanged
until Step 7.
