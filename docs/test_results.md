# Test Results — CDS Agent

> Last updated after RAG expansion to 62 guidelines across 14 specialties.

---

## 1. RAG Retrieval Quality Test

**Test file:** `src/backend/test_rag_quality.py`  
**What it tests:** Whether the RAG system retrieves the correct clinical guideline for a given clinical query.  
**Methodology:** 30 clinical queries, each with an expected guideline ID. For each query, the test retrieves the top-5 guidelines from ChromaDB and checks whether the expected guideline appears in the results, and whether it scores above the relevance threshold (0.4).

### Summary

| Metric | Value |
|--------|-------|
| Total queries | 30 |
| Passed | 30 |
| Failed | 0 |
| **Pass rate** | **100%** |
| Avg relevance score | 0.639 |
| Min relevance score | 0.519 |
| Max relevance score | 0.765 |
| Top-1 accuracy | 100% (correct guideline ranked #1 for all 30 queries) |

### Results by Specialty

| Specialty | Queries | Passed | Pass Rate | Avg Relevance |
|-----------|---------|--------|-----------|---------------|
| Cardiology | 4 | 4 | 100% | 0.65 |
| Emergency Medicine | 5 | 5 | 100% | 0.62 |
| Endocrinology | 3 | 3 | 100% | 0.64 |
| Pulmonology | 2 | 2 | 100% | 0.63 |
| Neurology | 2 | 2 | 100% | 0.66 |
| Gastroenterology | 2 | 2 | 100% | 0.61 |
| Infectious Disease | 2 | 2 | 100% | 0.67 |
| Psychiatry | 2 | 2 | 100% | 0.64 |
| Pediatrics | 2 | 2 | 100% | 0.63 |
| Nephrology | 2 | 2 | 100% | 0.65 |
| Hematology | 1 | 1 | 100% | 0.62 |
| Rheumatology | 1 | 1 | 100% | 0.64 |
| OB/GYN | 1 | 1 | 100% | 0.66 |
| Other | 1 | 1 | 100% | 0.61 |

### How to Reproduce

```bash
cd src/backend
python test_rag_quality.py --rebuild --verbose
```

**Flags:**
- `--rebuild` — Rebuild ChromaDB from `clinical_guidelines.json` before testing
- `--verbose` — Print each query, expected ID, actual top result, and relevance score
- `--stats` — Print summary statistics only
- `--query "chest pain"` — Test a single ad-hoc query

---

## 2. End-to-End Pipeline Test

**Test file:** `src/backend/test_e2e.py`  
**What it tests:** Full 5-step agent pipeline from free-text input to synthesized CDS report.  
**Test case:** 62-year-old male with crushing substernal chest pain, diaphoresis, nausea, HTN history, on lisinopril + metformin + atorvastatin.

### Pipeline Step Results

| Step | Status | Duration | Key Findings |
|------|--------|----------|--------------|
| 1. Parse Patient Data | PASSED | 7.8 s | Correctly extracted: age 62, male, chest pain chief complaint, 3 medications, HTN/DM history |
| 2. Clinical Reasoning | PASSED | 21.2 s | Top differential: Acute Coronary Syndrome (ACS). Also considered: GERD, PE, aortic dissection |
| 3. Drug Interaction Check | PASSED | 11.3 s | Queried OpenFDA + RxNorm for lisinopril, metformin, atorvastatin interactions |
| 4. Guideline Retrieval | PASSED | 9.6 s | Retrieved ACC/AHA chest pain / ACS guidelines from RAG corpus |
| 5. Synthesis | PASSED | 25.3 s | Generated comprehensive CDS report with differential, warnings, guideline recommendations |

**Total pipeline time:** 75.2 s

### How to Reproduce

```bash
# Start the backend first
cd src/backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# In another terminal
cd src/backend
python test_e2e.py
```

---

## 3. Clinical Test Suite

**Test file:** `src/backend/test_clinical_cases.py`  
**What it tests:** 22 diverse clinical scenarios across 14 medical specialties.  
**Methodology:** Each case has a clinical vignette, expected keywords in the CDS report output, and specialty classification. The test submits each case through the full pipeline and validates that expected terms appear in the report.

### Test Cases

| ID | Specialty | Scenario | Key Validation Keywords |
|----|-----------|----------|------------------------|
| `cardio_acs` | Cardiology | 62M crushing chest pain | ACS, troponin, ECG |
| `cardio_afib` | Cardiology | 72F palpitations, irregular pulse | Atrial fibrillation, anticoagulation, CHA2DS2-VASc |
| `cardio_hf` | Cardiology | 68M progressive dyspnea, edema | Heart failure, BNP, diuretic |
| `neuro_stroke` | Neurology | 75M sudden left-sided weakness | Stroke, CT, tPA, NIH Stroke Scale |
| `em_sepsis` | Emergency Medicine | 45F fever, tachycardia, hypotension | Sepsis, lactate, blood cultures, fluids |
| `em_anaphylaxis` | Emergency Medicine | 28F bee sting, urticaria, wheezing | Anaphylaxis, epinephrine, airway |
| `em_polytrauma` | Emergency Medicine | 35M MVC, multiple injuries | Trauma, ATLS, FAST, C-spine |
| `endo_dka` | Endocrinology | 22F T1DM, vomiting, Kussmaul breathing | DKA, insulin, potassium, anion gap |
| `endo_thyroid_storm` | Endocrinology | 40F graves, fever, tachycardia, AMS | Thyroid storm, PTU, beta-blocker |
| `endo_adrenal` | Endocrinology | 55M weakness, hypotension, hyperpigmentation | Adrenal insufficiency, cortisol, hydrocortisone |
| `pulm_pe` | Pulmonology | 50F post-surgical, sudden dyspnea | Pulmonary embolism, CT angiography, anticoagulation |
| `pulm_asthma` | Pulmonology | 19M severe wheezing, accessory muscles | Status asthmaticus, albuterol, steroids |
| `gi_bleed` | Gastroenterology | 60M hematemesis, melena, cirrhosis history | Upper GI bleed, endoscopy, PPI, variceal |
| `gi_pancreatitis` | Gastroenterology | 48F epigastric pain, lipase elevated | Pancreatitis, NPO, IV fluids, imaging |
| `neuro_seizure` | Neurology | 30F witnessed generalized seizure | Status epilepticus, benzodiazepine, EEG |
| `id_meningitis` | Infectious Disease | 20M fever, neck stiffness, photophobia | Meningitis, lumbar puncture, empiric antibiotics |
| `psych_suicidal` | Psychiatry | 35M suicidal ideation, plan, access | Suicide risk, safety assessment, hospitalization |
| `peds_fever` | Pediatrics | 3-week-old neonate, fever 38.5°C | Neonatal fever, sepsis workup, admit |
| `peds_dehydration` | Pediatrics | 2-year-old, 5 days diarrhea/vomiting | Dehydration, ORS, electrolytes |
| `nephro_hyperkalemia` | Nephrology | 70M CKD, K+ 7.2, ECG changes | Hyperkalemia, calcium gluconate, insulin/glucose, dialysis |
| `tox_acetaminophen` | Emergency Medicine | 23F intentional APAP overdose | Acetaminophen, NAC, liver, Rumack-Matthew |
| `geri_polypharmacy` | Geriatrics | 82F on 12 medications, recurrent falls | Polypharmacy, fall risk, medication reconciliation, Beers criteria |

### How to Reproduce

```bash
cd src/backend

# List all available cases
python test_clinical_cases.py --list

# Run a single case
python test_clinical_cases.py --case em_sepsis

# Run all cases in a specialty
python test_clinical_cases.py --specialty Cardiology

# Run all 22 cases
python test_clinical_cases.py

# Run all and save report to JSON
python test_clinical_cases.py --report results.json

# Quiet mode (summary only)
python test_clinical_cases.py --quiet
```

---

## 4. RAG Corpus Statistics

| Metric | Value |
|--------|-------|
| Total guidelines | 62 |
| Specialties covered | 14 |
| Guidelines stored in ChromaDB | 62 |
| Embedding model | all-MiniLM-L6-v2 (384 dimensions) |
| Embedding time (full rebuild) | ~5 s |
| ChromaDB persist directory | `./data/chroma` |
| Source file | `app/data/clinical_guidelines.json` |

### Guidelines per Specialty

| Specialty | Count |
|-----------|-------|
| Emergency Medicine | 10 |
| Cardiology | 8 |
| Endocrinology | 7 |
| Gastroenterology | 5 |
| Infectious Disease | 5 |
| Pulmonology | 4 |
| Neurology | 4 |
| Psychiatry | 4 |
| Pediatrics | 4 |
| Nephrology | 2 |
| Hematology | 2 |
| Rheumatology | 2 |
| OB/GYN | 2 |
| Preventive / Perioperative / Dermatology | 3 |

---

## 5. Test Infrastructure

| File | Lines | Purpose |
|------|-------|---------|
| `test_e2e.py` | 57 | Submit chest pain case, poll for completion, validate all 5 steps |
| `test_clinical_cases.py` | ~400 | 22 clinical cases with keyword validation, CLI flags for filtering |
| `test_rag_quality.py` | ~350 | 30 RAG retrieval queries with expected guideline IDs, relevance scoring |
| `test_poll.py` | ~30 | Utility: poll a case ID until completion |

### Dependencies for Testing

Tests use only the standard library + `httpx` (for REST calls) and the backend's own modules (for RAG tests). No additional test frameworks required beyond what's in `requirements.txt`.
