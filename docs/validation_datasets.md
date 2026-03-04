# Validation Dataset Landscape

> Comprehensive catalog of medical datasets, benchmarks, and evaluation frameworks applicable to the CDS Agent pipeline. Organized by **feasibility tier** — how much effort is required to integrate each dataset into our validation harness.
>
> Last updated: 2026-03-03

---

## Table of Contents

- [Tier 1 — Immediately Feasible](#tier-1--immediately-feasible)
- [Tier 2 — Moderate Effort](#tier-2--moderate-effort)
- [Tier 3 — Significant Effort](#tier-3--significant-effort)
- [Tier 4 — Institutional / Long-Term](#tier-4--institutional--long-term)
- [Tier 5 — Aspirational / Restricted](#tier-5--aspirational--restricted)
- [Coverage Matrix](#coverage-matrix)
- [Priority Recommendations](#priority-recommendations)

---

## Tier 1 — Immediately Feasible

*Free, publicly available, API/HuggingFace-accessible, minimal preprocessing. Could be integrated within a day using our existing harness patterns.*

### 1.1 Medical Question Answering

| Dataset | Size | Description | Pipeline Step Tested | Access |
|---------|------|-------------|---------------------|--------|
| **MedQA (USMLE)** | 12,723 | 4-option USMLE-style MCQs across all medical disciplines | Clinical Reasoning (Step 2) | ✅ Already integrated — HF `bigbio/med_qa` |
| **MedMCQA** | 194,000 | Indian AIIMS/NEET entrance exam MCQs, 21 medical subjects | Clinical Reasoning (Step 2) | ✅ Already integrated — HF `openlifescienceai/medmcqa` |
| **PubMedQA** | 1,000 labeled | Yes/no/maybe questions from PubMed abstracts with context | Clinical Reasoning (Step 2) | ✅ Already integrated — HF `qiaojin/PubMedQA` |
| **MedQA-5** | 12,723 | Same as MedQA but 5-option format (harder) | Clinical Reasoning (Step 2) | HF `bigbio/med_qa` (5-option split) |
| **MMLU-Medical** | ~3,000 | Medical subset of Massive Multitask Language Understanding: clinical knowledge, medical genetics, anatomy, professional medicine, college biology, college medicine | Clinical Reasoning (Step 2) | HF `cais/mmlu` — filter by subject |
| **HeadQA** | 6,792 | Spanish healthcare professional exam questions (available in English translation), covers medicine, nursing, pharmacology, psychology, biology, chemistry | Clinical Reasoning (Step 2) | HF `head_qa` |
| **JAMA Clinical Challenge** | ~500 | Curated clinical vignettes from JAMA Network with expert diagnoses | End-to-end pipeline | Scrape from JAMA open-access archive |
| **Medbullets** | ~3,500 | Step 2/3 USMLE-style questions with explanations | Clinical Reasoning (Step 2) | Community datasets on HF |

### 1.2 Biomedical Natural Language Inference

| Dataset | Size | Description | Pipeline Step Tested | Access |
|---------|------|-------------|---------------------|--------|
| **MedNLI** | 14,049 | Clinical sentence pairs (entailment/contradiction/neutral), derived from MIMIC-III clinical notes | Conflict Detection (Step 5) | HF `bigbio/mednli` |
| **SciTail** | 27,026 | Science entailment from exam questions and web text | Guideline Reasoning (Step 4-5) | HF `scitail` |
| **BioNLI** | 16,994 | NLI derived from biomedical literature contrasting claims | Conflict Detection (Step 5) | HF `biomistral/BioNLI` |
| **ManConCorpus** | 10,000 | Contradictions in medical literature (same-topic papers making opposing claims) | Conflict Detection (Step 5) | GitHub release |

### 1.3 Clinical Text & NER

| Dataset | Size | Description | Pipeline Step Tested | Access |
|---------|------|-------------|---------------------|--------|
| **MTSamples** | 4,999 | Transcribed medical reports across 40 specialties | Patient Parsing (Step 1) | ✅ Already integrated — mtsamples.com |
| **BC5CDR** | 1,500 articles | Chemical-disease relation extraction from PubMed abstracts | Drug Interactions (Step 3) | HF `bigbio/bc5cdr` |
| **NCBI Disease Corpus** | 793 abstracts | Disease name recognition and normalization | Patient Parsing (Step 1) | HF `ncbi_disease` |
| **ADE Corpus** | 20,967 sentences | Adverse drug event extraction from medical case reports | Drug Interactions (Step 3) | HF `ade_corpus_v2` |
| **CADEC** | 1,250 posts | Consumer adverse drug event extraction from AskAPatient.com | Drug Interactions (Step 3) | HF `cadec` |
| **n2c2 2018 ADE** | 505 notes | Adverse drug event detection in clinical notes (shared task) | Drug Interactions (Step 3) | Public after registration at n2c2.dbmi.hms.harvard.edu |
| **DDI Corpus** | 792 abstracts + 233 DrugBank entries | Drug-drug interaction extraction (mechanism, effect, advice, int) | Drug Interactions (Step 3) | HF `bigbio/ddi_corpus` |

### 1.4 Biomedical Relation Extraction

| Dataset | Size | Description | Pipeline Step Tested | Access |
|---------|------|-------------|---------------------|--------|
| **ChemProt** | 2,432 abstracts | Chemical-protein interaction extraction (13 classes) | Drug Interactions (Step 3) | HF `bigbio/chemprot` |
| **GAD** | 5,000+ | Gene-disease association sentences | Clinical Reasoning (Step 2) | HF `gad` |
| **EU-ADR** | 300 abstracts | Drug-disease, drug-target, target-disease relations | Drug Interactions (Step 3) | HF `bigbio/euadr` |

### 1.5 Evidence & Literature

| Dataset | Size | Description | Pipeline Step Tested | Access |
|---------|------|-------------|---------------------|--------|
| **PMC Case Reports** | Millions | Full-text case reports from PubMed Central | End-to-end pipeline | ✅ Already integrated — NCBI E-utilities API |
| **EBM-NLP** | 5,000 abstracts | Evidence-based medicine: PICO extraction from RCTs | Guideline Retrieval (Step 4) | HF `ebm_nlp` |
| **PICO Extraction** | 10,000+ | Population/Intervention/Comparator/Outcome extraction | Guideline Retrieval (Step 4) | Various HF datasets |
| **SciFact** | 1,409 claims | Scientific claim verification against abstracts | Conflict Detection (Step 5) | HF `allenai/scifact` |
| **HealthVer** | 14,330 claims | Health-related claim verification from COVID-19 misinformation | Conflict Detection (Step 5) | HF `healthver` |
| **PubHealth** | 11,832 claims | Public health claim fact-checking with explanations and evidence | All reasoning steps | HF `health_fact` |

---

## Tier 2 — Moderate Effort

*Free or open-access, but requires non-trivial data download, format conversion, API key setup, or moderate engineering work. Integration within 1-3 days.*

### 2.1 Clinical Case Repositories

| Dataset | Size | Description | Pipeline Step Tested | Access | Effort Notes |
|---------|------|-------------|---------------------|--------|--------------|
| **PMC Patients** | 167,000 patient summaries | Patient summaries extracted from PubMed Central case reports, linked to relevant articles | End-to-end pipeline | HF `zhengyun21/PMC-Patients` | Need to parse semi-structured fields into free text |
| **MIMIC-III Clinical Notes (free subset)** | Demo: 100 patients | Deidentified ICU clinical notes (discharge summaries, radiology, nursing) | Patient Parsing (Step 1) | PhysioNet (free after credentialing course) | CITI training required (~2hr), then download |
| **WikiMed / WikiDoc** | 10,000+ articles | Medical knowledge from Wikipedia, structured by disease | Guideline Retrieval (Step 4) | Free download | Need to chunk and index for RAG testing |
| **MedDialog** | 3.66M turns | Doctor-patient conversations (English + Chinese) from medical forums | Patient Parsing (Step 1) | HF `medical_dialog` | Conversational format needs summarization |
| **HealthSearchQA** | 3,375 | Consumer health questions from real search queries, curated by physicians | End-to-end pipeline | Google Research GitHub | Convert from consumer language to clinical format |
| **LiveQA-Medical** | 634 | Consumer health questions from NLM, expert-judged answers | End-to-end pipeline | TREC LiveQA shared task release | Moderate: consumer language, need reference mapping |
| **TREC Clinical Trials** | 50-75 topics/year | Patient case descriptions matched to clinical trials | Patient Parsing (Step 1), Clinical Reasoning (Step 2) | TREC 2021-2023 data | Need to build matching evaluation |
| **emrQA** | 455,000+ QA pairs | Questions about electronic medical records (derived from i2b2 data) | Patient Parsing (Step 1), Clinical Reasoning (Step 2) | GitHub release | Large dataset, need sampling strategy |

### 2.2 Diagnosis & Differential Diagnosis

| Dataset | Size | Description | Pipeline Step Tested | Access | Effort Notes |
|---------|------|-------------|---------------------|--------|--------------|
| **DDXPlus** | 1.3M patients | Synthetic patient data with differential diagnoses (49 diseases), symptoms, antecedents, lab results | Clinical Reasoning (Step 2), End-to-end | HF `alea-institute/ddxplus` | Structured → need to build vignettes from tabular data |
| **DxBench** | 570 cases | Diagnosis benchmark from NEJM CPC, JAMA, BMJ case reports | Clinical Reasoning (Step 2) | Paper + GitHub | Need to parse from scraped data |
| **TREC CDS** | 180 topics | Clinical decision support topics with graded relevance judgments | End-to-end pipeline | TREC 2014-2016 | Moderate: IR-focused, need adaptation |
| **Rare Disease Diagnosis** | ~5,000 | Cases from NORD, Orphanet with rare disease diagnoses | Clinical Reasoning (Step 2) | Orphanet API + GARD | Need to combine multiple sources |
| **PUMPS (Pediatric)** | 500+ | Pediatric diagnosis cases from teaching hospitals | Clinical Reasoning (Step 2) | Published case series | Manual collection from literature |
| **Isabel DDx Cases** | ~10,000 | Clinical vignettes with differential diagnosis lists | Clinical Reasoning (Step 2) | Isabel Healthcare (may require license) | API available, need to evaluate terms |

### 2.3 Drug & Pharmacology

| Dataset | Size | Description | Pipeline Step Tested | Access | Effort Notes |
|---------|------|-------------|---------------------|--------|--------------|
| **DrugBank** | 15,000+ drugs | Comprehensive drug database with interactions, targets, pathways, pharmacology | Drug Interactions (Step 3) | DrugBank Open Data (CC-BY-NC) | XML parsing, large file |
| **SIDER** | 1,430 drugs | Side effects from drug labels (MedDRA terms), 5,868 side effects | Drug Interactions (Step 3) | Download from sideeffects.embl.de | TSV format, MedDRA mapping needed |
| **TWOSIDES** | 3.3B drug-drug-event pairs | Drug-drug interaction adverse events mined from FAERS | Drug Interactions (Step 3) | Download from tatonettilab | Very large, need sampling |
| **OFFSIDES** | 174,000+ | Off-label drug side effects not on drug labels | Drug Interactions (Step 3) | Download from tatonettilab | TSV format, moderate processing |
| **FDA FAERS** | Millions of reports | FDA Adverse Event Reporting System quarterly extracts | Drug Interactions (Step 3) | openFDA API (already partially used) | Need to build structured test cases from reports |
| **PharmGKB** | 1,000+ | Pharmacogenomic drug-gene interactions and clinical annotations | Drug Interactions (Step 3) | Download after free registration | Need to build interaction scenario tests |
| **RxNorm** | Full terminology | Drug normalization and interaction checking | Drug Interactions (Step 3) | NLM UMLS (already partially used) | Extend existing RxNorm integration |
| **NDF-RT** | Full ontology | National Drug File - drug classes, mechanisms, physiological effects, contraindications | Drug Interactions (Step 3) | NLM download | RDF/OWL format parsing |

### 2.4 Clinical Guidelines & Recommendations

| Dataset | Size | Description | Pipeline Step Tested | Access | Effort Notes |
|---------|------|-------------|---------------------|--------|--------------|
| **NGC (National Guideline Clearinghouse) Archive** | 3,700+ summaries | Archived clinical practice guideline summaries (pre-2018) | Guideline Retrieval (Step 4) | AHRQ archive, Internet Archive | Need to parse HTML, may have copyright issues |
| **NICE Guidelines** | 330+ | UK National Institute for Health and Care Excellence guidelines | Guideline Retrieval (Step 4) | NICE website, structured API | Web scraping + parsing, well-structured |
| **WHO Guidelines** | 100+ | World Health Organization clinical recommendations | Guideline Retrieval (Step 4) | WHO website | PDF extraction needed |
| **UpToDate Evidence Summaries** | 12,000+ topics | Expert-curated clinical recommendations | Guideline Retrieval (Step 4) | Requires institutional subscription | $$$ — not freely available |
| **ClinicalKey (Elsevier)** | Thousands | Evidence-based clinical information | Guideline Retrieval (Step 4) | Requires subscription | $$$ — institutional access |
| **OpenCPG** | Varies | Open-access clinical practice guidelines collection | Guideline Retrieval (Step 4) | Various medical society websites | Need to aggregate from multiple sources |
| **GIN (Guidelines International Network)** | 6,500+ | International guideline library | Guideline Retrieval (Step 4) | gin.net (registration required) | International scope, multi-language |

### 2.5 Radiology & Imaging Reports

| Dataset | Size | Description | Pipeline Step Tested | Access | Effort Notes |
|---------|------|-------------|---------------------|--------|--------------|
| **MIMIC-CXR Reports** | 227,835 reports | Free-text radiology reports with structured labels | Patient Parsing (Step 1) | PhysioNet (credentialing) | Text only — no images needed for our pipeline |
| **OpenI (Indiana)** | 7,470 reports | Chest X-ray reports with findings/impression | Patient Parsing (Step 1) | NLM OpenI API | API-accessible, moderate parsing |
| **RadQA** | 4,926 QA pairs | Question-answer pairs from radiology reports | Clinical Reasoning (Step 2) | HF dataset | Radiology-specific evaluation |
| **CT-RATE** | 50,000+ reports | Radiology report dataset for CT scans | Patient Parsing (Step 1) | HF `ibrahimhamamci/CT-RATE` | Large, need text extraction |

### 2.6 Clinical Summarization & Generation

| Dataset | Size | Description | Pipeline Step Tested | Access | Effort Notes |
|---------|------|-------------|---------------------|--------|--------------|
| **MeQSum** | 1,000 | Consumer health question summarization | Synthesis (Step 6) | HF dataset | Small, good for summarization quality |
| **MEDIQA** | 2,479 | Medical question answering and NLI | Clinical Reasoning (Step 2), Synthesis (Step 6) | Shared task data | Multiple sub-tasks available |
| **BioASQ** | 5,000+ questions | Biomedical question answering challenge (yearly competition) | Clinical Reasoning (Step 2) | bioasq.org (registration) | Competition format, well-established |
| **CORD-19** | 1M+ papers | COVID-19 research literature | Guideline Retrieval (Step 4) | Semantic Scholar API | Good for RAG testing on large corpus |

---

## Tier 3 — Significant Effort

*Requires data use agreements, ethics training, substantial preprocessing, or building new evaluation paradigms. Integration requires 1-2 weeks.*

### 3.1 Electronic Health Record Datasets

| Dataset | Size | Description | Pipeline Step Tested | Access | Effort Notes |
|---------|------|-------------|---------------------|--------|--------------|
| **MIMIC-III** | 58,976 ICU stays | Deidentified ICU data: notes, labs, vitals, medications, diagnoses, procedures | Full pipeline | PhysioNet — CITI training + DUA | Gold standard for clinical NLP, extensive setup |
| **MIMIC-IV** | 524,000 admissions | Updated MIMIC with expanded data through 2022 | Full pipeline | PhysioNet — CITI training + DUA | Newer, more comprehensive than MIMIC-III |
| **MIMIC-IV-Note** | 331,794 notes | Discharge summaries and radiology reports from MIMIC-IV | Patient Parsing (Step 1), Synthesis (Step 6) | PhysioNet | Subset of MIMIC-IV, text-focused |
| **eICU** | 200,859 stays | Multi-center ICU data from Philips eICU program | Full pipeline | PhysioNet — CITI training + DUA | Multi-site, greater diversity than MIMIC |
| **MIMIC-IV-ED** | 448,972 ED stays | Emergency department data with triage notes, vitals, diagnoses | Patient Parsing (Step 1), Clinical Reasoning (Step 2) | PhysioNet | ED-focused, acute care scenarios |
| **AmsterdamUMCdb** | 23,106 admissions | Dutch ICU dataset with labs, vitals, medications | Full pipeline | Access request to Amsterdam UMC | International ICU data, European context |

### 3.2 i2b2/n2c2 Shared Tasks (Historical Gold Standards)

| Dataset | Year | Size | Description | Pipeline Step Tested | Effort Notes |
|---------|------|------|-------------|---------------------|--------------|
| **i2b2 2006 — Deidentification** | 2006 | 889 notes | PHI detection and removal | Patient Parsing (Step 1) | DUA required via n2c2.dbmi.hms.harvard.edu |
| **i2b2 2008 — Obesity** | 2008 | 1,237 notes | Obesity and 15 comorbidity detection | Patient Parsing (Step 1), Clinical Reasoning (Step 2) | DUA required |
| **i2b2 2009 — Medication** | 2009 | 268 notes | Medication name, dosage, frequency, mode, reason extraction | Patient Parsing (Step 1), Drug Interactions (Step 3) | DUA required |
| **i2b2 2010 — Relations** | 2010 | 871 notes | Problem-treatment-test concept extraction and relation classification | Patient Parsing (Step 1) | DUA required |
| **i2b2 2011 — Coreference** | 2011 | 853 notes | Clinical coreference resolution for concepts, people, pronouns | Patient Parsing (Step 1) | DUA required |
| **i2b2 2012 — Temporal** | 2012 | 310 notes | Temporal relation extraction between clinical events | Clinical Reasoning (Step 2) | DUA required |
| **n2c2 2018 — ADE** | 2018 | 505 notes | Adverse drug event detection and extraction | Drug Interactions (Step 3) | DUA required |
| **n2c2 2019 — Clinical Trial Matching** | 2019 | 100 patients | Patient matching to clinical trial eligibility criteria | Full pipeline | DUA required |
| **n2c2 2022 — SDoH** | 2022 | 4,480 notes | Social determinants of health extraction | Patient Parsing (Step 1) | DUA required |

### 3.3 ICD/CPT Coding & Classification

| Dataset | Size | Description | Pipeline Step Tested | Access | Effort Notes |
|---------|------|-------------|---------------------|--------|--------------|
| **MIMIC-III ICD Coding** | 58,976 stays | ICD-9 diagnosis and procedure codes with discharge summaries | Clinical Reasoning (Step 2) | PhysioNet (part of MIMIC-III) | Build ICD prediction evaluation from existing data |
| **PLM-ICD** | 47,723 notes | Automated ICD coding benchmark | Clinical Reasoning (Step 2) | Derived from MIMIC, requires MIMIC access | |
| **Multi-label ICD from MIMIC-IV** | 524,000+ | ICD-10 coding with clinical notes | Clinical Reasoning (Step 2) | Requires MIMIC-IV access | Modern ICD-10 codes |
| **CCS (Clinical Classifications Software)** | Full | HCUP Clinical Classifications Software mapping ICD codes to categories | Clinical Reasoning (Step 2) | AHRQ — free download | Useful for grouping diagnoses |

### 3.4 Clinical Trial & Evidence synthesis

| Dataset | Size | Description | Pipeline Step Tested | Access | Effort Notes |
|---------|------|-------------|---------------------|--------|--------------|
| **ClinicalTrials.gov** | 450,000+ trials | Full clinical trial registry with eligibility criteria, interventions, outcomes | Clinical Reasoning (Step 2), Guideline Retrieval (Step 4) | Public API | Massive, need to build patient-trial matching eval |
| **Cochrane Systematic Reviews** | 8,000+ | Gold-standard systematic reviews and meta-analyses | Guideline Retrieval (Step 4), Conflict Detection (Step 5) | API access (partial) | Need to extract recommendation strength |
| **CDSR (Cochrane Database)** | 12,000+ | Cochrane Database of Systematic Reviews | Guideline Retrieval (Step 4) | Subscription for full text | Partial open access |

### 3.5 Patient Safety & Adverse Events

| Dataset | Size | Description | Pipeline Step Tested | Access | Effort Notes |
|---------|------|-------------|---------------------|--------|--------------|
| **AHRQ WebM&M** | 500+ cases | Morbidity & Mortality case reports with expert analysis | Conflict Detection (Step 5) | psnet.ahrq.gov (free) | Need to scrape and structure cases |
| **AHRQ PSNet** | 1,000+ articles | Patient Safety Network case studies and alerts | Conflict Detection (Step 5) | psnet.ahrq.gov | Valuable for conflict detection testing |
| **ISMP Medication Error Reports** | 1,000+ | Institute for Safe Medication Practices error reports | Drug Interactions (Step 3), Conflict Detection (Step 5) | ismp.org (partial public access) | Medication safety focus |
| **FDA MedWatch** | Millions | Post-market safety reports for drugs, devices, biologics | Drug Interactions (Step 3) | openFDA API | Need to build structured test cases |
| **PSI (Patient Safety Indicators)** | 26 indicators | AHRQ Patient Safety Indicators definition and validation | Conflict Detection (Step 5) | AHRQ free download | Quality measurement metrics |

### 3.6 Specialty-Specific Datasets

| Dataset | Size | Description | Pipeline Step Tested | Access | Effort Notes |
|---------|------|-------------|---------------------|--------|--------------|
| **CheXpert** | 224,316 reports | Chest X-ray report labels for 14 observations | Clinical Reasoning (Step 2) | Stanford ML Group (DUA) | Report text only, no images needed |
| **MIMIC-III Cardiac** | Subset | Cardiac-specific ICU data (ECGs, echo reports, cardiac procedures) | Full pipeline (cardiology) | MIMIC-III subset | Specialty filtering needed |
| **psych-NLP** | Various | Psychiatric clinical notes NLP benchmarks (various shared tasks) | Patient Parsing (Step 1) | Various (CLPsych shared tasks) | Mental health domain-specific |
| **Dermatology DDx** | ~1,000 | Dermatology differential diagnosis cases | Clinical Reasoning (Step 2) | DermNet NZ, VisualDx (partial) | Visual + text cases, use text descriptions |
| **OphthoQA** | ~500 | Ophthalmology clinical questions | Clinical Reasoning (Step 2) | Literature compilation | Small, specialty-specific |
| **PathQA** | 4,998 | Pathology question-answer pairs | Clinical Reasoning (Step 2) | HF dataset | Pathology-specific evaluation |

### 3.7 Multilingual / International

| Dataset | Size | Description | Pipeline Step Tested | Access | Effort Notes |
|---------|------|-------------|---------------------|--------|--------------|
| **XMedQA** | Multiple languages | Cross-lingual medical QA (Chinese, French, German, etc.) | Clinical Reasoning (Step 2) | HF datasets | Multilingual evaluation |
| **CLEF eHealth** | Varies | European health NLP shared tasks (multi-year, multilingual) | Various | CLEF conference data | Multi-year, multi-task |
| **FrenchMedMCQA** | 3,105 | French medical exam questions | Clinical Reasoning (Step 2) | HF dataset | French language |
| **HEAD-QA (Spanish)** | 6,792 | Spanish healthcare exam questions | Clinical Reasoning (Step 2) | HF dataset | Spanish language |
| **ExamedQA** | 15,000+ | Medical exam QA in Portuguese (Brazil) | Clinical Reasoning (Step 2) | HF dataset | Portuguese language |
| **RuMedBench** | Multiple tasks | Russian medical NLP benchmark suite | Various | HF datasets | Russian language |
| **CBLUE** | Multiple tasks | Chinese Biomedical Language Understanding Evaluation | Various | HF datasets | Chinese language |
| **KorMedMCQA** | 6,000+ | Korean medical licensing exam questions | Clinical Reasoning (Step 2) | HF dataset | Korean language |

---

## Tier 4 — Institutional / Long-Term

*Requires formal institutional partnerships, IRB approval, or sustained multi-month engineering effort. Strategic investments for production-grade validation.*

### 4.1 Large-Scale EHR Partnerships

| Dataset / Source | Size | Description | Pipeline Step Tested | Access | Effort Notes |
|---------|------|-------------|---------------------|--------|--------------|
| **All of Us Research Program** | 500,000+ participants | NIH precision medicine cohort with EHR, genomics, surveys, wearables | Full pipeline | Researcher workbench (NIH application) | Federal program, 6-12 month onboarding |
| **UK Biobank** | 503,325 participants | Deep phenotyping + genomics with linked GP and hospital records | Full pipeline (epidemiological) | Application to UK Biobank (6-12 months) | UK-focused, rich longitudinal data |
| **PCORnet (Patient-Centered Outcomes Research)** | 60M+ patients | Multi-site US clinical research network, standardized EHR data | Full pipeline | Requires PCORnet partnership | Network of health systems |
| **OHDSI / OMOP CDM** | Billions of records | Observational Health Data Sciences and Informatics — standardized multi-site data | Full pipeline | Requires OHDSI network participation | Global, standardized CDM format |
| **TriNetX** | 300M+ patients | Global health research network with federated EHR data | Full pipeline | Commercial partnership | Federated queries, no raw data access |
| **Flatiron Health** | 3M+ cancer patients | Real-world oncology EHR data with treatment outcomes | Pipeline (oncology) | Commercial partnership | Oncology-specific |
| **Optum Labs** | 200M+ lives | Claims + EHR linked data | Full pipeline | Commercial partnership | Broad coverage, longitudinal |

### 4.2 Clinical Decision Support Validation

| Dataset / Source | Description | Pipeline Step Tested | Effort Notes |
|---------|-------------|---------------------|--------------|
| **AHRQ CDS Connect** | Repository of validated CDS artifacts (rules, order sets, documentation templates) | Conflict Detection (Step 5), Full pipeline | Would provide gold-standard CDS comparison |
| **OpenCDS** | Open-source clinical decision support engine and knowledge base | Full pipeline | Integration testing against established CDS |
| **Epic CDS Hooks Reference** | Epic's implementation of CDS Hooks with test patients | CDS Hooks output, Full pipeline | Requires Epic sandbox access |
| **Cerner (Oracle Health) Sandbox** | Oracle Health FHIR sandbox with sample patient data | FHIR input, CDS Hooks | Developer account needed |
| **SMART on FHIR App Gallery** | Repository of validated SMART apps with test data | FHIR input, CDS Hooks | Reference implementations for comparison |
| **Logica Health Sandbox** | Open-source FHIR sandbox (formerly HSPC) | FHIR input, CDS Hooks | Free, good for testing |

### 4.3 Prospective Clinical Validation

| Approach | Description | Pipeline Step Tested | Effort Notes |
|---------|-------------|---------------------|--------------|
| **Retrospective chart review** | Run pipeline on historical cases, compare against actual outcomes | Full pipeline | Requires IRB + data access at partner institution |
| **Clinician concordance study** | Compare CDS Agent recommendations to independent clinician assessments | Synthesis (Step 6) | Requires clinical collaborators (5-10 physicians) |
| **A/B testing in clinical workflow** | Controlled trial of CDS Agent vs. standard care decision support | Full pipeline | Requires integration into EHR, IRB approval |
| **Simulated patient encounters** | Standardized patients with known diagnoses evaluated by CDS Agent | Full pipeline | Requires medical education partnership |
| **Case conference comparison** | Compare against published tumor board / M&M conference decisions | Clinical Reasoning (Step 2) | Manual case collection |
| **Expert panel Delphi method** | Structured multi-round expert panel evaluation of CDS recommendations | All steps | Requires 5-10 domain experts, 3-6 months |

### 4.4 Quality Measurement Frameworks

| Framework | Description | Relevance | Effort Notes |
|---------|-------------|-----------|--------------|
| **HEDIS Measures** | Healthcare Effectiveness Data and Information Set (NCQA) | Guideline adherence validation | Requires claims/EHR data to measure |
| **CMS Quality Measures (eCQMs)** | Electronic Clinical Quality Measures | Conflict Detection, Guideline adherence | CQL-based, requires FHIR integration |
| **Joint Commission Standards** | Hospital accreditation quality standards | Safety and process quality | Framework for safety evaluation |
| **Leapfrog Group** | Hospital safety grades and metrics | Patient safety assessment | Benchmarking framework |

---

## Tier 5 — Aspirational / Restricted

*Highly restricted datasets, proprietary systems, or future research directions. Would represent gold-standard validation but require significant resources, time, or partnerships.*

### 5.1 Proprietary Clinical Knowledge Bases

| Source | Description | Relevance | Access |
|--------|-------------|-----------|--------|
| **UpToDate** | Definitive point-of-care clinical resource used by 1.9M clinicians | Gold-standard guideline comparison | Wolters Kluwer subscription (~$500/yr for individual) |
| **DynaMed** | Evidence-based clinical decision support | Guideline comparison | EBSCO subscription |
| **Clinical Pharmacology / Lexicomp** | Comprehensive drug interaction databases | Drug interaction validation | Subscription required |
| **Micromedex** | IBM drug information and clinical decision support | Drug interaction validation | Merative / IBM subscription |
| **First Databank (FDB)** | Drug knowledge base used in most EHR systems | Drug interaction ground truth | Commercial license |
| **Medi-Span** | Wolters Kluwer drug data, widely used in pharmacies | Drug interaction validation | Commercial license |
| **VisualDx** | Visual diagnosis decision support with clinical images | Diagnostic accuracy comparison | Subscription required |

### 5.2 Gold-Standard Diagnostic Systems (Comparison)

| System | Description | Comparison Value | Access |
|--------|-------------|-----------------|--------|
| **Isabel Healthcare** | AI-powered differential diagnosis tool (20+ years) | Head-to-head diagnostic accuracy comparison | API license |
| **DXplain (MGH)** | Clinical decision support system from Massachusetts General | Benchmark diagnostic reasoning | Institutional license |
| **GIDEON** | Global Infectious Disease Epidemiology Online Network | Infectious disease diagnosis validation | Subscription |
| **RADLogics / Aidoc** | AI-powered radiology triage and diagnosis | Radiology report interpretation | Commercial |
| **Buoy Health** | AI-powered symptom checker with clinical validation data | Consumer health triage comparison | Partnership |
| **Ada Health** | CE-marked AI symptom assessment | Validated symptom assessment comparison | Partnership |

### 5.3 Restricted Government / Military Data

| Source | Description | Relevance | Access |
|--------|-------------|-----------|--------|
| **VA VINCI** | Veterans Affairs clinical data research platform (20M+ veterans) | Massive longitudinal EHR for full pipeline validation | VA researcher status, IRB, DUA |
| **DoD MHS GENESIS** | Military Health System EHR data | Large-scale health system validation | Military/DoD researcher access |
| **CMS Medicare/Medicaid Claims** | Claims data for 100M+ beneficiaries | Population-level outcome validation | ResDAC application (CMS) |
| **CDC NHANES** | National Health and Nutrition Examination Survey | Population health benchmarks | Mostly public, restricted-use versions |

### 5.4 Synthetic / Simulated Data Generation

| Approach | Description | Use Case | Effort |
|---------|-------------|----------|--------|
| **Synthea** | Open-source synthetic patient generator with realistic FHIR records | FHIR adapter testing, pipeline stress testing | Free, open source — excellent for FHIR |
| **MDClone** | Synthetic data generation from real EHR data (preserving distributions) | Realistic patient cohorts without PHI | Commercial platform |
| **Gretel.ai Health** | Privacy-preserving synthetic health data | Training data augmentation | Freemium |
| **MITRE Health Data Fabric** | Synthetic health data for testing and research | Interoperability testing | Open-source tools |
| **CMS Synthetic Data** | CMS-generated synthetic Medicare claims | Claims-based testing | Free download from CMS |

### 5.5 Emerging Benchmarks (2025-2026)

| Benchmark | Description | Status | Relevance |
|-----------|-------------|--------|-----------|
| **MedBench** | Comprehensive medical AI benchmark suite (Google/DeepMind) | Active development | Multi-modal medical evaluation |
| **Open Medical LLM Leaderboard** | HuggingFace leaderboard for medical LLMs | Active | Direct model comparison |
| **CLUE (Clinical Language Understanding Evaluation)** | Proposed standard for clinical NLP evaluation | Proposed | Would unify evaluation |
| **FDA AI/ML SaMD Guidelines** | FDA framework for Software as Medical Device | Regulatory | Regulatory approval framework |
| **EU AI Act — High-Risk AI** | European regulatory framework for clinical AI | Regulatory | Compliance requirements |
| **WHO Ethics & Governance of AI for Health** | WHO guidance on responsible health AI | Ethical validation | Ethics framework |

---

## Coverage Matrix

How each pipeline step maps to available validation data:

| Pipeline Step | Tier 1 | Tier 2 | Tier 3 | Tier 4+ | Current Coverage |
|---------------|--------|--------|--------|---------|-----------------|
| **Step 1: Patient Parsing** | MTSamples ✅, NCBI Disease | MedDialog, MIMIC-CXR reports, PMC Patients | MIMIC-III/IV, i2b2 2009/2010, n2c2 | All of Us, VA VINCI | ✅ MTSamples |
| **Step 2: Clinical Reasoning** | MedQA ✅, MedMCQA ✅, PubMedQA ✅, MMLU-Medical, HeadQA | DDXPlus, BioASQ, DxBench | MIMIC ICD coding, CheXpert | Isabel, DXplain comparison | ✅ MedQA, MedMCQA, PubMedQA, PMC |
| **Step 3: Drug Interactions** | ADE Corpus, DDI Corpus, BC5CDR, CADEC | DrugBank, SIDER, TWOSIDES, PharmGKB | n2c2 2018 ADE, FDA FAERS | FDB, Lexicomp, Micromedex | ⚠️ Indirect via pipeline only |
| **Step 4: Guideline Retrieval** | EBM-NLP, SciFact, PubHealth | NICE Guidelines, ClinicalTrials.gov, CORD-19 | Cochrane, NGC Archive | UpToDate, DynaMed | ⚠️ RAG quality via full pipeline |
| **Step 5: Conflict Detection** | MedNLI, BioNLI, ManConCorpus, HealthVer | AHRQ WebM&M | AHRQ PSNet, ISMP | CDS Connect, eCQMs | ⚠️ Adversarial suite only |
| **Step 6: Synthesis** | All above (measured via report quality) | MeQSum, MEDIQA | Clinician concordance study | Expert panel review | ⚠️ Adversarial + regression |
| **FHIR Input** | — | Synthea, Logica Sandbox | MIMIC-IV FHIR, Cerner Sandbox | Epic Sandbox, SMART Gallery | 🆕 FHIR adapter created |
| **CDS Hooks Output** | — | CDS Hooks spec compliance testing | Epic/Cerner sandbox hooks | Real EHR integration | 🆕 CDS Hooks formatter created |

---

## Priority Recommendations

### Immediate (This Week)
1. **MMLU-Medical** — 6 medical subtasks, directly accessible on HuggingFace, same MCQ harness pattern as MedQA/MedMCQA
2. **DDXPlus** — 1.3M synthetic patients with differential diagnoses, directly tests diagnostic ranking quality
3. **MedNLI** — Clinical sentence-pair NLI, directly tests conflict detection reasoning
4. **Synthea** — Generate synthetic FHIR patients to stress-test the new FHIR adapter
5. **SciFact** — Claim verification against abstracts, tests evidence-based reasoning

### Near-Term (This Month)
6. **ADE Corpus + DDI Corpus** — Direct drug interaction extraction benchmarks
7. **NICE Guidelines** — Well-structured UK guidelines, expand RAG knowledge base
8. **PMC Patients** — 167K patient summaries, massive scale testing
9. **BioASQ** — Annual biomedical QA competition benchmark
10. **DrugBank Open Data** — Comprehensive drug interaction ground truth

### Medium-Term (This Quarter)
11. **MIMIC-III/IV** — Gold-standard clinical data (CITI training: 2 hours, approval: 1-2 weeks)
12. **i2b2/n2c2 shared tasks** — Historical gold standard annotations
13. **ClinicalTrials.gov** — Patient-trial matching evaluation
14. **AHRQ WebM&M** — Patient safety case review
15. **Head-to-head with Isabel/DXplain** — Comparative diagnostic accuracy

### Long-Term (This Year)
16. **Clinician concordance study** — 5-10 physicians reviewing CDS Agent outputs
17. **SMART on FHIR certification** — Full FHIR/CDS Hooks compliance testing
18. **FDA SaMD pre-submission** — Regulatory pathway exploration
19. **Retrospective chart review** — Partner institution validation
20. **Prospective pilot study** — Controlled clinical workflow integration

---

## Evaluation Metrics by Dataset Type

| Data Type | Primary Metrics | Secondary Metrics |
|-----------|----------------|-------------------|
| **MCQ Benchmarks** | Top-1 accuracy, Top-3 accuracy | Per-subject accuracy, difficulty stratification |
| **Differential Diagnosis** | Diagnosis-in-differential rate, rank of correct Dx | Mean reciprocal rank, NDCG |
| **Drug Interactions** | Precision, Recall, F1 for interaction detection | Severity accuracy, mechanism identification |
| **Clinical NER** | Entity-level F1, exact match, partial match | Per-entity-type performance |
| **NLI / Conflict Detection** | Accuracy, macro-F1 across 3 classes | Contradiction recall (most important class) |
| **Guideline Adherence** | Recommendation coverage rate | Evidence level accuracy |
| **Report Quality** | ROUGE, BERTScore, clinical accuracy | Completeness, safety flag recall |
| **FHIR Compliance** | Resource parse rate, field extraction F1 | Edge case handling, extension support |
| **Adversarial Robustness** | Pass rate across categories | Graceful degradation, uncertainty flagging |

---

## Notes

- **PHI/HIPAA**: Any dataset derived from real patients (MIMIC, i2b2, VA, etc.) requires special handling — never log or persist patient data outside the credentialed environment
- **Licensing**: Always verify dataset license compatibility (CC-BY, CC-BY-NC, DUA, etc.) before incorporating into automated pipelines
- **Bias**: Medical datasets skew toward US/English clinical practice; international datasets (CBLUE, FrenchMedMCQA, etc.) help assess cross-cultural applicability
- **Temporal validity**: Clinical guidelines change; validation against static datasets may not reflect current best practice
- **Synthetic vs. Real**: Synthetic data (Synthea, DDXPlus) is valuable for stress testing but doesn't capture the messiness of real clinical documentation
