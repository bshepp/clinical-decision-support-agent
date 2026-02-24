# CDS Agent — Demo Video Script

> **Target length:** 3 minutes (max)  
> **Format:** Screen recording with voiceover  
> **Tool suggestion:** OBS Studio, Loom, or similar  

---

## PRE-RECORDING CHECKLIST

- [ ] Ensure HF Dedicated Endpoint is running (check `https://bshepp-cds-agent.hf.space/api/health/config`)
- [ ] Open browser to `https://demo.briansheppard.com` (or `https://bshepp-cds-agent.hf.space`)
- [ ] Close unnecessary tabs/notifications
- [ ] Submit one case end-to-end before recording to confirm model is warm (watch for warm-up screen)
- [ ] Browser zoom ~110-125% for readability on video
- [ ] **Local fallback** (if Space is down): `cd src/backend && uvicorn app.main:app --host 0.0.0.0 --port 8002` + `cd src/frontend && npm run dev`, then open `http://localhost:3000`

---

## SCRIPT

### OPENING — The Problem (0:00 – 0:30)

**[SCREEN: Title slide or the app landing page]**

> "Clinical decision-making is one of the most cognitively demanding tasks in medicine. For every patient, a clinician must simultaneously parse the history, generate a differential, recall drug interactions, remember guidelines, and synthesize a care plan — all under time pressure.
>
> Diagnostic errors affect 12 million Americans annually. Many aren't from lack of knowledge — they're from the difficulty of integrating information from multiple sources at once.
>
> CDS Agent solves this with an agentic pipeline powered by MedGemma."

---

### LIVE DEMO — The Pipeline in Action (0:30 – 2:00)

**[SCREEN: App interface — PatientInput component visible]**

> "Let me show you how it works. I'll load a built-in sample case — a 55-year-old male presenting to the ED with acute substernal chest pain radiating to his left arm and jaw, with diaphoresis and nausea. He has hypertension, type 2 diabetes, and hyperlipidemia, and he's on metformin, lisinopril, atorvastatin, and aspirin."

**[ACTION: Click the "Chest Pain (55M)" sample case button, then click "Analyze Patient Case"]**

> "When I submit this case, the agent pipeline kicks off. You can see each step executing in real time on the left."

**[SCREEN: AgentPipeline component showing steps lighting up one by one]**

> "Step 1 — MedGemma parses the free-text narrative into structured patient data: demographics, vitals, labs, medications, allergies, history."

**[Wait for Step 1 to complete]**

> "Step 2 — Clinical reasoning. MedGemma generates a ranked differential diagnosis with chain-of-thought reasoning. It's considering ACS, GERD, PE, aortic dissection — weighing evidence for and against each."

**[Wait for Step 2 to complete]**

> "Steps 3 and 4 run in parallel. Step 3 — Drug interaction check. This isn't the LLM guessing — it's querying the actual OpenFDA and RxNorm databases for his four medications. Real API data, not hallucination. Step 4 — Guideline retrieval. Our RAG system searches 62 curated clinical guidelines across 14 specialties. For this case it pulls the ACC/AHA chest pain and ACS guidelines."

**[Wait for Steps 3 & 4 to complete]**

> "Step 5 — and this is what makes it a real safety tool — Conflict Detection. MedGemma compares what the guidelines recommend against what the patient is actually receiving. It surfaces omissions, contradictions, dosage concerns, and monitoring gaps."

**[Wait for Step 5 to complete]**

> "Step 6 — Synthesis. Everything gets integrated into a single comprehensive report."

**[Wait for Step 6 to complete. Total pipeline ~2-3 minutes]**

---

### THE REPORT — Reviewing Results (2:00 – 2:40)

**[SCREEN: Scroll through the CDSReport component]**

> "Here's the CDS report. At the top — the ranked differential diagnosis. ACS is correctly identified as the leading diagnosis, with clear reasoning. The elevated troponin and ST elevation in II, III, and aVF support an inferior STEMI."

**[ACTION: Scroll to drug interactions section]**

> "Drug interaction warnings pulled from federal databases — not LLM-generated, real data."

**[ACTION: Scroll to Conflicts & Gaps section — highlight the red-bordered cards]**

> "This is the most important section — Conflicts and Gaps. Each card shows a specific conflict: what the guideline recommends, what the patient data shows, the severity, and a suggested resolution. These are the gaps that lead to missed diagnoses and omitted treatments in real clinical practice."

**[ACTION: Scroll to guidelines section]**

> "Cited guideline recommendations from authoritative sources — ACC/AHA, ADA, and others."

**[ACTION: Click the "Download .md" button in the left panel]**

> "And clinicians can download the full report as Markdown for their records."

---

### CLOSING — Technical & Impact (2:40 – 3:00)

**[SCREEN: Back to app overview or a summary slide]**

> "Under the hood: MedGemma 27B powers four of six pipeline steps — parsing, reasoning, conflict detection, and synthesis. It's augmented with OpenFDA and RxNorm APIs for drug safety, and a 62-guideline RAG corpus for evidence-based recommendations.
>
> We validated on 50 MedQA USMLE cases with 94% pipeline reliability and 38% diagnostic mention rate — before any fine-tuning.
>
> With 140 million ED visits per year in the U.S. alone, even a modest improvement in diagnostic completeness and medication safety represents lives saved. CDS Agent is built to make that happen."

**[END]**

---

## TIMING SUMMARY

| Section | Duration | Cumulative |
|---------|----------|------------|
| Opening — The Problem | 30 sec | 0:30 |
| Live Demo — Pipeline Execution | 90 sec | 2:00 |
| Report Review | 40 sec | 2:40 |
| Closing — Tech & Impact | 20 sec | 3:00 |

> **Note on timing:** The pipeline typically takes 2-3 minutes on the live endpoint. You can speed up the wait portions (1.5x-2x) in post-editing while keeping narration at normal speed to fit within 3 minutes. Alternatively, record narration separately and overlay it.

## TIPS

- **Warm up before recording** — Submit a test case first. If the model has scaled to zero you'll see a "Model Warming Up" spinner; wait for it to complete (~1-2 min) before the real recording
- **Speak during pipeline wait times** — the pipeline execution is perfect narration time
- **Don't rush** — the real-time pipeline visualization IS the demo; let it breathe
- **Zoom into the Conflicts section** — it's the most visually impressive and differentiating feature
- **If the endpoint is slow** — speed up wait portions in post-editing (1.5x-2x) while keeping narration at normal speed
- **Retry resilience** — if a pipeline run fails, the "Try Again" button lets you retry without reloading the page
- **Backup plan** — if the HF endpoint is down, you can use Google AI Studio with Gemma 3 27B IT as a fallback (update .env accordingly)
