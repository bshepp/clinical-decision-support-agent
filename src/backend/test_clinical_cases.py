"""
Comprehensive Clinical Test Suite for the CDS Agent.

Tests diverse clinical scenarios across specialties, acuity levels, demographics,
and edge cases. Each test case is a realistic patient presentation designed to
exercise different parts of the pipeline.

Usage:
    python test_clinical_cases.py                    # Run all cases sequentially
    python test_clinical_cases.py --case cardio_acs   # Run a single case by ID
    python test_clinical_cases.py --specialty cardio   # Run all cases in a specialty
    python test_clinical_cases.py --list              # List available cases
"""
import httpx
import asyncio
import json
import time
import sys
import argparse

API = "http://localhost:8002"

# ─────────────────────────────────────────────────
# Test Case Definitions
# ─────────────────────────────────────────────────

TEST_CASES = [
    # ── Cardiology ──
    {
        "id": "cardio_acs",
        "specialty": "Cardiology",
        "title": "Acute Coronary Syndrome — Classic STEMI",
        "expected_keywords": ["ACS", "STEMI", "troponin", "cath lab", "PCI", "aspirin", "heparin"],
        "patient_text": (
            "62-year-old male presenting to ED with crushing substernal chest pain radiating to "
            "left arm and jaw for 45 minutes. Diaphoretic and nauseated. History: HTN, "
            "hyperlipidemia, 30 pack-year smoking history (quit 5 years ago), father had MI at 55. "
            "Medications: atorvastatin 40mg daily, lisinopril 10mg daily, ASA 81mg daily. "
            "Vitals: BP 155/98, HR 105, RR 22, SpO2 94% on RA, Temp 37.2°C. "
            "ECG: ST elevation in leads II, III, aVF with reciprocal changes in I, aVL. "
            "Initial troponin I: 2.8 ng/mL (normal <0.04)."
        ),
        "include_drug_check": True,
        "include_guidelines": True,
    },
    {
        "id": "cardio_afib",
        "specialty": "Cardiology",
        "title": "New-Onset Atrial Fibrillation with Rapid Ventricular Response",
        "expected_keywords": ["atrial fibrillation", "rate control", "CHA2DS2-VASc", "anticoagulation", "DOAC"],
        "patient_text": (
            "73-year-old female with 2-day history of palpitations and lightheadedness. "
            "She also reports mild exertional dyspnea. No syncope. PMH: HTN, type 2 DM, "
            "osteoarthritis, hypothyroidism. Medications: metformin 500mg BID, levothyroxine "
            "100mcg daily, amlodipine 5mg daily, ibuprofen 400mg PRN (takes daily for knee pain). "
            "Vitals: BP 138/82, HR 142 (irregular), RR 18, SpO2 96%. "
            "ECG: atrial fibrillation with rapid ventricular response, no ST changes. "
            "Labs: TSH 0.8, K+ 4.2, Cr 1.1, BNP 350."
        ),
        "include_drug_check": True,
        "include_guidelines": True,
    },
    {
        "id": "cardio_hf",
        "specialty": "Cardiology",
        "title": "Acute Decompensated Heart Failure",
        "expected_keywords": ["heart failure", "HFrEF", "diuretic", "volume overload", "BNP", "ejection fraction"],
        "patient_text": (
            "58-year-old male presents with progressive dyspnea over 1 week, now orthopneic "
            "with 3-pillow requirement. Reports 10-lb weight gain. PMH: ischemic cardiomyopathy "
            "(EF 25% 6 months ago), HTN, CKD stage 3b (baseline Cr 2.1). Medications: carvedilol "
            "25mg BID, sacubitril-valsartan 97/103mg BID, spironolactone 25mg daily, furosemide "
            "40mg BID, dapagliflozin 10mg daily. Vitals: BP 98/62, HR 98, RR 26, SpO2 89% on RA. "
            "Exam: JVD to earlobes, bilateral crackles to mid-lung, S3, 3+ bilateral LE edema. "
            "Labs: BNP 2,800, Cr 2.8 (from 2.1), K+ 5.4, Na+ 128, troponin I 0.08."
        ),
        "include_drug_check": True,
        "include_guidelines": True,
    },
    # ── Emergency Medicine ──
    {
        "id": "em_stroke",
        "specialty": "Emergency Medicine",
        "title": "Acute Ischemic Stroke — tPA Window",
        "expected_keywords": ["stroke", "tPA", "alteplase", "NIHSS", "CT", "thrombolysis"],
        "patient_text": (
            "68-year-old female found by husband at 7:15 AM with right-sided weakness and difficulty "
            "speaking. Last known well at 11 PM the night before (went to bed normal). PMH: "
            "atrial fibrillation (not on anticoagulation — patient declined), HTN, "
            "hyperlipidemia. Medications: metoprolol 50mg BID, atorvastatin 20mg daily. "
            "Vitals: BP 182/105, HR 88 (irregular), RR 16, SpO2 97%. "
            "Exam: NIHSS 14 — right hemiparesis (arm 4/5, leg 3/5), expressive aphasia, "
            "right facial droop, neglect. CT head: no hemorrhage. CTA: left M1 occlusion."
        ),
        "include_drug_check": True,
        "include_guidelines": True,
    },
    {
        "id": "em_sepsis",
        "specialty": "Emergency Medicine",
        "title": "Sepsis from Urinary Source",
        "expected_keywords": ["sepsis", "antibiotics", "lactate", "fluid resuscitation", "vasopressor", "cultures"],
        "patient_text": (
            "82-year-old female nursing home resident brought to ED with confusion, fever, and "
            "decreased oral intake for 2 days. PMH: Alzheimer dementia, type 2 DM, recurrent "
            "UTIs, HTN, CKD stage 3a. Medications: donepezil 10mg daily, glipizide 5mg BID, "
            "lisinopril 10mg daily, cranberry supplement. "
            "Vitals: BP 85/50, HR 115, RR 26, SpO2 92% on RA, Temp 39.4°C (103°F). "
            "Exam: confused (GCS 13), dry mucous membranes, suprapubic tenderness. "
            "Labs: WBC 18.5K, lactate 4.2, Cr 2.4 (baseline 1.3), glucose 45, "
            "UA positive for nitrites, leukocyte esterase 3+, bacteria many."
        ),
        "include_drug_check": True,
        "include_guidelines": True,
    },
    {
        "id": "em_anaphylaxis",
        "specialty": "Emergency Medicine",
        "title": "Anaphylaxis — Peanut Allergy",
        "expected_keywords": ["anaphylaxis", "epinephrine", "airway", "allergic reaction", "antihistamine"],
        "patient_text": (
            "22-year-old male brought by EMS after eating a cookie at a party that contained "
            "peanuts. Known severe peanut allergy. Symptoms started 15 minutes after ingestion: "
            "diffuse urticaria, lip and tongue swelling, throat tightness, wheezing, "
            "lightheadedness. EpiPen administered by friend at scene. PMH: peanut allergy, "
            "asthma (mild intermittent). Medications: albuterol PRN, carries EpiPen. "
            "Vitals: BP 88/52, HR 130, RR 28, SpO2 89%, Temp 37.0°C. "
            "Exam: diffuse urticaria, angioedema of lips and tongue, stridor, bilateral wheezing, "
            "use of accessory muscles."
        ),
        "include_drug_check": True,
        "include_guidelines": True,
    },
    {
        "id": "em_trauma",
        "specialty": "Emergency Medicine",
        "title": "Polytrauma — MVC with Hemorrhagic Shock",
        "expected_keywords": ["trauma", "hemorrhage", "transfusion", "FAST", "ABC", "resuscitation"],
        "patient_text": (
            "35-year-old male restrained driver in high-speed MVC, significant front-end damage, "
            "prolonged extrication (30 min). GCS 12 (E3V4M5) in the field. C-collar in place. "
            "PMH: healthy, no medications, NKDA. Social: occasional alcohol (denies today). "
            "Vitals: BP 78/45, HR 135, RR 30, SpO2 88%, Temp 35.8°C. "
            "Primary survey: A — speaking in short sentences. B — decreased breath sounds left, "
            "subcutaneous crepitus left chest wall. C — tachycardic, thready pulses, no external "
            "hemorrhage, abdomen distended and tender. D — GCS 12. E — left leg deformity, "
            "pelvis unstable on compression. "
            "FAST: positive for free fluid in Morrison's pouch and pelvis. "
            "CXR: left hemopneumothorax. Labs pending."
        ),
        "include_drug_check": False,
        "include_guidelines": True,
    },
    # ── Endocrinology ──
    {
        "id": "endo_dka",
        "specialty": "Endocrinology",
        "title": "Diabetic Ketoacidosis — New-Onset T1DM",
        "expected_keywords": ["DKA", "insulin", "ketoacidosis", "potassium", "fluid resuscitation", "anion gap"],
        "patient_text": (
            "19-year-old male college student brought by roommate with nausea, vomiting, and "
            "confusion. Reports 3-week history of polyuria, polydipsia, and 15-lb weight loss. "
            "Denies alcohol or drug use. No significant PMH. Family history of type 1 diabetes "
            "(mother). No medications. "
            "Vitals: BP 100/60, HR 120, RR 32 (Kussmaul), SpO2 99%, Temp 37.1°C. "
            "Exam: dry mucous membranes, fruity breath odor, diffusely tender abdomen, "
            "lethargic but arousable (GCS 14). "
            "Labs: glucose 485 mg/dL, pH 7.12, bicarb 8, K+ 5.8, Na+ 131 (corrected 138), "
            "anion gap 28, BUN 32, Cr 1.6, serum ketones strongly positive, "
            "urine ketones 3+, A1C 13.2%."
        ),
        "include_drug_check": False,
        "include_guidelines": True,
    },
    {
        "id": "endo_thyroid_storm",
        "specialty": "Endocrinology",
        "title": "Thyroid Storm",
        "expected_keywords": ["thyroid storm", "PTU", "propylthiouracil", "beta-blocker", "hyperthyroidism", "Graves"],
        "patient_text": (
            "42-year-old female presents with high fever, agitation, tachycardia, and vomiting. "
            "She was recently diagnosed with Graves disease 2 months ago but ran out of "
            "methimazole 3 weeks ago and did not refill. She had a tooth extraction yesterday. "
            "PMH: Graves disease, anxiety. Medications: none (ran out of methimazole). "
            "Vitals: BP 160/70, HR 168, RR 28, SpO2 95%, Temp 40.2°C (104.4°F). "
            "Exam: agitated, tremulous, diaphoretic, exophthalmos, diffuse goiter with bruit, "
            "hyperactive bowel sounds, fine tremor, warm and flushed skin. "
            "Labs: TSH <0.01, free T4 7.8 (normal 0.8-1.7), free T3 22 (normal 2-4.4), "
            "WBC 11.2, AST 95, ALT 82, total bilirubin 2.1. "
            "Burch-Wartofsky Point Scale: 65 (highly suggestive of thyroid storm)."
        ),
        "include_drug_check": True,
        "include_guidelines": True,
    },
    {
        "id": "endo_adrenal_crisis",
        "specialty": "Endocrinology",
        "title": "Adrenal Crisis",
        "expected_keywords": ["adrenal crisis", "hydrocortisone", "Addison", "hypotension", "cortisol"],
        "patient_text": (
            "38-year-old female with known primary adrenal insufficiency (Addison disease) presents "
            "with 2-day history of GI illness (vomiting and diarrhea) and progressive weakness. "
            "She continued her usual hydrocortisone dose but could not keep medication down due to "
            "vomiting. Found by partner lying on bathroom floor, minimally responsive. "
            "PMH: Addison disease, hypothyroidism. Medications: hydrocortisone 15mg AM/5mg PM, "
            "fludrocortisone 0.1mg daily, levothyroxine 75mcg. "
            "Vitals: BP 72/40 (despite 1L NS in the field), HR 128, RR 22, SpO2 96%, Temp 38.5°C. "
            "Labs: Na+ 122, K+ 6.3, glucose 52, Cr 1.8, random cortisol 1.2 mcg/dL. "
            "Exam: obtunded (GCS 10), no focal neuro deficits, hyperpigmented skin creases."
        ),
        "include_drug_check": True,
        "include_guidelines": True,
    },
    # ── Pulmonology ──
    {
        "id": "pulm_pe",
        "specialty": "Pulmonology",
        "title": "Massive Pulmonary Embolism",
        "expected_keywords": ["pulmonary embolism", "anticoagulation", "thrombolysis", "D-dimer", "CTPA", "tPA"],
        "patient_text": (
            "45-year-old female presents with sudden-onset severe dyspnea and near-syncope while "
            "at work. She had right calf pain for 3 days. PMH: obesity (BMI 38), oral "
            "contraceptive use, recent 8-hour flight 1 week ago. Medications: combined OCP. "
            "Vitals: BP 82/50, HR 130, RR 32, SpO2 84% on high-flow O2, Temp 37.3°C. "
            "Exam: distressed, diaphoretic, JVD, loud P2, right calf swelling and tenderness. "
            "ECG: sinus tachycardia, S1Q3T3 pattern, right axis deviation. "
            "Labs: troponin I 1.2 (elevated), BNP 890, D-dimer >5000. "
            "Bedside echo: RV dilation with septal bowing, McConnell's sign. "
            "CTPA: bilateral extensive PE with saddle embolus at main PA bifurcation."
        ),
        "include_drug_check": True,
        "include_guidelines": True,
    },
    {
        "id": "pulm_asthma_exacerbation",
        "specialty": "Pulmonology",
        "title": "Severe Asthma Exacerbation (Status Asthmaticus)",
        "expected_keywords": ["asthma", "bronchodilator", "albuterol", "corticosteroid", "magnesium", "intubation"],
        "patient_text": (
            "28-year-old male with history of poorly controlled asthma presents to ED with severe "
            "dyspnea that started 6 hours ago. He ran out of his controller inhaler (fluticasone) "
            "2 weeks ago and has been using albuterol 6-8 times daily. Reports URI symptoms for "
            "3 days. PMH: asthma (2 ICU admissions, 1 intubation last year), allergic rhinitis, "
            "GERD. Medications: albuterol MDI PRN (using heavily), montelukast 10mg daily. "
            "Vitals: BP 140/88, HR 125, RR 36, SpO2 87% on NRB, Temp 37.4°C. "
            "Exam: tripod position, accessory muscle use, speaking in 1-2 word sentences, "
            "bilateral inspiratory and expiratory wheezes with decreased air entry at bases, "
            "pulsus paradoxus 22 mmHg. PEFR: unable to perform. "
            "ABG: pH 7.28, pCO2 52, pO2 58. "
            "Given 3 rounds of continuous albuterol/ipratropium, IV methylprednisolone — minimal improvement."
        ),
        "include_drug_check": True,
        "include_guidelines": True,
    },
    # ── Gastroenterology ──
    {
        "id": "gi_ugib",
        "specialty": "Gastroenterology",
        "title": "Upper GI Bleeding — Peptic Ulcer",
        "expected_keywords": ["GI bleeding", "hematemesis", "PPI", "endoscopy", "transfusion", "H. pylori"],
        "patient_text": (
            "65-year-old male presents with 3 episodes of hematemesis (bright red blood) over "
            "the past 4 hours. Also reports melena for 2 days. Mild epigastric pain. "
            "PMH: osteoarthritis, atrial fibrillation on warfarin. Medications: warfarin 5mg daily, "
            "naproxen 500mg BID (started 2 weeks ago for arthritis flare), omeprazole 20mg daily "
            "(ran out 1 month ago). Allergies: sulfa. "
            "Vitals: BP 92/58, HR 118, RR 20, SpO2 97%, Temp 36.8°C. "
            "Exam: pale, diaphoretic, epigastric tenderness without rebound, "
            "rectal exam positive for melena. "
            "Labs: Hb 6.8 (from 13.2 three months ago), INR 3.8, BUN 45, Cr 1.0, "
            "platelets 195K. Glasgow-Blatchford Score: 14."
        ),
        "include_drug_check": True,
        "include_guidelines": True,
    },
    {
        "id": "gi_pancreatitis",
        "specialty": "Gastroenterology",
        "title": "Acute Gallstone Pancreatitis",
        "expected_keywords": ["pancreatitis", "lipase", "gallstone", "fluid resuscitation", "cholecystectomy"],
        "patient_text": (
            "48-year-old female presents with severe epigastric pain radiating to the back for "
            "12 hours, worsening after a fatty meal. Nausea and vomiting x6 episodes. "
            "PMH: cholelithiasis (known but declined surgery), obesity (BMI 34), "
            "hyperlipidemia. Medications: simvastatin 20mg daily. "
            "Vitals: BP 105/70, HR 108, RR 22, SpO2 96%, Temp 38.1°C. "
            "Exam: moderate distress, epigastric tenderness with guarding, decreased bowel sounds, "
            "positive Murphy's sign. "
            "Labs: lipase 2,850 U/L (normal <60), amylase 1,200, WBC 15.2K, "
            "total bilirubin 3.2, direct bilirubin 2.5, ALT 285, AST 312, "
            "Alk phos 380, triglycerides 220, Cr 0.9. "
            "CT abdomen: diffuse pancreatic edema and peripancreatic fluid, "
            "gallbladder with multiple stones, dilated CBD 10mm."
        ),
        "include_drug_check": True,
        "include_guidelines": True,
    },
    # ── Neurology ──
    {
        "id": "neuro_seizure",
        "specialty": "Neurology",
        "title": "Status Epilepticus",
        "expected_keywords": ["seizure", "status epilepticus", "benzodiazepine", "lorazepam", "antiepileptic", "levetiracetam"],
        "patient_text": (
            "34-year-old male with known epilepsy brought by EMS with continuous seizure activity "
            "for >10 minutes that has not stopped. Bystander reports he was walking, stiffened, "
            "and fell with tonic-clonic activity. EMS gave midazolam 10mg IM 5 minutes ago with "
            "brief pause, but seizure activity resumed. PMH: focal epilepsy (left temporal lesion), "
            "prior breakthrough seizures (non-compliant with meds). Medications: levetiracetam "
            "750mg BID (admits he hasn't taken it in 1 week). "
            "Vitals: BP 175/95, HR 130, RR 8 (postictal), SpO2 85%, Temp 38.2°C. "
            "Exam: continuous left-gaze deviation with rhythmic bilateral limb jerking, "
            "no verbal response, bite mark on tongue, urinary incontinence."
        ),
        "include_drug_check": True,
        "include_guidelines": True,
    },
    {
        "id": "neuro_meningitis",
        "specialty": "Neurology",
        "title": "Bacterial Meningitis",
        "expected_keywords": ["meningitis", "lumbar puncture", "ceftriaxone", "vancomycin", "dexamethasone", "CSF"],
        "patient_text": (
            "20-year-old male college student presents with 18-hour history of severe headache, "
            "fever, and neck stiffness. Roommate reports he's been confused for the past 3 hours. "
            "Recently had an upper respiratory infection. Lives in a dormitory. "
            "PMH: healthy, no medications, vaccinations up to date (received meningococcal "
            "conjugate vaccine but not serogroup B vaccine). "
            "Vitals: BP 98/62, HR 112, RR 24, SpO2 96%, Temp 39.8°C (103.6°F). "
            "Exam: appears toxic, positive Kernig and Brudzinski signs, photophobia, "
            "no focal neurologic deficits, GCS 12 (E3V4M5), petechial rash on trunk and extremities. "
            "Labs: WBC 22K with 92% PMNs, lactate 3.1, Cr 1.0, platelets 120K."
        ),
        "include_drug_check": False,
        "include_guidelines": True,
    },
    # ── Psychiatry ──
    {
        "id": "psych_suicide",
        "specialty": "Psychiatry",
        "title": "Acute Suicidal Ideation with Plan",
        "expected_keywords": ["suicide", "safety", "psychiatr", "risk assessment", "lethal means", "hospitalization"],
        "patient_text": (
            "45-year-old male veteran brought to ED by wife after she found a note. He reports "
            "active suicidal ideation with plan to use his firearm (has access at home). States "
            "he's been increasingly hopeless since job loss 3 months ago, not sleeping, lost "
            "20 lbs, withdrawn from family and friends. Drinks 6-8 beers daily (increased from "
            "occasional). PMH: PTSD (combat-related), major depressive disorder, chronic back pain. "
            "Medications: sertraline 100mg daily, trazodone 50mg QHS, ibuprofen 600mg TID. "
            "Denies prior suicide attempts. Father completed suicide at age 50. "
            "Vitals: stable. Exam: sad affect, poor eye contact, psychomotor retardation, "
            "coherent but hopeless in content. PHQ-9 score: 24 (severe). "
            "Columbia Suicide Severity Rating Scale: active ideation with specific plan and intent."
        ),
        "include_drug_check": True,
        "include_guidelines": True,
    },
    # ── Pediatrics ──
    {
        "id": "peds_fever",
        "specialty": "Pediatrics",
        "title": "Febrile Neonate — 21-day-old",
        "expected_keywords": ["neonate", "fever", "sepsis workup", "lumbar puncture", "ampicillin", "cefotaxime"],
        "patient_text": (
            "21-day-old male infant brought by concerned mother for fever. Temperature measured "
            "at home: 38.4°C (101.1°F) rectal. Born full-term via uncomplicated vaginal delivery, "
            "GBS negative. Breastfeeding well until today — decreased feeds for past 6 hours, "
            "seems more sleepy than usual. No cough, rash, or vomiting. "
            "One older sibling with runny nose. "
            "Vitals: Temp 38.6°C (101.5°F) rectal, HR 180 (normal 100-160 for age), "
            "RR 44, SpO2 98%. Birth weight 3.4 kg, current weight 3.7 kg. "
            "Exam: slightly lethargic, soft fontanelle, no bulging, no rash, "
            "cap refill 3 seconds, mottled skin, abdomen soft, no organomegaly. "
            "Labs: WBC 22K, ANC 6500, CRP 35 mg/L, procalcitonin 0.8 ng/mL, "
            "urinalysis negative."
        ),
        "include_drug_check": False,
        "include_guidelines": True,
    },
    {
        "id": "peds_dehydration",
        "specialty": "Pediatrics",
        "title": "Severe Dehydration — Pediatric Gastroenteritis",
        "expected_keywords": ["dehydration", "fluid", "oral rehydration", "IV fluid", "bolus", "electrolyte"],
        "patient_text": (
            "2-year-old female brought by parents for 3 days of watery diarrhea (8-10 episodes/day) "
            "and vomiting (4-5 times/day). Decreased oral intake, last wet diaper >12 hours ago. "
            "Daycare contacts have similar illness. PMH: healthy, vaccinations up to date including "
            "rotavirus. No medications. Allergies: NKDA. "
            "Vitals: HR 170 (tachycardic for age), BP 72/45, RR 30, SpO2 99%, "
            "Temp 38.3°C, Weight 10.5 kg (12 kg at well child visit 1 month ago — 12.5% weight loss). "
            "Exam: lethargic but arousable, sunken eyes, sunken anterior fontanelle, "
            "dry oral mucosa with no tears, decreased skin turgor (tenting >2 seconds), "
            "cap refill 4 seconds, cool extremities, abdomen diffusely tender, "
            "hyperactive bowel sounds."
        ),
        "include_drug_check": False,
        "include_guidelines": True,
    },
    # ── Nephrology ──
    {
        "id": "renal_hyperkalemia",
        "specialty": "Nephrology",
        "title": "Severe Hyperkalemia with ECG Changes",
        "expected_keywords": ["hyperkalemia", "calcium", "insulin", "dextrose", "dialysis", "potassium", "ECG"],
        "patient_text": (
            "72-year-old male with ESRD on hemodialysis (missed last 2 sessions due to "
            "transportation issues) presents with generalized weakness and palpitations. "
            "PMH: ESRD on MWF HD, type 2 DM, HTN, peripheral vascular disease. "
            "Medications: sevelamer 800mg TID, calcitriol 0.25mcg daily, EPO injection "
            "at dialysis, amlodipine 10mg, insulin glargine 20 units nightly, insulin lispro "
            "sliding scale. "
            "Vitals: BP 190/105, HR 52 (bradycardic), RR 22, SpO2 94%, Temp 36.5°C. "
            "ECG: widened QRS (140ms), peaked T waves globally, loss of P waves, "
            "junctional bradycardia. "
            "Labs: K+ 7.8, BUN 98, Cr 11.2, bicarb 14, pH 7.22, glucose 220."
        ),
        "include_drug_check": True,
        "include_guidelines": True,
    },
    # ── Infectious Disease ──
    {
        "id": "id_pneumonia",
        "specialty": "Infectious Disease",
        "title": "Community-Acquired Pneumonia — Moderate Severity",
        "expected_keywords": ["pneumonia", "antibiotic", "CURB-65", "ceftriaxone", "azithromycin", "chest x-ray"],
        "patient_text": (
            "58-year-old male presents with 4-day history of productive cough (yellow-green sputum), "
            "fever, chills, and pleuritic right-sided chest pain. Reports dyspnea with minimal "
            "exertion. PMH: COPD (FEV1 55% predicted), type 2 DM, moderate alcohol use "
            "(2-3 drinks daily). Medications: tiotropium 18mcg inhaled daily, albuterol PRN, "
            "metformin 1000mg BID. Former 20 pack-year smoker, quit 5 years ago. "
            "Vitals: BP 128/78, HR 102, RR 26, SpO2 90% on RA, Temp 39.2°C (102.6°F). "
            "Exam: appears ill, right basilar crackles with bronchial breath sounds, "
            "egophony right base, no wheezing currently. "
            "CXR: right lower lobe consolidation with air bronchograms. "
            "Labs: WBC 16.8K, BUN 28, Cr 1.0, procalcitonin 3.2, lactate 1.8. "
            "CURB-65 score: 2 (confusion absent, urea elevated, RR ≥30, BP normal, age <65)."
        ),
        "include_drug_check": True,
        "include_guidelines": True,
    },
    # ── Hematology ──
    {
        "id": "heme_dvt_pe",
        "specialty": "Hematology",
        "title": "DVT with Moderate PE — Cancer Patient",
        "expected_keywords": ["DVT", "PE", "anticoagulation", "LMWH", "cancer", "thrombosis"],
        "patient_text": (
            "62-year-old female with recently diagnosed pancreatic adenocarcinoma (on chemotherapy, "
            "cycle 2 of gemcitabine/nab-paclitaxel) presents with 5-day left leg swelling and "
            "new-onset dyspnea on exertion since yesterday. No chest pain, hemoptysis, or syncope. "
            "PMH: pancreatic cancer stage III (diagnosed 6 weeks ago), HTN, hypothyroidism. "
            "Medications: gemcitabine/nab-paclitaxel q28d, ondansetron PRN, levothyroxine 88mcg, "
            "lisinopril 10mg. "
            "Vitals: BP 118/72, HR 98, RR 22, SpO2 93% on RA, Temp 37.0°C. "
            "Exam: left leg circumference 4cm greater than right, tender calf, "
            "Homan's sign positive. Lungs CTA bilaterally. "
            "Labs: D-dimer 8,500, Hb 10.2, plt 85K, INR 1.0, Cr 0.8. "
            "LE Doppler: occlusive thrombus left femoral and popliteal veins. "
            "CTPA: segmental PE in right lower lobe pulmonary artery. "
            "Troponin normal, BNP 120, RV normal on echo."
        ),
        "include_drug_check": True,
        "include_guidelines": True,
    },
    # ── OB/GYN ──
    {
        "id": "obgyn_preeclampsia",
        "specialty": "OB/GYN",
        "title": "Severe Preeclampsia at 33 Weeks",
        "expected_keywords": ["preeclampsia", "magnesium sulfate", "blood pressure", "HELLP", "delivery"],
        "patient_text": (
            "28-year-old G2P1 at 33 weeks gestation presents with severe headache unresponsive "
            "to acetaminophen, visual disturbances (scotomata), and right upper quadrant pain "
            "for 6 hours. First pregnancy was uncomplicated. This pregnancy: mild chronic HTN, "
            "started on labetalol 200mg BID at 16 weeks. Started low-dose aspirin at 12 weeks. "
            "Medications: labetalol 200mg BID, aspirin 81mg daily, prenatal vitamins. "
            "Vitals: BP 178/112 (confirmed on repeat after 15 min: 174/108), HR 92, "
            "RR 18, SpO2 98%, Temp 37.0°C. "
            "Exam: 3+ pitting edema bilateral LE, RUQ tenderness, brisk reflexes (3+) "
            "with 2 beats of clonus bilaterally. Fetal heart tones 140s, reassuring. "
            "Labs: PLT 82K, AST 220, ALT 195, LDH 650, Cr 1.3, uric acid 8.2, "
            "protein/creatinine ratio 4.8, peripheral smear: schistocytes present."
        ),
        "include_drug_check": True,
        "include_guidelines": True,
    },
    # ── Toxicology ──
    {
        "id": "tox_acetaminophen",
        "specialty": "Toxicology",
        "title": "Acetaminophen Overdose",
        "expected_keywords": ["acetaminophen", "NAC", "N-acetylcysteine", "Rumack", "liver", "antidote"],
        "patient_text": (
            "24-year-old female brought to ED by friend 6 hours after intentionally ingesting "
            "~30 tablets of Extra Strength Tylenol (500mg each = ~15g total). She now has nausea "
            "and vomiting. Reports this was a suicide attempt after breakup. Currently expressing "
            "regret and requesting help. PMH: depression, anxiety. "
            "Medications: escitalopram 10mg daily. No OTC meds regularly. "
            "Vitals: BP 108/68, HR 92, RR 16, SpO2 99%, Temp 37.0°C. "
            "Exam: mild epigastric tenderness, otherwise unremarkable. "
            "Labs: APAP level at 6 hours: 180 mcg/mL (treatment line at 4h is 150), "
            "AST 42, ALT 38, INR 1.0, Cr 0.7, glucose 95. "
            "Above treatment line on Rumack-Matthew nomogram."
        ),
        "include_drug_check": True,
        "include_guidelines": True,
    },
    # ── Multi-system / Complex ──
    {
        "id": "complex_polypharmacy",
        "specialty": "Geriatrics",
        "title": "Elderly Polypharmacy with Falls and AKI",
        "expected_keywords": ["fall", "polypharmacy", "kidney", "AKI", "dehydration", "medication review"],
        "patient_text": (
            "84-year-old female brought from assisted living after being found on the floor this "
            "morning. Staff reports she's had decreased oral intake for 3 days due to 'stomach bug.' "
            "History of 2 falls in the past month. PMH: HTN, CHF (EF 45%), atrial fibrillation, "
            "type 2 DM, CKD stage 3 (baseline Cr 1.4), osteoporosis, depression, insomnia, "
            "osteoarthritis. "
            "Medications: metoprolol succinate 100mg daily, apixaban 5mg BID, lisinopril 20mg, "
            "furosemide 40mg daily, metformin 500mg BID, spironolactone 25mg, amlodipine 5mg, "
            "mirtazapine 15mg QHS, zolpidem 5mg QHS, alendronate 70mg weekly, "
            "calcium/vitamin D, aspirin 81mg, acetaminophen 650mg TID. "
            "Vitals: BP 88/52 (lying), 62/40 (sitting — orthostatic), HR 48 (irregular), "
            "RR 18, SpO2 95%, Temp 36.2°C. "
            "Exam: dry mucous membranes, irregular irregularly rhythm, clear lungs, "
            "mild confusion (baseline oriented x3, now x1), right hip tenderness. "
            "Labs: Na+ 126, K+ 6.1, BUN 62, Cr 3.2 (from 1.4), glucose 52, TSH 4.5, "
            "Hb 10.2, WBC 9.8, INR 1.0 (on apixaban). "
            "Hip XR: right intertrochanteric hip fracture."
        ),
        "include_drug_check": True,
        "include_guidelines": True,
    },
]


# ─────────────────────────────────────────────────
# Test Runner
# ─────────────────────────────────────────────────

async def run_case(client: httpx.AsyncClient, case: dict, verbose: bool = True) -> dict:
    """Submit a case and poll until done. Returns result dict with timing."""
    case_id_label = case["id"]
    title = case["title"]

    if verbose:
        print(f"\n{'='*70}")
        print(f"  [{case_id_label}] {title}")
        print(f"  Specialty: {case['specialty']}")
        print(f"{'='*70}")

    start = time.time()

    # Submit
    body = {
        "patient_text": case["patient_text"],
        "include_drug_check": case.get("include_drug_check", True),
        "include_guidelines": case.get("include_guidelines", True),
    }
    r = await client.post(f"{API}/api/cases/submit", json=body)
    if r.status_code != 200:
        return {"case_id": case_id_label, "error": f"Submit failed: {r.status_code} {r.text}", "elapsed": 0}

    data = r.json()
    server_case_id = data["case_id"]
    if verbose:
        print(f"  Submitted: {server_case_id}")

    # Poll
    result = None
    steps = []
    for i in range(90):  # up to 7.5 minutes
        await asyncio.sleep(5)
        r = await client.get(f"{API}/api/cases/{server_case_id}")
        result = r.json()
        state = result.get("state", {})
        steps = state.get("steps", [])

        if verbose and i % 3 == 0:
            statuses = [f"{s['step_id']}={s['status']}" for s in steps]
            print(f"    [{i*5}s] {', '.join(statuses)}")

        all_done = all(s["status"] in ("completed", "failed", "skipped") for s in steps)
        if all_done:
            break

    elapsed = round(time.time() - start, 1)

    # Analyze results
    step_summary = {}
    for s in steps:
        step_summary[s["step_id"]] = {
            "status": s["status"],
            "duration_ms": s.get("duration_ms", 0),
            "error": s.get("error", ""),
        }

    report = result.get("report") if result else None
    all_passed = all(s["status"] == "completed" for s in steps)
    any_failed = any(s["status"] == "failed" for s in steps)

    # Check expected keywords in report
    keyword_hits = []
    keyword_misses = []
    if report:
        report_text = json.dumps(report).lower()
        for kw in case.get("expected_keywords", []):
            if kw.lower() in report_text:
                keyword_hits.append(kw)
            else:
                keyword_misses.append(kw)

    result_data = {
        "case_id": case_id_label,
        "title": title,
        "specialty": case["specialty"],
        "all_passed": all_passed,
        "any_failed": any_failed,
        "elapsed_seconds": elapsed,
        "steps": step_summary,
        "keyword_hits": keyword_hits,
        "keyword_misses": keyword_misses,
        "keyword_coverage": (
            f"{len(keyword_hits)}/{len(keyword_hits) + len(keyword_misses)}"
            if (keyword_hits or keyword_misses) else "N/A"
        ),
        "report": report,
    }

    if verbose:
        print(f"\n  Results ({elapsed}s total):")
        for sid, info in step_summary.items():
            status_icon = "✓" if info["status"] == "completed" else ("✗" if info["status"] == "failed" else "○")
            print(f"    {status_icon} {sid:12s} {info['status']:10s} ({info['duration_ms']}ms)")
            if info["error"]:
                print(f"      ERROR: {info['error'][:120]}")

        if report:
            print(f"\n  Keywords found: {', '.join(keyword_hits) if keyword_hits else 'none'}")
            if keyword_misses:
                print(f"  Keywords missing: {', '.join(keyword_misses)}")
            print(f"  Keyword coverage: {result_data['keyword_coverage']}")

            # Print condensed report
            if verbose:
                print(f"\n  --- Report Summary ---")
                print(f"  Patient: {report.get('patient_summary', 'N/A')[:200]}")
                dx = report.get("differential_diagnosis", [])
                if dx:
                    print(f"  Top diagnosis: {dx[0].get('diagnosis', 'N/A')} ({dx[0].get('likelihood', 'N/A')})")
                warnings = report.get("drug_interaction_warnings", [])
                if warnings:
                    print(f"  Drug warnings: {len(warnings)}")
                recs = report.get("guideline_recommendations", [])
                if recs:
                    print(f"  Guideline recs: {len(recs)}")
                steps_rec = report.get("suggested_next_steps", [])
                if steps_rec:
                    print(f"  Next steps: {len(steps_rec)}")
        else:
            print(f"  ⚠ No report generated")

    return result_data


async def main():
    parser = argparse.ArgumentParser(description="CDS Agent Clinical Test Suite")
    parser.add_argument("--case", help="Run a single test case by ID")
    parser.add_argument("--specialty", help="Run all cases for a specialty (partial match)")
    parser.add_argument("--list", action="store_true", help="List available test cases")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument("--report", help="Save results to JSON file")
    args = parser.parse_args()

    if args.list:
        print(f"\nAvailable test cases ({len(TEST_CASES)} total):\n")
        by_specialty = {}
        for tc in TEST_CASES:
            by_specialty.setdefault(tc["specialty"], []).append(tc)
        for spec, cases in sorted(by_specialty.items()):
            print(f"  {spec}:")
            for c in cases:
                print(f"    {c['id']:30s} {c['title']}")
        return

    # Filter cases
    cases_to_run = TEST_CASES
    if args.case:
        cases_to_run = [c for c in TEST_CASES if c["id"] == args.case]
        if not cases_to_run:
            print(f"Case '{args.case}' not found. Use --list to see available cases.")
            return
    elif args.specialty:
        cases_to_run = [c for c in TEST_CASES if args.specialty.lower() in c["specialty"].lower()]
        if not cases_to_run:
            print(f"No cases found for specialty '{args.specialty}'. Use --list.")
            return

    verbose = not args.quiet
    print(f"\n{'#'*70}")
    print(f"  CDS Agent — Clinical Test Suite")
    print(f"  Running {len(cases_to_run)} test case(s)")
    print(f"  API: {API}")
    print(f"{'#'*70}")

    # Check health
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            health = await client.get(f"{API}/api/health")
            if health.status_code != 200:
                print(f"\n  ✗ Backend health check failed: {health.status_code}")
                return
            print(f"  ✓ Backend healthy\n")
        except Exception as e:
            print(f"\n  ✗ Cannot reach backend at {API}: {e}")
            return

        results = []
        for case in cases_to_run:
            try:
                result = await run_case(client, case, verbose=verbose)
                results.append(result)
            except Exception as e:
                print(f"\n  ✗ Exception running case {case['id']}: {e}")
                results.append({
                    "case_id": case["id"],
                    "title": case["title"],
                    "error": str(e),
                    "all_passed": False,
                })

    # Summary
    print(f"\n\n{'#'*70}")
    print(f"  SUMMARY — {len(results)} cases")
    print(f"{'#'*70}\n")

    passed = sum(1 for r in results if r.get("all_passed"))
    failed = sum(1 for r in results if r.get("any_failed"))
    total_time = sum(r.get("elapsed_seconds", 0) for r in results)

    for r in results:
        icon = "✓" if r.get("all_passed") else "✗"
        kw = r.get("keyword_coverage", "N/A")
        elapsed = r.get("elapsed_seconds", 0)
        print(f"  {icon} [{r['case_id']:30s}] {elapsed:6.1f}s  keywords:{kw:>5s}  {r.get('title', '')}")

    print(f"\n  Passed: {passed}/{len(results)}")
    print(f"  Failed: {failed}/{len(results)}")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")

    # Save report
    if args.report:
        with open(args.report, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to {args.report}")


if __name__ == "__main__":
    asyncio.run(main())
