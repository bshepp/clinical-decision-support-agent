"use client";

import { useState } from "react";

// Sample patient cases for quick testing
const SAMPLE_CASES = [
  {
    label: "Chest Pain (55M)",
    text: `55-year-old male presenting to the ED with acute onset substernal chest pain for the past 2 hours.
Pain is described as pressure-like, 8/10 severity, radiating to left arm and jaw. Associated with diaphoresis and nausea.

PMH: Hypertension, Type 2 Diabetes, Hyperlipidemia, Former smoker (30 pack-years, quit 5 years ago)
Medications: Metformin 1000mg BID, Lisinopril 20mg daily, Atorvastatin 40mg daily, Aspirin 81mg daily
Allergies: Penicillin (rash)

Vitals: BP 165/95, HR 98, Temp 98.6°F, RR 22, SpO2 95% on room air
Labs: Troponin I 0.45 ng/mL (ref <0.04), BMP normal, CBC unremarkable
ECG: ST elevation in leads II, III, aVF`,
  },
  {
    label: "Shortness of Breath (68F)",
    text: `68-year-old female presenting with worsening shortness of breath over 3 days, now dyspneic at rest.
Associated with bilateral lower extremity edema, orthopnea (uses 3 pillows), and PND.
Weight gain of 8 lbs in the past week.

PMH: CHF (EF 30%), Atrial fibrillation, CKD Stage 3, COPD
Medications: Carvedilol 25mg BID, Furosemide 40mg daily, Warfarin 5mg daily, Lisinopril 10mg daily, Albuterol PRN
Allergies: Sulfa drugs

Vitals: BP 142/88, HR 110 irregular, Temp 98.2°F, RR 28, SpO2 88% on room air
Labs: BNP 1850 pg/mL (ref <100), Creatinine 2.1 (baseline 1.5), K+ 5.2, Hgb 10.2
CXR: Bilateral pleural effusions, cardiomegaly, pulmonary vascular congestion`,
  },
  {
    label: "Diabetic Emergency (42M)",
    text: `42-year-old male brought by EMS with altered mental status. Found confused at home by family.
History of poorly controlled Type 1 Diabetes. Has not been taking insulin for 2 days due to running out.
Reports polyuria, polydipsia, abdominal pain, and vomiting for the past day.

PMH: Type 1 Diabetes (diagnosed age 12), Depression
Medications: Insulin glargine 30 units nightly, Insulin lispro sliding scale, Sertraline 100mg daily
Allergies: NKDA

Vitals: BP 100/60, HR 120, Temp 99.1°F, RR 32 (Kussmaul breathing), SpO2 98%
Labs: Glucose 485 mg/dL, pH 7.15, Bicarb 8, Anion gap 28, BHB 6.2, K+ 5.8 (but total body K+ depleted), Na+ 128, BUN 35, Cr 1.8
Urinalysis: Large ketones, glucose >1000`,
  },
];

interface PatientInputProps {
  onSubmit: (patientText: string) => void;
  isLoading: boolean;
}

export function PatientInput({ onSubmit, isLoading }: PatientInputProps) {
  const [text, setText] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (text.trim()) {
      onSubmit(text.trim());
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Sample case buttons */}
      <div>
        <label className="block text-sm font-medium text-gray-600 mb-2">
          Quick Start — Load a sample case:
        </label>
        <div className="flex flex-wrap gap-2">
          {SAMPLE_CASES.map((sample) => (
            <button
              key={sample.label}
              type="button"
              onClick={() => setText(sample.text)}
              className="px-3 py-1.5 text-sm bg-white border border-gray-300 rounded-md hover:bg-gray-50 transition-colors"
            >
              {sample.label}
            </button>
          ))}
        </div>
      </div>

      {/* Text input */}
      <div>
        <label
          htmlFor="patient-text"
          className="block text-sm font-medium text-gray-600 mb-2"
        >
          Patient Case Description
        </label>
        <textarea
          id="patient-text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={16}
          className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm font-mono"
          placeholder="Enter a patient case description including demographics, chief complaint, history, medications, labs, and vitals..."
          disabled={isLoading}
        />
      </div>

      {/* Submit */}
      <button
        type="submit"
        disabled={isLoading || !text.trim()}
        className="w-full py-3 px-6 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
      >
        {isLoading ? "Running Agent Pipeline..." : "Analyze Patient Case"}
      </button>
    </form>
  );
}
