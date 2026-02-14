"""
Validation framework for the Clinical Decision Support Agent.

Validates the CDS pipeline against three external clinical datasets:
  - MedQA (USMLE-style questions) — diagnostic accuracy
  - MTSamples (medical transcriptions) — parse robustness
  - PMC Case Reports (published cases) — real-world diagnostic accuracy
"""
