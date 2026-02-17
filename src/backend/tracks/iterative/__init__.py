# [Track C: Iterative Refinement]
"""
Track C — Serial Iterative Diagnosis Refinement

Passes the differential through repeated reasoning rounds, each time
asking the model to self-critique and refine, until the cost/benefit
curve flattens or a max iteration cap is reached.

Files in this package may import from:
  - app.*           (baseline services, schemas, tools)
  - tracks.shared.* (cost tracking, comparison)
  - standard library / third-party packages

Files in this package must NOT import from:
  - tracks.rag_variants.*
  - tracks.arbitrated.*
"""
