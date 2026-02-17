# [Track D: Arbitrated Parallel]
"""
Track D — Parallel Specialist Agents with Arbiter

Runs multiple domain-specialist reasoning agents in parallel,
collects their differentials, and feeds them to an arbiter that
synthesizes a consensus diagnosis with tailored resubmissions.

Files in this package may import from:
  - app.*           (baseline services, schemas, tools)
  - tracks.shared.* (cost tracking, comparison)
  - standard library / third-party packages

Files in this package must NOT import from:
  - tracks.rag_variants.*
  - tracks.iterative.*
"""
