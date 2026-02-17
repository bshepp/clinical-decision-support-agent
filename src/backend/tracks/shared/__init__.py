# [Shared: Track Utilities]
"""
Shared utilities for the experimental track system.

This package provides cross-track tools:
  - cost_tracker: Token/dollar cost accounting per LLM call
  - compare: Cross-track result comparison and chart generation

Import rules:
  - May import from app/ (Track A baseline)
  - May NOT import from any specific track (rag_variants/, iterative/, arbitrated/)
  - All tracks may import from here
"""
