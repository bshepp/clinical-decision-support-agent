# [Track C: Iterative Refinement]
"""
Track C — Configuration for iterative refinement experiments.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IterativeConfig:
    """Configuration for one iterative-refinement experiment."""
    config_id: str
    max_iterations: int = 5
    convergence_threshold: float = 0.05
    """
    Stop early if the cosine similarity between consecutive differential
    outputs exceeds (1 - convergence_threshold) — i.e. the model is
    repeating itself and further iterations add no value.
    """
    temperature: float = 0.3
    critique_temperature: float = 0.4
    """
    Slightly higher temperature for the self-critique prompt so the model
    is more willing to challenge its own prior answer.
    """
    max_tokens_reasoning: int = 3072
    max_tokens_critique: int = 2048
    description: str = ""


# ──────────────────────────────────────────────
# Experiment configurations
# ──────────────────────────────────────────────

CONFIGS = [
    IterativeConfig(
        config_id="C0_2rounds",
        max_iterations=2,
        description="Baseline: single self-critique pass",
    ),
    IterativeConfig(
        config_id="C1_3rounds",
        max_iterations=3,
        description="Two self-critique passes",
    ),
    IterativeConfig(
        config_id="C2_5rounds",
        max_iterations=5,
        description="Four self-critique passes — watch for cost/benefit inflection",
    ),
    IterativeConfig(
        config_id="C3_aggressive",
        max_iterations=5,
        critique_temperature=0.6,
        convergence_threshold=0.03,
        description="More aggressive critic (higher temp), tighter convergence gate",
    ),
]
