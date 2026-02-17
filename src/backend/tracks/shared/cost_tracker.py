# [Shared: Track Utilities]
"""
Cost Tracker — measures LLM token usage and estimated dollar cost per call.

Used by all experimental tracks to build cost/benefit charts.
Wraps MedGemmaService to intercept calls and record token counts.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional


# ──────────────────────────────────────────────
# Pricing constants (approximate, per 1K tokens)
# ──────────────────────────────────────────────

# HuggingFace Dedicated Endpoint: ~$2.50/hr for A100 80GB
# At ~20 tokens/sec throughput, that's roughly:
COST_PER_1K_INPUT_TOKENS = 0.0015    # ~$1.50 / 1M input tokens
COST_PER_1K_OUTPUT_TOKENS = 0.0020   # ~$2.00 / 1M output tokens


@dataclass
class LLMCallRecord:
    """Record of a single LLM call with cost metadata."""
    call_id: str
    track_id: str
    step_name: str
    iteration: int = 0                     # For iterative/arbitrated tracks
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0
    temperature: float = 0.0
    max_tokens_requested: int = 0
    estimated_cost_usd: float = 0.0
    timestamp: float = 0.0


@dataclass
class CostLedger:
    """
    Running ledger of all LLM calls for a track run.

    Provides aggregate cost, per-iteration cost breakdowns,
    and data for cost/benefit charts.
    """
    track_id: str
    calls: List[LLMCallRecord] = field(default_factory=list)

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.calls)

    @property
    def total_cost_usd(self) -> float:
        return sum(c.estimated_cost_usd for c in self.calls)

    @property
    def total_latency_ms(self) -> int:
        return sum(c.latency_ms for c in self.calls)

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def cost_at_iteration(self, iteration: int) -> float:
        """Cumulative cost through a given iteration."""
        return sum(c.estimated_cost_usd for c in self.calls if c.iteration <= iteration)

    def calls_at_iteration(self, iteration: int) -> List[LLMCallRecord]:
        """All calls for a specific iteration."""
        return [c for c in self.calls if c.iteration == iteration]

    def cost_per_iteration(self) -> dict[int, float]:
        """Map of iteration → incremental cost."""
        iterations = sorted(set(c.iteration for c in self.calls))
        return {
            i: sum(c.estimated_cost_usd for c in self.calls if c.iteration == i)
            for i in iterations
        }

    def to_dict(self) -> dict:
        """Serialize for JSON output."""
        return {
            "track_id": self.track_id,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_latency_ms": self.total_latency_ms,
            "call_count": self.call_count,
            "cost_per_iteration": {
                str(k): round(v, 6) for k, v in self.cost_per_iteration().items()
            },
        }


def estimate_tokens(text: str) -> int:
    """
    Rough token count estimation (4 chars ≈ 1 token for English text).

    This is an approximation. For precise counts, use the tokenizer directly.
    Good enough for cost/benefit comparisons across tracks.
    """
    return max(1, len(text) // 4)


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost for a single LLM call."""
    input_cost = (input_tokens / 1000) * COST_PER_1K_INPUT_TOKENS
    output_cost = (output_tokens / 1000) * COST_PER_1K_OUTPUT_TOKENS
    return input_cost + output_cost


def record_call(
    ledger: CostLedger,
    step_name: str,
    prompt: str,
    response: str,
    latency_ms: int,
    iteration: int = 0,
    temperature: float = 0.0,
    max_tokens: int = 0,
) -> LLMCallRecord:
    """
    Record an LLM call in the ledger.

    Call this after every MedGemma call in experimental tracks.
    """
    input_tokens = estimate_tokens(prompt)
    output_tokens = estimate_tokens(response)
    total_tokens = input_tokens + output_tokens
    cost = estimate_cost(input_tokens, output_tokens)

    record = LLMCallRecord(
        call_id=f"{ledger.track_id}_{step_name}_{iteration}_{len(ledger.calls)}",
        track_id=ledger.track_id,
        step_name=step_name,
        iteration=iteration,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        latency_ms=latency_ms,
        temperature=temperature,
        max_tokens_requested=max_tokens,
        estimated_cost_usd=cost,
        timestamp=time.time(),
    )
    ledger.calls.append(record)
    return record
