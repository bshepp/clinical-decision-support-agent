# [Track B: RAG Variants]
"""
Track B — Guideline chunking strategies.

Splits whole-guideline documents into smaller segments before embedding.
The baseline (Track A) stores each guideline as a single document —
this module tests whether smaller chunks improve retrieval recall.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from tracks.rag_variants.config import ChunkStrategy


@dataclass
class Chunk:
    """A single chunk produced from a guideline document."""
    text: str
    source_guideline_id: str
    chunk_index: int
    metadata: dict


def chunk_guideline(
    guideline: dict,
    strategy: ChunkStrategy,
    guideline_index: int = 0,
) -> List[Chunk]:
    """
    Split a guideline dict into chunks according to the given strategy.

    Args:
        guideline: Dict with at least 'text', 'title', 'source', optionally 'id', 'specialty'.
        strategy: Which chunking method to apply.
        guideline_index: Fallback index for generating IDs.

    Returns:
        List of Chunk objects. For NONE strategy, returns a single chunk (the whole doc).
    """
    text = guideline.get("text", "")
    gid = guideline.get("id", f"guideline_{guideline_index}")
    base_meta = {
        "title": guideline.get("title", ""),
        "source": guideline.get("source", ""),
        "url": guideline.get("url", ""),
        "specialty": guideline.get("specialty", "General"),
        "parent_guideline_id": gid,
    }

    if strategy == ChunkStrategy.NONE:
        return [Chunk(text=text, source_guideline_id=gid, chunk_index=0, metadata=base_meta)]

    if strategy == ChunkStrategy.SENTENCE:
        segments = _split_sentences(text)
    elif strategy == ChunkStrategy.PARAGRAPH:
        segments = _split_paragraphs(text)
    elif strategy == ChunkStrategy.FIXED_256:
        segments = _split_fixed(text, window=256, overlap=0)
    elif strategy == ChunkStrategy.FIXED_512:
        segments = _split_fixed(text, window=512, overlap=0)
    elif strategy == ChunkStrategy.OVERLAP_256_64:
        segments = _split_fixed(text, window=256, overlap=64)
    else:
        segments = [text]

    # Filter out empty or very short chunks
    segments = [s.strip() for s in segments if len(s.strip()) > 20]

    return [
        Chunk(
            text=seg,
            source_guideline_id=gid,
            chunk_index=i,
            metadata={**base_meta, "chunk_index": i, "total_chunks": len(segments)},
        )
        for i, seg in enumerate(segments)
    ]


def chunk_all_guidelines(
    guidelines: List[dict],
    strategy: ChunkStrategy,
) -> List[Chunk]:
    """Chunk every guideline in a corpus."""
    all_chunks: List[Chunk] = []
    for idx, g in enumerate(guidelines):
        all_chunks.extend(chunk_guideline(g, strategy, guideline_index=idx))
    return all_chunks


# ──────────────────────────────────────────────
# Splitting helpers
# ──────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Rough estimate: 4 characters ≈ 1 token."""
    return max(1, len(text) // 4)


def _split_sentences(text: str) -> List[str]:
    """Split on sentence boundaries (period + space, or newline)."""
    # Simple regex: split at ". " or ".\n" but keep the period with the sentence
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p for p in parts if p.strip()]


def _split_paragraphs(text: str) -> List[str]:
    """Split on double-newline paragraph boundaries."""
    parts = re.split(r'\n\s*\n', text)
    return [p.strip() for p in parts if p.strip()]


def _split_fixed(text: str, window: int = 256, overlap: int = 0) -> List[str]:
    """
    Split text into chunks of approximately `window` tokens with optional overlap.

    Uses word boundaries to avoid cutting mid-word.
    """
    words = text.split()
    # Approximate: 1 token ≈ 0.75 words (conservative)
    words_per_window = int(window * 0.75)
    words_overlap = int(overlap * 0.75)

    if words_per_window < 1:
        words_per_window = 1
    step = max(1, words_per_window - words_overlap)

    chunks: List[str] = []
    for start in range(0, len(words), step):
        end = start + words_per_window
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break

    return chunks
