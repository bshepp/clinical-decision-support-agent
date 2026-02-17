# [Track B: RAG Variants]
"""
Track B — Variant configuration matrix.

Each RAGVariant defines ONE experiment: a specific combination of
chunking strategy, embedding model, top-k, and optional re-ranking.

Run `run_variants.py` to sweep the full matrix against the MedQA dataset.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class ChunkStrategy(str, Enum):
    """How to split guideline documents before embedding."""
    NONE = "none"                    # Baseline: one doc per guideline
    FIXED_256 = "fixed_256"          # 256-token fixed-window chunks
    FIXED_512 = "fixed_512"          # 512-token fixed-window chunks
    SENTENCE = "sentence"            # Split on sentence boundaries
    PARAGRAPH = "paragraph"          # Split on paragraph boundaries (double newline)
    OVERLAP_256_64 = "overlap_256_64"  # 256-token window, 64-token overlap


class EmbeddingModel(str, Enum):
    """Sentence-transformer models to compare."""
    MINILM_L6 = "sentence-transformers/all-MiniLM-L6-v2"          # Baseline: 384d, fast
    MINILM_L12 = "sentence-transformers/all-MiniLM-L12-v2"        # 384d, deeper
    MPNET = "sentence-transformers/all-mpnet-base-v2"              # 768d, best quality
    MEDCPT = "ncbi/MedCPT-Query-Encoder"                          # 768d, biomedical


@dataclass
class RAGVariant:
    """A single experimental configuration for Track B."""
    variant_id: str
    chunk_strategy: ChunkStrategy
    embedding_model: EmbeddingModel
    top_k: int = 5
    rerank: bool = False                # Cross-encoder reranking
    rerank_model: Optional[str] = None  # e.g. "cross-encoder/ms-marco-MiniLM-L-6-v2"
    description: str = ""


# ──────────────────────────────────────────────
# Variant matrix — add new experiments here
# ──────────────────────────────────────────────

VARIANTS: List[RAGVariant] = [
    # ---------- Baseline ----------
    RAGVariant(
        variant_id="B0_baseline",
        chunk_strategy=ChunkStrategy.NONE,
        embedding_model=EmbeddingModel.MINILM_L6,
        top_k=5,
        description="Exact baseline configuration (Track A equivalent)",
    ),

    # ---------- Chunking experiments (same embedding) ----------
    RAGVariant(
        variant_id="B1_fixed256",
        chunk_strategy=ChunkStrategy.FIXED_256,
        embedding_model=EmbeddingModel.MINILM_L6,
        top_k=5,
        description="256-token fixed chunks, baseline embeddings",
    ),
    RAGVariant(
        variant_id="B2_fixed512",
        chunk_strategy=ChunkStrategy.FIXED_512,
        embedding_model=EmbeddingModel.MINILM_L6,
        top_k=5,
        description="512-token fixed chunks, baseline embeddings",
    ),
    RAGVariant(
        variant_id="B3_sentence",
        chunk_strategy=ChunkStrategy.SENTENCE,
        embedding_model=EmbeddingModel.MINILM_L6,
        top_k=5,
        description="Sentence-level chunks, baseline embeddings",
    ),
    RAGVariant(
        variant_id="B4_overlap",
        chunk_strategy=ChunkStrategy.OVERLAP_256_64,
        embedding_model=EmbeddingModel.MINILM_L6,
        top_k=5,
        description="256-token chunks with 64-token overlap",
    ),

    # ---------- Embedding experiments (no chunking) ----------
    RAGVariant(
        variant_id="B5_mpnet",
        chunk_strategy=ChunkStrategy.NONE,
        embedding_model=EmbeddingModel.MPNET,
        top_k=5,
        description="MPNet embeddings, no chunking",
    ),
    RAGVariant(
        variant_id="B6_medcpt",
        chunk_strategy=ChunkStrategy.NONE,
        embedding_model=EmbeddingModel.MEDCPT,
        top_k=5,
        description="MedCPT biomedical embeddings, no chunking",
    ),

    # ---------- top-k sweep ----------
    RAGVariant(
        variant_id="B7_topk3",
        chunk_strategy=ChunkStrategy.NONE,
        embedding_model=EmbeddingModel.MINILM_L6,
        top_k=3,
        description="top-k=3 with baseline config",
    ),
    RAGVariant(
        variant_id="B8_topk10",
        chunk_strategy=ChunkStrategy.NONE,
        embedding_model=EmbeddingModel.MINILM_L6,
        top_k=10,
        description="top-k=10 with baseline config",
    ),

    # ---------- Best combo + rerank ----------
    RAGVariant(
        variant_id="B9_best_rerank",
        chunk_strategy=ChunkStrategy.OVERLAP_256_64,
        embedding_model=EmbeddingModel.MPNET,
        top_k=10,
        rerank=True,
        rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Best embedding + chunking + reranking",
    ),
]
