# [Track B: RAG Variants]
"""
Track B — Modified retriever that replaces the baseline RAG system.

Creates a per-variant ChromaDB collection, chunks guidelines accordingly,
and provides the same `run(query, n_results)` interface as the baseline
GuidelineRetrievalTool so it can be swapped into the pipeline cleanly.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from app.models.schemas import GuidelineExcerpt, GuidelineRetrievalResult
from tracks.rag_variants.config import RAGVariant
from tracks.rag_variants.chunker import chunk_all_guidelines

logger = logging.getLogger(__name__)

# Same source as Track A
GUIDELINES_DATA_PATH = Path(__file__).resolve().parent.parent.parent / "app" / "data" / "clinical_guidelines.json"
CHROMA_BASE_DIR = Path(__file__).resolve().parent / "data" / "chroma"


class VariantRetriever:
    """
    Drop-in replacement for GuidelineRetrievalTool that uses a variant config.

    Each variant gets its own ChromaDB collection so experiments don't interfere.
    """

    def __init__(self, variant: RAGVariant):
        self.variant = variant
        self._collection = None
        self._embedding_fn = None
        self._reranker = None

    async def _ensure_initialized(self):
        """Lazy-init ChromaDB collection with variant-specific config."""
        if self._collection is not None:
            return

        import chromadb
        from chromadb.utils import embedding_functions

        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.variant.embedding_model.value,
        )

        persist_dir = str(CHROMA_BASE_DIR / self.variant.variant_id)
        client = chromadb.PersistentClient(path=persist_dir)

        collection_name = f"trackB_{self.variant.variant_id}"
        # Truncate to ChromaDB's 63-char limit
        collection_name = collection_name[:63]

        self._collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        # Populate if empty
        if self._collection.count() == 0:
            await self._populate()

        # Lazy-load reranker if configured
        if self.variant.rerank and self.variant.rerank_model:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(self.variant.rerank_model)
                logger.info(f"Loaded reranker: {self.variant.rerank_model}")
            except ImportError:
                logger.warning("sentence-transformers not installed; skipping reranker")

    async def _populate(self):
        """Load, chunk, and embed guidelines into the collection."""
        guidelines = self._load_guidelines()
        if not guidelines:
            logger.warning("No guidelines found — retriever will be empty")
            return

        chunks = chunk_all_guidelines(guidelines, self.variant.chunk_strategy)
        logger.info(
            f"[{self.variant.variant_id}] {len(guidelines)} guidelines → "
            f"{len(chunks)} chunks (strategy: {self.variant.chunk_strategy.value})"
        )

        documents = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]
        ids = [
            f"{c.source_guideline_id}_chunk{c.chunk_index}"
            for c in chunks
        ]

        # ChromaDB's add() may choke on very large batches — split to 500
        batch = 500
        for start in range(0, len(documents), batch):
            end = start + batch
            self._collection.add(
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end],
            )

    async def run(self, query: str, n_results: int = 5) -> GuidelineRetrievalResult:
        """
        Retrieve guidelines using the variant's config.

        Returns the same GuidelineRetrievalResult schema as the baseline.
        """
        await self._ensure_initialized()

        # For reranking: fetch more candidates then prune
        fetch_k = n_results * 3 if self.variant.rerank else n_results
        fetch_k = min(fetch_k, self._collection.count() or 1)

        results = self._collection.query(
            query_texts=[query],
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"],
        )

        if not results or not results["documents"] or not results["documents"][0]:
            return GuidelineRetrievalResult(query=query, excerpts=[])

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        # Optional rerank
        if self._reranker and self.variant.rerank:
            pairs = [(query, doc) for doc in docs]
            scores = self._reranker.predict(pairs)
            ranked = sorted(
                zip(docs, metas, distances, scores),
                key=lambda x: x[3],
                reverse=True,
            )
            docs = [r[0] for r in ranked[:n_results]]
            metas = [r[1] for r in ranked[:n_results]]
            distances = [r[2] for r in ranked[:n_results]]
        else:
            docs = docs[:n_results]
            metas = metas[:n_results]
            distances = distances[:n_results]

        excerpts = [
            GuidelineExcerpt(
                title=m.get("title", "Clinical Guideline"),
                excerpt=doc,
                source=m.get("source", "Unknown"),
                url=m.get("url") or None,
                relevance_score=round(1 - dist, 4),
            )
            for doc, m, dist in zip(docs, metas, distances)
        ]

        return GuidelineRetrievalResult(query=query, excerpts=excerpts)

    @staticmethod
    def _load_guidelines() -> List[dict]:
        """Load the canonical guideline corpus."""
        if GUIDELINES_DATA_PATH.exists():
            with open(GUIDELINES_DATA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        logger.error(f"Guidelines file not found: {GUIDELINES_DATA_PATH}")
        return []
