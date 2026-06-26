"""
RAG Engine — Vector store (query-only), reranker, and the full RAG pipeline.

Optimized for static deployment: no ingestion, no BM25, no query expansion.
Uses Cohere for embeddings + reranking, Groq for LLM generation.
"""

import logging
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
import cohere
from groq import Groq

from config import (
    CHROMA_DIR, EMBED_MODEL, COLLECTION_NAME,
    TOP_K, RERANK_TOP_N, RERANK_MODEL,
    GROQ_API_KEY, GROQ_MODEL, COHERE_API_KEY,
    LLM_TEMPERATURE, LLM_MAX_TOKENS,
)
from prompts import SYSTEM_PROMPT, build_answer_prompt
from guardrails import check_input, clean_output, check_grounding

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Vector Store (ChromaDB wrapper — query only)
# ═══════════════════════════════════════════════════════════════

class VectorStore:
    """Manages the ChromaDB collection for Indian law documents (read-only)."""

    def __init__(self):
        ef = embedding_functions.CohereEmbeddingFunction(
            api_key=COHERE_API_KEY,
            model_name=EMBED_MODEL,
        )
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB loaded — collection=%s chunks=%d",
            COLLECTION_NAME,
            self.collection.count(),
        )

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """Retrieve top-K chunks via cosine similarity."""
        count = self.collection.count()
        if count == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "text": doc,
                "source": meta.get("source"),
                "act_name": meta.get("act_name"),
                "part": meta.get("part"),
                "chapter": meta.get("chapter"),
                "article": meta.get("article"),
                "section": meta.get("section"),
                "page": meta.get("page"),
                "score": round(1 - dist, 4),
            })
        return hits

    def stats(self) -> dict:
        """Return knowledge-base statistics."""
        n = self.collection.count()
        if n == 0:
            return {"total_chunks": 0, "documents": [], "document_count": 0}

        meta = self.collection.get(limit=n, include=["metadatas"])
        files = sorted({m["source"] for m in meta["metadatas"]})
        return {
            "total_chunks": n,
            "document_count": len(files),
            "documents": files,
        }


# ═══════════════════════════════════════════════════════════════
#  Reranker
# ═══════════════════════════════════════════════════════════════

class Reranker:
    """Cohere reranker for improving retrieval precision."""

    def __init__(self):
        self.client = cohere.Client(api_key=COHERE_API_KEY)
        logger.info("Reranker loaded — model=%s", RERANK_MODEL)

    def rerank(self, query: str, hits: list[dict],
               top_n: int = RERANK_TOP_N) -> list[dict]:
        if not hits:
            return hits

        docs = [h["text"] for h in hits]

        try:
            results = self.client.rerank(
                model=RERANK_MODEL,
                query=query,
                documents=docs,
                top_n=top_n
            )

            reranked_hits = []
            for r in results.results:
                hit = hits[r.index]
                hit["rerank_score"] = float(r.relevance_score)
                reranked_hits.append(hit)

            return reranked_hits
        except Exception as e:
            logger.error("Reranking failed: %s", e)
            # Fallback if Cohere fails: just return top_n from original hits
            return hits[:top_n]


# ═══════════════════════════════════════════════════════════════
#  RAG Pipeline
# ═══════════════════════════════════════════════════════════════

def _build_context(hits: list[dict]) -> str:
    """Format retrieved chunks into a context block for the LLM."""
    if not hits:
        return "No relevant legal provisions found."

    blocks = []
    for i, h in enumerate(hits, start=1):
        parts = []
        if h.get("act_name"):
            parts.append(h["act_name"])
        if h.get("part"):
            parts.append(h["part"])
        if h.get("article"):
            parts.append(f"Article {h['article']}")
        if h.get("section"):
            parts.append(f"Section {h['section']}")

        header = f"[{' | '.join(parts)}]" if parts else f"[Provision {i}]"
        blocks.append(header + "\n" + h["text"])

    return "\n\n---\n\n".join(blocks)


class RAGPipeline:
    """End-to-end RAG: guard → dense retrieve → rerank → generate → clean."""

    def __init__(self, vector_store: VectorStore, reranker: Reranker):
        self.vs = vector_store
        self.reranker = reranker
        self.groq_client = (
            Groq(api_key=GROQ_API_KEY.strip()) if GROQ_API_KEY else None
        )
        logger.info("RAG pipeline ready — Groq model=%s", GROQ_MODEL)

    def ask(self, question: str,
            chat_history: list[dict] | None = None) -> dict:
        """
        Answer a legal question using the full RAG pipeline.

        Returns:
            dict with 'answer', 'confidence', 'guardrail_blocked'
        """
        # ── 1. Input Guardrail ─────────────────────────────────
        guard = check_input(question)
        if not guard.passed:
            return {
                "answer": guard.reason,
                "confidence": 0.0,
                "guardrail_blocked": True,
            }

        query = guard.sanitized_input

        # ── 2. Dense Retrieval (ChromaDB → Cohere rerank) ─────
        hits = self.vs.retrieve(query, top_k=TOP_K)
        hits = self.reranker.rerank(query, hits, top_n=RERANK_TOP_N)

        # ── 3. Build context ───────────────────────────────────
        context = _build_context(hits)

        # ── 4. Generate answer via Groq ────────────────────────
        if not self.groq_client:
            return {
                "answer": (
                    "⚠️ The AI engine is not configured. "
                    "Please set the GROQ_API_KEY in the backend .env file."
                ),
                "confidence": 0.0,
                "guardrail_blocked": False,
            }

        user_prompt = build_answer_prompt(query, context, chat_history)

        try:
            response = self.groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
            answer = response.choices[0].message.content
        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            return {
                "answer": (
                    "I encountered an error generating a response. "
                    "Please try again in a moment."
                ),
                "confidence": 0.0,
                "guardrail_blocked": False,
            }

        # ── 5. Output Guardrail — clean & check grounding ─────
        answer = clean_output(answer)
        context_texts = [h["text"] for h in hits]
        confidence = check_grounding(answer, context_texts)

        return {
            "answer": answer,
            "confidence": round(confidence, 2),
            "guardrail_blocked": False,
        }
