"""
RAG Engine — fully local ONNX embeddings, no PyTorch, no reranker download.

Pipeline:
  1. Hybrid retrieval: BM25 (keyword) + DefaultEmbeddingFunction (ONNX dense)
     merged via Reciprocal Rank Fusion — top RERANK_TOP_N returned directly
  2. Groq LLM generation
"""

import logging
import re

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from groq import Groq
from rank_bm25 import BM25Okapi

from config import (
    CHROMA_DIR, COLLECTION_NAME,
    TOP_K, RERANK_TOP_N,
    GROQ_API_KEY, GROQ_MODEL,
    LLM_TEMPERATURE, LLM_MAX_TOKENS,
)
from prompts import SYSTEM_PROMPT, build_answer_prompt
from guardrails import check_input, clean_output, check_grounding

logger = logging.getLogger(__name__)

RRF_K = 60


def _tokenize(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


# ═══════════════════════════════════════════════════════════════
#  Vector Store
# ═══════════════════════════════════════════════════════════════

class VectorStore:
    def __init__(self):
        ef = DefaultEmbeddingFunction()
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        count = self.collection.count()
        logger.info("ChromaDB — collection=%s  chunks=%d", COLLECTION_NAME, count)

        # Pre-warm the ONNX model so the first real query doesn't time out
        # while downloading 79MB
        logger.info("Pre-warming ONNX embedding model...")
        ef(["warmup"])
        logger.info("ONNX model ready")

        self._all_chunks: list[dict] = []
        self._bm25: BM25Okapi | None = None
        if count > 0:
            self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        n = self.collection.count()
        result = self.collection.get(limit=n, include=["documents", "metadatas"])
        self._all_chunks = []
        for doc, meta in zip(result["documents"], result["metadatas"]):
            self._all_chunks.append({
                "text":     doc,
                "source":   meta.get("source"),
                "act_name": meta.get("act_name"),
                "part":     meta.get("part"),
                "chapter":  meta.get("chapter"),
                "article":  meta.get("article"),
                "section":  meta.get("section"),
                "page":     meta.get("page"),
            })
        self._bm25 = BM25Okapi([_tokenize(c["text"]) for c in self._all_chunks])
        logger.info("BM25 index built — %d chunks", n)

    def retrieve_dense(self, query: str, top_k: int = TOP_K) -> list[dict]:
        count = self.collection.count()
        if count == 0:
            return []
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "text":     doc,
                "source":   meta.get("source"),
                "act_name": meta.get("act_name"),
                "part":     meta.get("part"),
                "chapter":  meta.get("chapter"),
                "article":  meta.get("article"),
                "section":  meta.get("section"),
                "page":     meta.get("page"),
                "score":    round(1 - dist, 4),
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def retrieve_bm25(self, query: str, top_k: int = TOP_K) -> list[dict]:
        if not self._bm25 or not self._all_chunks:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        hits = []
        for idx, score in scored:
            if score == 0:
                break
            chunk = dict(self._all_chunks[idx])
            chunk["bm25_score"] = round(float(score), 4)
            hits.append(chunk)
        return hits

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """Hybrid BM25 + dense via Reciprocal Rank Fusion."""
        dense = self.retrieve_dense(query, top_k=top_k)
        bm25  = self.retrieve_bm25(query, top_k=top_k)

        rrf: dict[str, float] = {}
        chunks: dict[str, dict] = {}

        for rank, hit in enumerate(dense):
            key = hit["text"][:120]
            rrf[key] = rrf.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            chunks[key] = hit

        for rank, hit in enumerate(bm25):
            key = hit["text"][:120]
            rrf[key] = rrf.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            if key not in chunks:
                chunks[key] = hit

        results = []
        for key in sorted(rrf, key=lambda k: rrf[k], reverse=True)[:RERANK_TOP_N]:
            h = dict(chunks[key])
            h["rrf_score"] = round(rrf[key], 6)
            results.append(h)
        return results

    def stats(self) -> dict:
        n = self.collection.count()
        if n == 0:
            return {"total_chunks": 0, "documents": [], "document_count": 0}
        meta = self.collection.get(limit=n, include=["metadatas"])
        files = sorted({m["source"] for m in meta["metadatas"]})
        return {"total_chunks": n, "document_count": len(files), "documents": files}


# ═══════════════════════════════════════════════════════════════
#  Reranker — stub, no model download, just returns top_n
# ═══════════════════════════════════════════════════════════════

class Reranker:
    """No-op reranker — RRF in VectorStore already handles ranking."""

    def __init__(self):
        logger.info("Reranker: using RRF (no cross-encoder, saves ~150MB RAM)")

    def rerank(self, query: str, hits: list[dict],
               top_n: int = RERANK_TOP_N) -> list[dict]:
        return hits[:top_n]


# ═══════════════════════════════════════════════════════════════
#  RAG Pipeline
# ═══════════════════════════════════════════════════════════════

def _build_context(hits: list[dict]) -> str:
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
    def __init__(self, vector_store: VectorStore, reranker: Reranker):
        self.vs       = vector_store
        self.reranker = reranker
        self.groq_client = Groq(api_key=GROQ_API_KEY.strip()) if GROQ_API_KEY else None
        logger.info("RAG pipeline ready — LLM=%s", GROQ_MODEL)

    def ask(self, question: str, chat_history: list[dict] | None = None) -> dict:
        guard = check_input(question)
        if not guard.passed:
            return {"answer": guard.reason, "confidence": 0.0, "guardrail_blocked": True}

        query   = guard.sanitized_input
        hits    = self.vs.retrieve(query, top_k=TOP_K)
        hits    = self.reranker.rerank(query, hits, top_n=RERANK_TOP_N)
        context = _build_context(hits)

        if not self.groq_client:
            return {
                "answer": "⚠️ GROQ_API_KEY is not configured.",
                "confidence": 0.0,
                "guardrail_blocked": False,
            }

        user_prompt = build_answer_prompt(query, context, chat_history)
        try:
            response = self.groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
            answer = response.choices[0].message.content
        except Exception as e:
            logger.error("LLM failed: %s", e)
            return {
                "answer": "I encountered an error. Please try again.",
                "confidence": 0.0,
                "guardrail_blocked": False,
            }

        answer     = clean_output(answer)
        confidence = check_grounding(answer, [h["text"] for h in hits])
        return {"answer": answer, "confidence": round(confidence, 2), "guardrail_blocked": False}
