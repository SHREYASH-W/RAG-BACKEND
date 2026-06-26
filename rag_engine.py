"""
RAG Engine — fully local, no external embedding/reranking APIs.

Pipeline:
  1. Hybrid retrieval: BM25 (keyword) + sentence-transformers (dense)
     merged via Reciprocal Rank Fusion
  2. Cross-encoder reranking (local, ms-marco-MiniLM-L-6-v2)
  3. Groq LLM generation (only external call)
"""

import logging
import re

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import CrossEncoder
from groq import Groq
from rank_bm25 import BM25Okapi

from config import (
    CHROMA_DIR, EMBED_MODEL, COLLECTION_NAME,
    TOP_K, RERANK_TOP_N, RERANK_MODEL,
    GROQ_API_KEY, GROQ_MODEL,
    LLM_TEMPERATURE, LLM_MAX_TOKENS,
)
from prompts import SYSTEM_PROMPT, build_answer_prompt
from guardrails import check_input, clean_output, check_grounding

logger = logging.getLogger(__name__)

RRF_K = 60  # Reciprocal Rank Fusion constant


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split for BM25."""
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


# ═══════════════════════════════════════════════════════════════
#  Vector Store
# ═══════════════════════════════════════════════════════════════

class VectorStore:
    """ChromaDB collection with local sentence-transformer embeddings."""

    def __init__(self):
        ef = SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL,
            device="cpu",
        )
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        count = self.collection.count()
        logger.info("ChromaDB — collection=%s  chunks=%d", COLLECTION_NAME, count)

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
        tokenized = [_tokenize(c["text"]) for c in self._all_chunks]
        self._bm25 = BM25Okapi(tokenized)
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
        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "text":     doc,
                "source":   meta.get("source"),
                "act_name": meta.get("act_name"),
                "part":     meta.get("part"),
                "chapter":  meta.get("chapter"),
                "article":  meta.get("article"),
                "section":  meta.get("section"),
                "page":     meta.get("page"),
                "score":    round(1 - dist, 4),
            })
        return hits

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

    def retrieve_hybrid(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """BM25 + dense → Reciprocal Rank Fusion."""
        dense_hits = self.retrieve_dense(query, top_k=top_k)
        bm25_hits  = self.retrieve_bm25(query, top_k=top_k)

        rrf_scores: dict[str, float] = {}
        chunk_map:  dict[str, dict]  = {}

        for rank, hit in enumerate(dense_hits):
            key = hit["text"][:120]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            chunk_map[key] = hit

        for rank, hit in enumerate(bm25_hits):
            key = hit["text"][:120]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            if key not in chunk_map:
                chunk_map[key] = hit

        sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
        results = []
        for key in sorted_keys[:top_k]:
            hit = dict(chunk_map[key])
            hit["rrf_score"] = round(rrf_scores[key], 6)
            results.append(hit)

        logger.debug("Hybrid — dense=%d bm25=%d merged=%d", len(dense_hits), len(bm25_hits), len(results))
        return results

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        return self.retrieve_hybrid(query, top_k=top_k)

    def stats(self) -> dict:
        n = self.collection.count()
        if n == 0:
            return {"total_chunks": 0, "documents": [], "document_count": 0}
        meta = self.collection.get(limit=n, include=["metadatas"])
        files = sorted({m["source"] for m in meta["metadatas"]})
        return {"total_chunks": n, "document_count": len(files), "documents": files}


# ═══════════════════════════════════════════════════════════════
#  Cross-Encoder Reranker (local)
# ═══════════════════════════════════════════════════════════════

class Reranker:
    """Local cross-encoder reranker — no API calls."""

    def __init__(self):
        self.model = CrossEncoder(RERANK_MODEL, device="cpu")
        logger.info("Cross-encoder reranker loaded — %s", RERANK_MODEL)

    def rerank(self, query: str, hits: list[dict],
               top_n: int = RERANK_TOP_N) -> list[dict]:
        if not hits:
            return hits
        try:
            pairs = [(query, h["text"]) for h in hits]
            scores = self.model.predict(pairs)
            ranked = sorted(
                zip(hits, scores), key=lambda x: x[1], reverse=True
            )[:top_n]
            result = []
            for hit, score in ranked:
                h = dict(hit)
                h["rerank_score"] = round(float(score), 4)
                result.append(h)
            return result
        except Exception as e:
            logger.error("Reranking failed: %s", e)
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
    """guard → hybrid retrieve → cross-encoder rerank → Groq generate → clean"""

    def __init__(self, vector_store: VectorStore, reranker: Reranker):
        self.vs      = vector_store
        self.reranker = reranker
        self.groq_client = Groq(api_key=GROQ_API_KEY.strip()) if GROQ_API_KEY else None
        logger.info("RAG pipeline ready — LLM=%s", GROQ_MODEL)

    def ask(self, question: str, chat_history: list[dict] | None = None) -> dict:
        # 1. Input guard
        guard = check_input(question)
        if not guard.passed:
            return {"answer": guard.reason, "confidence": 0.0, "guardrail_blocked": True}

        query = guard.sanitized_input

        # 2. Hybrid retrieval
        hits = self.vs.retrieve(query, top_k=TOP_K)

        # 3. Cross-encoder rerank
        hits = self.reranker.rerank(query, hits, top_n=RERANK_TOP_N)

        # 4. Build context
        context = _build_context(hits)

        # 5. Generate
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

        # 6. Output guard
        answer = clean_output(answer)
        confidence = check_grounding(answer, [h["text"] for h in hits])

        return {
            "answer": answer,
            "confidence": round(confidence, 2),
            "guardrail_blocked": False,
        }
