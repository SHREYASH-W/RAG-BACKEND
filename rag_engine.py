"""
RAG Engine — local ONNX dense search + SQLite FTS5 keyword search.

No in-memory BM25 index — FTS5 queries go directly to ChromaDB's SQLite file.
This keeps startup RAM under 200MB on Render free tier.

Pipeline:
  1. Dense: ChromaDB ONNX (DefaultEmbeddingFunction, all-MiniLM-L6-v2)
  2. FTS5: SQLite full-text search on chroma.sqlite3
  3. Merge via Reciprocal Rank Fusion
  4. Groq LLM generation
"""

import logging
import re
import sqlite3
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from groq import Groq

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
SQLITE_PATH = CHROMA_DIR / "chroma.sqlite3"


# ═══════════════════════════════════════════════════════════════
#  Vector Store
# ═══════════════════════════════════════════════════════════════

class VectorStore:
    def __init__(self):
        # Copy bundled ONNX model to chromadb's hardcoded cache dir
        import shutil
        bundled = CHROMA_DIR.parent / "onnx_models" / "all-MiniLM-L6-v2"
        cache   = Path.home() / ".cache" / "chroma" / "onnx_models" / "all-MiniLM-L6-v2"
        if bundled.exists() and not cache.exists():
            logger.info("Copying bundled ONNX model to cache...")
            shutil.copytree(str(bundled), str(cache))
            logger.info("ONNX model copied")

        self._ef = DefaultEmbeddingFunction()
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )
        count = self.collection.count()
        logger.info("ChromaDB — collection=%s  chunks=%d", COLLECTION_NAME, count)

        # Pre-warm ONNX model
        logger.info("Pre-warming ONNX model...")
        self._ef(["warmup"])
        logger.info("ONNX model ready")

    # ── Dense retrieval ────────────────────────────────────────
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

    # ── FTS5 keyword retrieval ─────────────────────────────────
    def retrieve_fts(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """Full-text search using ChromaDB's built-in SQLite FTS5 index."""
        if not SQLITE_PATH.exists():
            return []
        try:
            # Sanitize query for FTS5 (escape special chars)
            fts_query = re.sub(r'[^\w\s]', ' ', query).strip()
            if not fts_query:
                return []

            conn = sqlite3.connect(str(SQLITE_PATH))
            cur  = conn.cursor()

            # FTS5 match query — returns rowids ranked by relevance
            cur.execute(
                """
                SELECT c.rowid, c.c0, rank
                FROM embedding_fulltext_search f
                JOIN embedding_fulltext_search_content c ON c.rowid = f.rowid
                WHERE embedding_fulltext_search MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, top_k),
            )
            rows = cur.fetchall()

            if not rows:
                conn.close()
                return []

            # Get metadata for matched rows via embedding_id
            hits = []
            for rowid, doc_text, rank in rows:
                # Get the embedding_id for this rowid
                cur.execute(
                    "SELECT embedding_id FROM embeddings WHERE id = ?", (rowid,)
                )
                emb_row = cur.fetchone()
                if not emb_row:
                    continue
                emb_id = emb_row[0]

                # Get metadata for this embedding
                cur.execute(
                    "SELECT key, string_value FROM embedding_metadata WHERE id = "
                    "(SELECT id FROM embeddings WHERE embedding_id = ?)",
                    (emb_id,),
                )
                meta = {r[0]: r[1] for r in cur.fetchall()}
                hits.append({
                    "text":     doc_text,
                    "source":   meta.get("source"),
                    "act_name": meta.get("act_name"),
                    "part":     meta.get("part"),
                    "chapter":  meta.get("chapter"),
                    "article":  meta.get("article"),
                    "section":  meta.get("section"),
                    "page":     meta.get("page"),
                    "fts_rank": float(rank) if rank else 0.0,
                })

            conn.close()
            return hits
        except Exception as e:
            logger.warning("FTS search failed: %s", e)
            return []

    # ── Hybrid RRF ─────────────────────────────────────────────
    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        dense = self.retrieve_dense(query, top_k=top_k)
        fts   = self.retrieve_fts(query, top_k=top_k)

        rrf: dict[str, float] = {}
        chunks: dict[str, dict] = {}

        for rank, hit in enumerate(dense):
            key = hit["text"][:120]
            rrf[key] = rrf.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            chunks[key] = hit

        for rank, hit in enumerate(fts):
            key = hit["text"][:120]
            rrf[key] = rrf.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            if key not in chunks:
                chunks[key] = hit

        results = []
        for key in sorted(rrf, key=lambda k: rrf[k], reverse=True)[:RERANK_TOP_N]:
            h = dict(chunks[key])
            h["rrf_score"] = round(rrf[key], 6)
            results.append(h)

        logger.debug("Hybrid — dense=%d fts=%d merged=%d", len(dense), len(fts), len(results))
        return results

    def stats(self) -> dict:
        n = self.collection.count()
        if n == 0:
            return {"total_chunks": 0, "documents": [], "document_count": 0}
        meta = self.collection.get(limit=n, include=["metadatas"])
        files = sorted({m["source"] for m in meta["metadatas"]})
        return {"total_chunks": n, "document_count": len(files), "documents": files}


# ═══════════════════════════════════════════════════════════════
#  Reranker (no-op, RRF handles ranking)
# ═══════════════════════════════════════════════════════════════

class Reranker:
    def __init__(self):
        logger.info("Reranker ready (RRF)")

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
            return {"answer": "⚠️ GROQ_API_KEY not configured.", "confidence": 0.0, "guardrail_blocked": False}

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
            return {"answer": "I encountered an error. Please try again.", "confidence": 0.0, "guardrail_blocked": False}

        answer     = clean_output(answer)
        confidence = check_grounding(answer, [h["text"] for h in hits])
        return {"answer": answer, "confidence": round(confidence, 2), "guardrail_blocked": False}
