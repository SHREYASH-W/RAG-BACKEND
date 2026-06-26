"""
Configuration for the Indian Law RAG backend.
Fully local embeddings — no external API for vector search.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"
DATA_DIR   = BASE_DIR / "data"

# ── Local Embedding Model (sentence-transformers) ──────────────
# all-MiniLM-L6-v2: 80 MB, 384-dim, fast, good semantic quality
EMBED_MODEL  = "all-MiniLM-L6-v2"

# ── Local Cross-Encoder Reranker ───────────────────────────────
# cross-encoder/ms-marco-MiniLM-L-6-v2: ~80 MB, strong reranking
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── ChromaDB ───────────────────────────────────────────────────
COLLECTION_NAME = "indian_law"

# ── Chunking (build_db.py only) ────────────────────────────────
CHUNK_SIZE    = 350   # words per chunk
CHUNK_OVERLAP = 75    # overlapping words between chunks

# ── Retrieval ──────────────────────────────────────────────────
TOP_K       = 15   # initial dense + BM25 retrieval count
RERANK_TOP_N = 5   # final count after cross-encoder reranking

# ── LLM ────────────────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL     = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS  = 2000

# ── Conversation Memory ────────────────────────────────────────
MAX_CHAT_HISTORY = 10
