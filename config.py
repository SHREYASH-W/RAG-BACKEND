"""
Configuration for the Indian Law RAG backend.
Fully local ONNX-based embeddings via fastembed — no PyTorch, ~50MB RAM.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"
DATA_DIR   = BASE_DIR / "data"

# ── Local Embedding Model (fastembed / ONNX) ───────────────────
# BAAI/bge-small-en-v1.5: ~25MB, 384-dim, excellent quality, ONNX
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# ── Local Cross-Encoder Reranker (fastembed / ONNX) ────────────
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── ChromaDB ───────────────────────────────────────────────────
COLLECTION_NAME = "indian_law"

# ── Chunking (build_db.py only) ────────────────────────────────
CHUNK_SIZE    = 350
CHUNK_OVERLAP = 75

# ── Retrieval ──────────────────────────────────────────────────
TOP_K        = 15
RERANK_TOP_N = 5

# ── LLM ────────────────────────────────────────────────────────
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL      = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS  = 2000

# ── Conversation Memory ────────────────────────────────────────
MAX_CHAT_HISTORY = 10
