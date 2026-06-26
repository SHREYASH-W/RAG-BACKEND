"""
Configuration for the Indian Law RAG backend.
Embeddings: local ONNX (DefaultEmbeddingFunction, all-MiniLM-L6-v2, 384-dim).
Keyword search: SQLite FTS5 (built into ChromaDB's sqlite file, zero extra RAM).
LLM: Groq llama-3.3-70b-versatile.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"
DATA_DIR   = BASE_DIR / "data"

# ── ChromaDB ───────────────────────────────────────────────────
COLLECTION_NAME = "indian_law"

# ── Chunking (build_db.py only) ────────────────────────────────
CHUNK_SIZE    = 350
CHUNK_OVERLAP = 75

# ── Retrieval ──────────────────────────────────────────────────
TOP_K        = 15
RERANK_TOP_N = 5

# ── API Keys ───────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── LLM ────────────────────────────────────────────────────────
GROQ_MODEL      = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS  = 2000

# ── Conversation Memory ────────────────────────────────────────
MAX_CHAT_HISTORY = 10
