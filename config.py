"""
Configuration for the Indian Law RAG backend.
Optimized for static (query-only) deployment on Render free tier (512 MB RAM).
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent               # repo root
CHROMA_DIR = BASE_DIR / "chroma_db"
DATA_DIR = BASE_DIR / "data"

# ── Embedding & Reranking Models ───────────────────────────────
EMBED_MODEL = "embed-english-v3.0"
RERANK_MODEL = "rerank-english-v3.0"

# ── ChromaDB ───────────────────────────────────────────────────
COLLECTION_NAME = "indian_law"

# ── Chunking (used by build_db.py only) ────────────────────────
CHUNK_SIZE = 350          # words per chunk
CHUNK_OVERLAP = 75        # overlapping words between consecutive chunks

# ── Retrieval ──────────────────────────────────────────────────
TOP_K = 15                # initial retrieval count
RERANK_TOP_N = 5          # final count after reranking

# ── LLM & API Keys ───────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 2000

# ── Conversation Memory ───────────────────────────────────────
MAX_CHAT_HISTORY = 10     # max messages to include in context

# ── Act name mapping ──────────────────────────────────────────
ACT_MAP = {
    "constitution_of_india.pdf": "Constitution of India",
    "the_bharatiya_nagarik_suraksha_sanhita_2023.pdf":
        "Bharatiya Nagarik Suraksha Sanhita, 2023",
}
