"""
FastAPI backend for the Indian Law RAG system.
Static (query-only) deployment — optimized for Render free tier (512 MB).
"""
import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import MAX_CHAT_HISTORY
from rag_engine import VectorStore, Reranker, RAGPipeline

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-18s │ %(levelname)-5s │ %(message)s",
)
logger = logging.getLogger("indian_law_api")

# Suppress noisy ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"
os.environ["OTEL_SDK_DISABLED"] = "true"

# ── Globals (populated at startup) ─────────────────────────────
vector_store: VectorStore | None = None
reranker: Reranker | None = None
pipeline: RAGPipeline | None = None


# ── Lifespan ───────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store, reranker, pipeline
    logger.info("🚀 Loading ChromaDB and connecting to APIs…")
    vector_store = VectorStore()
    reranker = Reranker()
    pipeline = RAGPipeline(vector_store, reranker)
    logger.info("✅ All systems ready")
    yield
    logger.info("🛑 Shutting down")


# ── App ────────────────────────────────────────────────────────
app = FastAPI(
    title="Nyaya AI — Indian Law RAG API",
    description="Retrieval-Augmented Generation for Indian legal documents",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global Error Handler ──────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal error occurred. Please try again.",
        },
    )


# ── Request / Response Models ──────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str
    chat_history: list[ChatMessage] = []


class AskResponse(BaseModel):
    answer: str
    confidence: float = 0.0


class StatsResponse(BaseModel):
    total_chunks: int
    document_count: int
    documents: list[str]


class HealthResponse(BaseModel):
    status: str
    chromadb_chunks: int
    version: str = "2.1.0"


# ── Endpoints ──────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    count = vector_store.collection.count() if vector_store else 0
    return HealthResponse(status="ok", chromadb_chunks=count)


@app.post("/api/ask", response_model=AskResponse)
async def ask_question(req: AskRequest):
    """Ask a question about Indian law."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    logger.info("❓ Question: %s", req.question[:100])

    # Convert chat history to plain dicts, limit to MAX_CHAT_HISTORY
    history = [
        {"role": msg.role, "content": msg.content}
        for msg in req.chat_history[-MAX_CHAT_HISTORY:]
    ] if req.chat_history else None

    result = pipeline.ask(req.question, chat_history=history)

    return AskResponse(
        answer=result["answer"],
        confidence=result.get("confidence", 0.0),
    )


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get knowledge base statistics."""
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not ready")
    stats = vector_store.stats()
    return StatsResponse(**stats)


# ── Run ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
