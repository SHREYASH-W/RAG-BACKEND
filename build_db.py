"""
Build ChromaDB vector database from all files in the data directory.
Uses local sentence-transformers — no API keys, no rate limits.
Resume-safe: skips files already ingested by file hash.
"""
import logging
from pathlib import Path
from dotenv import load_dotenv

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from config import CHROMA_DIR, DATA_DIR, COLLECTION_NAME, EMBED_MODEL
from chunker import chunk_pdf, chunk_json, file_hash

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-18s │ %(levelname)-5s │ %(message)s",
)
logger = logging.getLogger(__name__)

BATCH_SIZE = 200  # local embeddings — no rate limit, use large batches


def ingest_file(collection, filepath: Path) -> int:
    md5 = file_hash(filepath)
    existing = collection.get(where={"pdf_hash": md5}, limit=1)
    if existing["ids"]:
        logger.info("SKIP  %s", filepath.name)
        return 0

    logger.info("START %s", filepath.name)
    ext = filepath.suffix.lower()
    if ext == ".pdf":
        chunks = chunk_pdf(filepath)
    elif ext == ".json":
        chunks = chunk_json(filepath)
    else:
        return 0

    if not chunks:
        logger.warning("EMPTY %s", filepath.name)
        return 0

    docs, metas, ids = [], [], []
    for idx, c in enumerate(chunks):
        ids.append(f"{md5}_{idx}")
        docs.append(c["text"])
        meta = {
            "source":   c.get("source", filepath.name),
            "act_name": c.get("act_name", ""),
            "page":     int(c.get("page") or 0),
            "pdf_hash": md5,
        }
        for key in ("part", "chapter", "article", "section"):
            if c.get(key):
                meta[key] = str(c[key])
        metas.append(meta)

    added = 0
    num_batches = (len(ids) + BATCH_SIZE - 1) // BATCH_SIZE
    for b, i in enumerate(range(0, len(ids), BATCH_SIZE), start=1):
        collection.add(
            ids=ids[i:i + BATCH_SIZE],
            documents=docs[i:i + BATCH_SIZE],
            metadatas=metas[i:i + BATCH_SIZE],
        )
        added += len(ids[i:i + BATCH_SIZE])
        logger.info("  batch %d/%d (+%d)", b, num_batches, len(ids[i:i + BATCH_SIZE]))

    logger.info("DONE  %s — %d chunks", filepath.name, added)
    return added


def build_database():
    logger.info("Loading local embedding model: %s", EMBED_MODEL)
    ef = SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL,
        device="cpu",
    )
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Delete existing collection so we start clean with new embedding model
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info("Deleted old collection")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("DB starts with %d chunks", collection.count())

    # Priority: statute PDFs first, then JSON case files
    priority = [
        "constitution_of_india.pdf",
        "the_bharatiya_nagarik_suraksha_sanhita,_2023.pdf",
        "a202345.pdf",
        "A2000-21 (1).pdf",
        "c9fe9c9b6840524844316f74bb1c556c.pdf",
        "ca7ce5c746fa7480804bbdeb6cb704f0.pdf",
        "civictech_constitution.json",
        "indian_constitution_2024_tagged.json",
        "indian_constitution_sharath.json",
    ]
    priority_paths = [DATA_DIR / n for n in priority if (DATA_DIR / n).exists()]
    remaining = sorted(
        [f for f in DATA_DIR.iterdir()
         if f.is_file() and f.name not in priority
         and f.suffix.lower() in (".pdf", ".json")],
        key=lambda f: f.name,
    )

    total_added = 0
    for filepath in priority_paths + remaining:
        total_added += ingest_file(collection, filepath)

    logger.info("=" * 60)
    logger.info("Done. Added %d chunks this run.", total_added)
    logger.info("Total in DB: %d", collection.count())


if __name__ == "__main__":
    build_database()
