import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import cohere

from config import CHROMA_DIR, DATA_DIR, COLLECTION_NAME, EMBED_MODEL, COHERE_API_KEY
from chunker import chunk_pdf, chunk_json, file_hash

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(name)-18s │ %(levelname)-5s │ %(message)s")
logger = logging.getLogger(__name__)

def build_database():
    """Builds the ChromaDB vector database from files in the data directory."""
    if not COHERE_API_KEY:
        logger.error("COHERE_API_KEY is not set in .env")
        return

    logger.info("Initializing ChromaDB...")
    ef = embedding_functions.CohereEmbeddingFunction(
        api_key=COHERE_API_KEY,
        model_name=EMBED_MODEL,
    )
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    if not DATA_DIR.exists():
        logger.error(f"Data directory not found: {DATA_DIR}")
        return

    logger.info(f"Scanning {DATA_DIR} for PDF and JSON files...")
    
    total_added = 0
    for filepath in DATA_DIR.iterdir():
        if not filepath.is_file():
            continue
            
        filename = filepath.name.lower()
        if not (filename.endswith('.pdf') or filename.endswith('.json')):
            continue

        md5 = file_hash(filepath)
        existing = collection.get(where={"pdf_hash": md5}, limit=1)
        if existing["ids"]:
            logger.info(f"Skipping {filepath.name} (already ingested)")
            continue

        logger.info(f"Processing {filepath.name}...")
        
        if filename.endswith('.pdf'):
            chunks = chunk_pdf(filepath)
        else:
            chunks = chunk_json(filepath)
            
        if not chunks:
            logger.warning(f"No chunks extracted from {filepath.name}")
            continue

        docs, metas, ids = [], [], []
        for idx, c in enumerate(chunks):
            ids.append(f"{md5}_{idx}")
            docs.append(c["text"])
            meta = {
                "source": c.get("source", filepath.name),
                "act_name": c.get("act_name", ""),
                "page": int(c.get("page", 0)),
                "pdf_hash": md5,
            }
            for key in ("part", "chapter", "article", "section"):
                if c.get(key):
                    meta[key] = str(c[key])
            metas.append(meta)

        import time
        # Batch upsert - small batch to respect trial rate limits
        batch_size = 50
        for i in range(0, len(ids), batch_size):
            try:
                collection.add(
                    ids=ids[i:i + batch_size],
                    documents=docs[i:i + batch_size],
                    metadatas=metas[i:i + batch_size],
                )
                time.sleep(2)  # Delay to prevent rate limits
            except Exception as e:
                logger.error(f"Rate limit or error: {e}")
                time.sleep(10)
                # Retry once
                collection.add(
                    ids=ids[i:i + batch_size],
                    documents=docs[i:i + batch_size],
                    metadatas=metas[i:i + batch_size],
                )
            
        logger.info(f"Added {len(ids)} chunks from {filepath.name}")
        total_added += len(ids)

    logger.info(f"Database build complete. Total chunks added in this run: {total_added}")
    logger.info(f"Total chunks in database: {collection.count()}")

if __name__ == "__main__":
    build_database()
