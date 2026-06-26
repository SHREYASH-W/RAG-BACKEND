"""
Priority ingestion script — ingests the Constitution and BNSS PDFs first,
then as many JSON files as the remaining API quota allows.
"""
import os
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

from config import CHROMA_DIR, DATA_DIR, COLLECTION_NAME, EMBED_MODEL, COHERE_API_KEY
from chunker import chunk_pdf, chunk_json, file_hash

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-18s │ %(levelname)-5s │ %(message)s",
)
logger = logging.getLogger(__name__)


def ingest_file(collection, filepath: Path) -> int:
    """Ingest a single file. Returns number of chunks added, or 0 if skipped."""
    md5 = file_hash(filepath)
    existing = collection.get(where={"pdf_hash": md5}, limit=1)
    if existing["ids"]:
        logger.info(f"Skipping {filepath.name} (already ingested)")
        return 0

    logger.info(f"Processing {filepath.name}...")
    if filepath.suffix.lower() == ".pdf":
        chunks = chunk_pdf(filepath)
    else:
        chunks = chunk_json(filepath)

    if not chunks:
        logger.warning(f"No chunks from {filepath.name}")
        return 0

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

    batch_size = 25
    total = 0
    for i in range(0, len(ids), batch_size):
        retries = 0
        while retries < 5:
            try:
                collection.add(
                    ids=ids[i:i + batch_size],
                    documents=docs[i:i + batch_size],
                    metadatas=metas[i:i + batch_size],
                )
                total += len(ids[i:i + batch_size])
                logger.info(
                    f"  Batch {i//batch_size + 1}: added {len(ids[i:i+batch_size])} chunks. "
                    f"Sleeping 65s..."
                )
                time.sleep(65)
                break
            except Exception as e:
                retries += 1
                wait = 60 * retries
                logger.warning(f"  Rate limit (attempt {retries}): waiting {wait}s...")
                time.sleep(wait)
        else:
            logger.error(f"  Failed batch {i} after 5 retries. Stopping this file.")
            return total

    logger.info(f"✅ {filepath.name}: {total} chunks added")
    return total


def main():
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
    logger.info(f"DB currently has {collection.count()} chunks")

    # Priority order: Constitution PDF first, then BNSS, then JSON files
    priority_files = [
        DATA_DIR / "constitution_of_india.pdf",
        DATA_DIR / "the_bharatiya_nagarik_suraksha_sanhita,_2023.pdf",
        DATA_DIR / "A2000-21 (1).pdf",
        DATA_DIR / "a202345.pdf",
        DATA_DIR / "c9fe9c9b6840524844316f74bb1c556c.pdf",
        DATA_DIR / "ca7ce5c746fa7480804bbdeb6cb704f0.pdf",
    ]

    # Then all JSON files not yet ingested
    all_files = list(DATA_DIR.iterdir())
    json_files = sorted([f for f in all_files if f.suffix.lower() == ".json"])

    files_to_process = priority_files + json_files

    total_added = 0
    for filepath in files_to_process:
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            continue
        added = ingest_file(collection, filepath)
        total_added += added

    logger.info(f"\n🎉 Done. Added {total_added} chunks this run.")
    logger.info(f"Total in DB: {collection.count()}")


if __name__ == "__main__":
    main()
