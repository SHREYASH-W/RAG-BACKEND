"""
Chunker — sentence-aware, legal-structure-preserving document chunking.

Produces overlapping chunks that respect sentence boundaries and
preserve hierarchical legal structure (Part → Chapter → Article → Section).
"""

import re
import hashlib
import json
import logging
from pathlib import Path

import pdfplumber

from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Act Name Mapping
# ═══════════════════════════════════════════════════════════════

ACT_MAP: dict[str, str] = {
    "constitution_of_india.pdf": "Constitution of India",
    "the_bharatiya_nagarik_suraksha_sanhita,_2023.pdf":
        "Bharatiya Nagarik Suraksha Sanhita, 2023",
    "the_bharatiya_nagarik_suraksha_sanhita_2023.pdf":
        "Bharatiya Nagarik Suraksha Sanhita, 2023",
    "a202345.pdf": "Bharatiya Nyaya Sanhita, 2023",
    "a2000-21 (1).pdf": "Information Technology Act, 2000",
}


def get_act_name(filename: str) -> str:
    """Map a filename to a human-readable act name."""
    return ACT_MAP.get(
        filename.lower(),
        Path(filename).stem.replace("_", " ").title(),
    )


# ═══════════════════════════════════════════════════════════════
#  Legal Structure Detection
# ═══════════════════════════════════════════════════════════════

_PART_RE = re.compile(r"PART\s+([IVXLCDM]+)", re.IGNORECASE)
_CHAPTER_RE = re.compile(r"CHAPTER\s+([IVXLCDM]+)", re.IGNORECASE)
_ARTICLE_RE = re.compile(r"Article\s+(\d+[A-Z]?)", re.IGNORECASE)
_SECTION_RE = re.compile(r"Section\s+(\d+[A-Z]?)", re.IGNORECASE)

# Sentence boundary — split on period/question mark/exclamation
# followed by whitespace and an uppercase letter, or on newlines.
_SENTENCE_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z\(])"
    r"|(?<=\n)\s*(?=\S)"
)


def _extract_hierarchy(text: str) -> dict[str, str]:
    """Extract Part/Chapter/Article/Section metadata from text."""
    meta: dict[str, str] = {}

    m = _PART_RE.search(text)
    if m:
        meta["part"] = f"Part {m.group(1)}"

    m = _CHAPTER_RE.search(text)
    if m:
        meta["chapter"] = f"Chapter {m.group(1)}"

    m = _ARTICLE_RE.search(text)
    if m:
        meta["article"] = m.group(1)

    m = _SECTION_RE.search(text)
    if m:
        meta["section"] = m.group(1)

    return meta


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping each sentence intact."""
    sentences = _SENTENCE_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


# ═══════════════════════════════════════════════════════════════
#  Sentence-Aware Chunking
# ═══════════════════════════════════════════════════════════════

def chunk_text(
    text: str,
    source: str,
    page: int = 0,
    act_name: str = "",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split text into overlapping chunks that respect sentence boundaries.

    Each chunk contains:
    - text: the chunk content
    - source: filename
    - page: page number (for PDFs)
    - act_name: human-readable act name
    - part/chapter/article/section: legal hierarchy if detected
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[dict] = []
    running_meta: dict[str, str] = {}
    i = 0

    while i < len(sentences):
        # Build a chunk by accumulating sentences up to chunk_size words
        chunk_sentences: list[str] = []
        word_count = 0

        j = i
        while j < len(sentences) and word_count < chunk_size:
            sent = sentences[j]
            sent_words = len(sent.split())
            # Don't exceed chunk_size by too much (allow 20% overflow to
            # avoid cutting a sentence)
            if word_count + sent_words > chunk_size * 1.2 and chunk_sentences:
                break
            chunk_sentences.append(sent)
            word_count += sent_words
            j += 1

        if not chunk_sentences:
            i += 1
            continue

        chunk_text_str = " ".join(chunk_sentences)

        # Update running hierarchy metadata
        current_meta = _extract_hierarchy(chunk_text_str)
        running_meta.update(current_meta)

        chunk_data = {
            "text": chunk_text_str,
            "source": source,
            "page": page,
            "act_name": act_name,
        }
        chunk_data.update(running_meta)
        chunks.append(chunk_data)

        # Advance by (chunk_size - overlap) words, but snap to sentence
        # boundary — find the sentence that crosses the overlap threshold
        target_advance = chunk_size - chunk_overlap
        advance_words = 0
        advance_sents = 0
        for k, sent in enumerate(chunk_sentences):
            advance_words += len(sent.split())
            advance_sents = k + 1
            if advance_words >= target_advance:
                break

        # Advance at least 1 sentence to avoid infinite loops
        i += max(1, advance_sents)

    return chunks


# ═══════════════════════════════════════════════════════════════
#  PDF Extraction
# ═══════════════════════════════════════════════════════════════

def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    """Extract text from every page of a PDF."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                pages.append({"page": page_num, "text": text.strip()})
    return pages


def extract_json_entries(json_path: Path) -> list[dict]:
    """
    Extract text entries from a JSON file.
    Handles list-of-dicts (instruction/output format) and plain lists.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries: list[dict] = []

    if isinstance(data, list):
        for i, item in enumerate(data, start=1):
            if isinstance(item, dict):
                # Instruction/output format (e.g., Indian Constitution tagged)
                text_parts = []
                if "instruction" in item:
                    text_parts.append(item["instruction"])
                if "output" in item:
                    text_parts.append(item["output"])
                if not text_parts:
                    text_parts.append(
                        item.get("text", item.get("content", json.dumps(item)))
                    )
                entries.append({"page": i, "text": " ".join(text_parts).strip()})
            elif isinstance(item, str):
                entries.append({"page": i, "text": item.strip()})
            else:
                entries.append({"page": i, "text": str(item).strip()})
    elif isinstance(data, dict):
        entries.append({"page": 1, "text": json.dumps(data, indent=2)})
    else:
        entries.append({"page": 1, "text": str(data)})

    return [e for e in entries if e["text"]]


# ═══════════════════════════════════════════════════════════════
#  Full Document Chunking Pipeline
# ═══════════════════════════════════════════════════════════════

def chunk_pdf(pdf_path: Path) -> list[dict]:
    """Extract and chunk an entire PDF."""
    pages = extract_pdf_pages(pdf_path)
    act_name = get_act_name(pdf_path.name)

    all_chunks: list[dict] = []
    for page in pages:
        chunks = chunk_text(
            text=page["text"],
            source=pdf_path.name,
            page=page["page"],
            act_name=act_name,
        )
        all_chunks.extend(chunks)

    logger.info("Chunked PDF %s → %d chunks", pdf_path.name, len(all_chunks))
    return all_chunks


def chunk_json(json_path: Path) -> list[dict]:
    """Extract and chunk a JSON file."""
    entries = extract_json_entries(json_path)
    act_name = get_act_name(json_path.name)

    all_chunks: list[dict] = []
    for entry in entries:
        chunks = chunk_text(
            text=entry["text"],
            source=json_path.name,
            page=entry["page"],
            act_name=act_name,
        )
        all_chunks.extend(chunks)

    logger.info("Chunked JSON %s → %d chunks", json_path.name, len(all_chunks))
    return all_chunks


def file_hash(path: Path) -> str:
    """Compute MD5 hash for dedup."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()
