"""
PremierDrive School of Motoring - PDF ingestion script.

Reads premierdrive_knowledge_base.pdf, splits it into overlapping text
chunks, embeds each chunk with bge-small-en-v1.5, and upserts everything
into a Qdrant Cloud collection so agent.py can search over it.

Run once (or again whenever the PDF changes):
    python ingest.py
"""

import os
import re
import time
import uuid

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()

QDRANT_URL     = os.environ["QDRANT_URL"].rstrip("/")
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
COLLECTION     = os.environ.get("QDRANT_COLLECTION", "premierdrive_demo")

PDF_PATH         = os.path.join(os.path.dirname(__file__), "premierdrive_knowledge_base.pdf")
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"   # 33M params, ~130MB, 384-dim
CHUNK_SIZE       = 800    # characters per chunk
CHUNK_OVERLAP    = 150    # overlap between consecutive chunks
BATCH_SIZE       = 50     # points per upsert request (avoids ConnectionResetError)

HEADERS = {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}

# Session with retry — same fix that solved the ConnectionResetError
# (Errno 104) we hit sending too many chunks in one request.
session = requests.Session()
retry_cfg = Retry(
    total=5,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["DELETE", "PUT", "POST", "GET"],
    raise_on_status=False,
)
session.mount("https://", HTTPAdapter(max_retries=retry_cfg))


def extract_pages(pdf_path):
    """Returns a list of (page_number, text) tuples, 1-indexed."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages.append((i, text))
    return pages


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Simple sliding-window chunker on characters."""
    chunks = []
    start, length = 0, len(text)
    while start < length:
        end = min(start + size, length)
        chunks.append(text[start:end])
        if end == length:
            break
        start = end - overlap
    return chunks


def build_points(pages):
    """Chunks every page's text into (page, text) dicts (vectors added later)."""
    points = []
    for page_num, text in pages:
        for chunk in chunk_text(text):
            points.append({"id": str(uuid.uuid4()), "page": page_num, "text": chunk})
    return points


def reset_collection(vector_size):
    print(f"Resetting collection '{COLLECTION}' ...")
    session.delete(f"{QDRANT_URL}/collections/{COLLECTION}", headers=HEADERS, timeout=30)
    resp = session.put(
        f"{QDRANT_URL}/collections/{COLLECTION}",
        headers=HEADERS,
        json={"vectors": {"size": vector_size, "distance": "Cosine"}},
        timeout=30,
    )
    resp.raise_for_status()
    print("Collection created.")


def upsert_points(points, vectors):
    total = len(points)
    for i in range(0, total, BATCH_SIZE):
        batch_points  = points[i:i + BATCH_SIZE]
        batch_vectors = vectors[i:i + BATCH_SIZE]
        payload = {
            "points": [
                {"id": p["id"], "vector": v, "payload": {"page": p["page"], "text": p["text"]}}
                for p, v in zip(batch_points, batch_vectors)
            ]
        }
        for attempt in range(3):
            try:
                resp = session.put(
                    f"{QDRANT_URL}/collections/{COLLECTION}/points?wait=true",
                    headers=HEADERS,
                    json=payload,
                    timeout=60,
                )
                resp.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                print(f"  batch {i // BATCH_SIZE + 1}: attempt {attempt + 1} failed ({e}); retrying...")
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"Batch {i // BATCH_SIZE + 1} failed after 3 attempts.")
        print(f"  upserted {min(i + BATCH_SIZE, total)}/{total} chunks")


def main():
    print("Health check...")
    ping = session.get(f"{QDRANT_URL}/healthz", headers=HEADERS, timeout=15)
    print(f"  Qdrant status: {ping.status_code}")

    print(f"Reading PDF: {PDF_PATH}")
    pages = extract_pages(PDF_PATH)
    print(f"  {len(pages)} pages with text extracted")

    points = build_points(pages)
    print(f"  {len(points)} chunks built (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    vector_size = model.get_sentence_embedding_dimension()

    print("Embedding chunks (documents get NO query prefix)...")
    texts = [p["text"] for p in points]
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True).tolist()

    reset_collection(vector_size)

    print("Upserting to Qdrant...")
    upsert_points(points, vectors)

    print(f"\nDone. {len(points)} chunks indexed into '{COLLECTION}'.")


if __name__ == "__main__":
    main()
