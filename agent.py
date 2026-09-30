"""
PremierDrive School of Motoring RAG demo - agent.

Two tools, one LangGraph ReAct agent (langgraph.prebuilt.create_react_agent):
  - search_knowledge_base(query)      -> semantic search over the ingested PDF (Qdrant)
  - lookup_booking(booking_reference) -> mock live pupil booking status (bookings.json)

The model decides which tool(s) a question needs:
  - "How much does an intensive course cost?"            -> search_knowledge_base only
  - "What's the status of my booking PD-3003?"            -> lookup_booking only
  - "My booking is PD-3007 - when's my practical test?"   -> both tools

app.py imports answer() from here - nothing in this file is meant to be
run directly.
"""

import json
import os

import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

load_dotenv()

QDRANT_URL     = os.environ["QDRANT_URL"].rstrip("/")
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
COLLECTION     = os.environ.get("QDRANT_COLLECTION", "premierdrive_demo")
GROQ_API_KEY   = os.environ["GROQ_API_KEY"]
MAIN_MODEL     = os.environ.get("MAIN_MODEL", "openai/gpt-oss-120b")

# bge-small-en-v1.5 (33M params, ~130MB, 384-dim).
# Forced to CPU - letting sentence-transformers auto-detect CUDA on a
# shared / virtual GPU produced NaN vectors in earlier deployments.
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BOOKINGS_PATH    = os.path.join(os.path.dirname(__file__), "bookings.json")
TOP_K            = 4

HEADERS = {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}

# Load once at import time - avoids repeated 1-2 s model loads per request.
_embed_model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")

with open(BOOKINGS_PATH) as f:
    _BOOKINGS = {b["booking_reference"]: b for b in json.load(f)}


@tool
def search_knowledge_base(query: str) -> str:
    """Search PremierDrive School of Motoring's knowledge base -- lesson
    packages, pricing, intensive courses, theory test, hazard perception,
    practical test, manoeuvres, booking policy, cancellation policy,
    payment terms, Pass Plus, motorway lessons, eligibility rules, and FAQs.

    Use this for ANY question about services, pricing, tests, policies, or
    how driving lessons/tests work. Always call this before answering a
    policy, pricing, or driving-test question -- never answer from memory."""

    # bge-small-en-v1.5 requires this prefix on QUERIES only.
    # Documents ingested by ingest.py are NOT prefixed.
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    vector   = _embed_model.encode(prefixed, normalize_embeddings=True).tolist()

    resp = requests.post(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
        headers=HEADERS,
        json={"vector": vector, "limit": TOP_K, "with_payload": True},
        timeout=30,
    )
    resp.raise_for_status()

    hits = resp.json()["result"]
    if not hits:
        return "No matching content found in PremierDrive's knowledge base."

    blocks = [
        f"[page {h['payload']['page']}] {h['payload']['text']}"
        for h in hits
    ]
    return "\n\n".join(blocks)


@tool
def lookup_booking(booking_reference: str) -> str:
    """Look up the live status of a PremierDrive pupil booking by its
    reference number, e.g. "PD-3003". Use this whenever a pupil asks about
    their booking, package hours remaining, next lesson, instructor,
    theory/practical test status, or balance due. Never guess a booking's
    status from the knowledge base -- only lookup_booking contains real
    booking data."""

    ref = booking_reference.strip().upper()
    # Tolerate "3003" as well as "PD-3003"
    if not ref.startswith("PD-"):
        ref = f"PD-{ref}"

    booking = _BOOKINGS.get(ref)
    if not booking:
        return (
            f"No booking found with reference {ref}. "
            "Please ask the pupil to double-check the reference in their "
            "booking confirmation email or the PremierDrive app."
        )
    return json.dumps(booking, indent=2)


SYSTEM_PROMPT = """You are the customer assistant for PremierDrive School of \
Motoring, a DVSA-approved driving school based in Manchester, UK \
(phone: 0161 555 0291, out-of-hours: 07712 900 338).

Answer ONLY from what search_knowledge_base and lookup_booking return -- \
never invent a price, policy, test rule, or booking status.

Rules:
- For any question about lesson prices, packages, intensive courses, theory \
  test, hazard perception, practical test, manoeuvres, Pass Plus, motorway \
  lessons, booking policy, cancellation, payment, or eligibility: call \
  search_knowledge_base first.
- For any question about a specific pupil's booking (hours remaining, next \
  lesson, instructor, test status, balance due): call lookup_booking with \
  the booking reference (PD-XXXX format).
- If the question needs both (e.g. "My booking is PD-3005 -- how many more \
  hours until I'm test-ready?"): call both tools before answering.
- Keep answers short and direct, in plain, friendly customer-support language.
- If nothing relevant is found, say so honestly instead of guessing.
- When quoting a price or policy, mention which section of the knowledge \
  base it came from if the page number is available."""


_model = ChatGroq(model=MAIN_MODEL, api_key=GROQ_API_KEY, temperature=0.1)
try:
    _graph = create_react_agent(_model, tools=[search_knowledge_base, lookup_booking],
                                state_modifier=SYSTEM_PROMPT)
except TypeError:
    # Older langgraph uses 'prompt' instead of 'state_modifier'
    _graph = create_react_agent(_model, tools=[search_knowledge_base, lookup_booking],
                                prompt=SYSTEM_PROMPT)


def answer(message: str, history: list) -> str:
    """history is a list of {"role": "user"/"assistant", "content": str} dicts."""
    messages = []
    for turn in history:
        role    = turn.get("role")
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=message))

    result = _graph.invoke({"messages": messages})

    for msg in reversed(result["messages"]):
        content = getattr(msg, "content", "")
        if isinstance(msg, AIMessage) and content:
            return content
    return "Sorry, I couldn't generate an answer -- please try rephrasing."
