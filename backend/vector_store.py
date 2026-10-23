import chromadb
from chromadb.utils import embedding_functions
import os

# ── SETUP ─────────────────────────────────────────────────────────────────────
# Persist to disk so documents survive server restarts
client = chromadb.PersistentClient(path="./chroma_db")

# Use local sentence-transformers model — no API key needed
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Two separate collections — documents and memories
doc_collection = client.get_or_create_collection(
    name="documents",
    embedding_function=embedder
)

memory_collection = client.get_or_create_collection(
    name="memories",
    embedding_function=embedder
)

print("ChromaDB ready.")


# ── DOCUMENT FUNCTIONS ────────────────────────────────────────────────────────
def store_chunks(label, chunks):
    """
    Store document chunks in ChromaDB.
    Deletes existing chunks for that label first — prevents duplicates on reload.
    """
    # Delete existing entries for this label
    existing = doc_collection.get(where={"label": label})
    if existing["ids"]:
        doc_collection.delete(ids=existing["ids"])

    # Add new chunks with metadata
    ids = [f"{label}_{i}" for i in range(len(chunks))]
    metadatas = [{"label": label, "chunk_index": i} for i in range(len(chunks))]

    doc_collection.add(
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )

    print(f"Stored {len(chunks)} chunks for '{label}'")


def query_documents(question, label=None, n_results=3):
    """
    Find most semantically relevant chunks for a question.
    Optionally filter by document label.
    Returns combined string of top results with source labels.
    """
    where = {"label": label} if label else None

    results = doc_collection.query(
        query_texts=[question],
        n_results=min(n_results, doc_collection.count()),
        where=where
    )

    if not results["documents"][0]:
        return "No relevant content found."

    output = ""
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        output += f"[Source: {meta['label']}]\n{doc}\n\n"

    return output.strip()


def get_loaded_labels():
    """Return list of all document labels currently stored."""
    results = doc_collection.get()
    if not results["metadatas"]:
        return []
    labels = list(set(m["label"] for m in results["metadatas"]))
    return labels


# ── MEMORY FUNCTIONS ──────────────────────────────────────────────────────────
def store_memory(memory_text, session_id="default"):
    """Store a memory with a unique ID."""
    import uuid
    memory_collection.add(
        documents=[memory_text],
        ids=[str(uuid.uuid4())],
        metadatas=[{"session_id": session_id}]
    )


def recall_memories(query, n_results=3):
    """Retrieve most relevant memories for the current context."""
    if memory_collection.count() == 0:
        return ""

    results = memory_collection.query(
        query_texts=[query],
        n_results=min(n_results, memory_collection.count())
    )

    if not results["documents"][0]:
        return ""

    return "\n".join(results["documents"][0])