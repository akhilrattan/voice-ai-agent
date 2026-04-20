import os
from datetime import date
from tavily import TavilyClient
from doc.documents import load_pdf, chunk_text
from vector_store import store_chunks, query_documents, get_loaded_labels

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def search_web(query):
    """Search the web for current information."""
    try:
        today = date.today().strftime("%B %d %Y")
        response = tavily_client.search(
            query=f"{query} {today}",
            search_depth="advanced",
            max_results=3
        )
        results = response.get("results", [])
        if not results:
            return "No results found."
        summary = ""
        for i, r in enumerate(results, 1):
            summary += f"Source {i}: {r['title']}\n{r['content']}\n\n"
        return summary.strip()
    except Exception as e:
        return f"Search failed: {e}"


def load_document(filepath, label=None):
    """Load PDF, chunk it, store in ChromaDB vector store."""
    text, error = load_pdf(filepath)
    if error:
        return error

    chunks = chunk_text(text)
    name = label if label else os.path.basename(filepath)

    # Store in vector store instead of plain dict
    store_chunks(name, chunks)

    return f"'{name}' loaded — {len(chunks)} chunks stored in vector DB."


def query_document(label, question):
    """Semantic search over loaded document chunks."""
    loaded = get_loaded_labels()

    if label not in loaded:
        return f"'{label}' not found. Loaded: {loaded}"

    return query_documents(question, label=label)


def get_loaded_documents():
    """Return labels of all loaded documents."""
    return get_loaded_labels()