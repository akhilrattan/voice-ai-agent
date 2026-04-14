from dotenv import load_dotenv
import os
from tavily import TavilyClient
from doc.documents import load_pdf, chunk_text, find_relevant_chunks
from datetime import date

load_dotenv()  # loads .env file

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Stores loaded PDF chunks — key is label, value is list of chunks
loaded_documents = {}

def search_web(query):
    """Search the web for current information."""
    try:
        today = date.today().strftime("%B %d %Y")
        query_with_date = f"{query} {today}"

        response = tavily_client.search(
            query=query_with_date,
            search_depth="advanced",
            max_results=3
        )

        results = response.get("results", [])
        if not results:
            return "No results found."

        summary = ""
        for i, result in enumerate(results, 1):
            summary += f"Source {i}: {result['title']}\n"
            summary += f"{result['content']}\n\n"

        return summary.strip()

    except Exception as e:
        return f"Search failed: {e}"


def load_document(filepath, label=None):
    """Load a PDF and store chunks under a label."""
    text, error = load_pdf(filepath)

    if error:
        return error

    chunks = chunk_text(text)
    name = label if label else os.path.basename(filepath)
    loaded_documents[name] = chunks

    return f"'{name}' loaded — {len(chunks)} chunks ready."


def query_document(label, question):
    """Answer a question from a loaded document by its label."""
    if label not in loaded_documents:
        available = list(loaded_documents.keys())
        return f"'{label}' not found. Loaded: {available}"

    chunks = loaded_documents[label]
    relevant = find_relevant_chunks(question, chunks)

    if not relevant:
        return "No relevant content found."

    return relevant


