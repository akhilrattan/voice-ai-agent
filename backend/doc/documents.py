import os
from pypdf import PdfReader

def load_pdf(filepath, label=None):
    """
    Extract all text from a PDF file.
    Returns a single string of the full content.
    """
    if not os.path.exists(filepath):
        return None, f"File not found: {filepath}"

    try:
        reader = PdfReader(filepath)
        full_text = ""

        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n" #entire text in one big string

        if not full_text.strip():
            return None, "Could not extract text — PDF may be scanned or image-based."

        return full_text, None

    except Exception as e:
        return None, f"Failed to read PDF: {e}"


def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into overlapping chunks.
    
    Why chunks: LLMs have context limits — you can't send a 50-page PDF.
    Why overlap: so sentences at chunk boundaries aren't split mid-thought.
    
    Returns list of text chunks.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # overlap means we step back slightly

    return chunks


def find_relevant_chunks(query, chunks, max_chunks=3):
    """
    Find chunks most relevant to the query.
    
    Simple keyword matching for now — no vectors needed at this scale.
    Month 2 replaces this with proper vector similarity search.
    
    Returns top matching chunks as a single string.
    """
    query_words = set(query.lower().split())
    scored = []

    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        # score = how many query words appear in this chunk
        score = len(query_words.intersection(chunk_words))
        scored.append((score, chunk))

    # Sort by score, take top chunks
    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [chunk for score, chunk in scored[:max_chunks] if score > 0]

    if not top_chunks:
        return None

    return "\n\n---\n\n".join(top_chunks)