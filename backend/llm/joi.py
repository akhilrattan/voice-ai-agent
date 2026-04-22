from dotenv import load_dotenv
import os
from openai import OpenAI
import json
from web_search.tools import (
    search_web, query_document, get_loaded_documents
)
from memory import get_all_facts, extract_and_save_facts, load_facts, trim_messages
from vector_store import recall_memories

load_dotenv()  #loads .env file

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

MODEL="openai/gpt-4o-mini"

# Define the search tool for the LLM the description is what llm sees and it should be apt and concise


# ── TOOL MAP ──────────────────────────────────────────────────────────────────
# A dictionary that maps tool name strings to actual Python functions. When the LLM says "call search_web with query=Bitcoin price",

TOOL_MAP = {
    "search_web": lambda args: search_web(args["query"]),
    "query_document": lambda args: query_document(args["label"], args["question"])
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information, news, prices or recent events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_document",
            "description": "Answer questions from a loaded PDF document using its label.",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "The label of the loaded document e.g. 'my resume'"
                    },
                    "question": {
                        "type": "string",
                        "description": "The question to answer from the document"
                    }
                },
                "required": ["label", "question"]
            }
        }
    },
]

# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
def build_system_prompt(user_query=""):
    """Build system prompt with loaded documents and memories."""

    # Long term facts
    facts = get_all_facts()
    facts_section = f"What I know about you:\n{facts}" if facts else ""

    # Semantic memory — relevant to current query
    if user_query:
        relevant_memories = recall_memories(user_query)
        memory_section = f"Relevant context from memory:\n{relevant_memories}" if relevant_memories else ""
    else:
        memory_section = ""

    # Loaded documents
    loaded = get_loaded_documents()
    if loaded:
        doc_lines = "\n".join([f"- '{label}'" for label in loaded])
        doc_section = f"Documents loaded:\n{doc_lines}"
    else:
        doc_section = "No documents loaded yet."

    return f"""
You are Aria, a sharp voice assistant with persistent memory.
Keep every reply under 2 sentences.
Never use bullet points or markdown — you are speaking out loud.
Speak naturally and concisely.

{facts_section}

{memory_section}

{doc_section}

You have two tools:
- search_web: for current events, news, prices, weather
- query_document: for questions about loaded PDF documents

Rules:
- Use query_document for loaded document questions
- Use search_web for current or time-sensitive questions
- Never make up answers — use a tool if unsure
- Use memory context to personalise replies when relevant
"""

messages = [{"role": "system", "content": build_system_prompt()}]

def refresh_system_prompt(user_query=""):
    """Rebuild system prompt with latest memory and documents."""
    if get_loaded_documents() or load_facts():
        messages[0] = {"role": "system", "content": build_system_prompt(user_query)}


# ── REPLY FUNCTIONS ───────────────────────────────────────────────────────────
def get_reply_non_streaming(user_text):
    """Get full reply — handles tool use and memory extraction."""
    # Refresh prompt with memories relevant to this query
    refresh_system_prompt(user_text)

    messages.append({"role": "user", "content": user_text})

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=150,
            tools=TOOLS,
            tool_choice="auto"
        )

        message = response.choices[0].message

        if message.tool_calls:
            tool_call = message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            print(f"[Aria is using tool: {tool_name}]")

            tool_result = TOOL_MAP[tool_name](tool_args)

            messages.append(message)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(tool_result)
            })

            response2 = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=150
            )

            reply = response2.choices[0].message.content
            messages.append({"role": "assistant", "content": reply})

        else:
            reply = message.content
            messages.append({"role": "assistant", "content": reply})

        # Extract and save memorable facts in background
        recent = f"User: {user_text}\nAria: {reply}"
        extract_and_save_facts(recent, client, MODEL)
        #modifies the list in place — important because messages is a module-level list referenced elsewhere. Reassigning with messages = trim_messages(...) would create a new local variable and break the reference.
        messages[:] = trim_messages(messages, client, MODEL)

        return reply

    except Exception as e:
        messages.pop()
        print(f"LLM error: {e}")
        return "Sorry, I ran into an issue. Please try again."


def get_reply_streaming(user_text):
    """Yield reply sentence by sentence for TTS."""
    reply = get_reply_non_streaming(user_text)
    sentences = reply.replace("!", ".").replace("?", ".").split(".")
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            yield sentence


def get_history():
    """Return full conversation history."""
    return messages
