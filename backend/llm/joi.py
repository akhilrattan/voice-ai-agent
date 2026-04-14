from dotenv import load_dotenv
import os
from openai import OpenAI
import json
from web_search.tools import (
    search_web, query_document , loaded_documents
)

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
    "search_web":     lambda args: search_web(args["query"]),
    "query_document": lambda args: query_document(args["label"], args["question"]),
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
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
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
def build_system_prompt():
    """Build system prompt dynamically based on what is currently loaded."""

#This is a list comprehension inside `join`. It loops through every item in `loaded_documents` and builds one line per document:

    if loaded_documents:
        doc_lines = "\n".join([
            f"- '{label}' ({len(chunks)} chunks)"               
            for label, chunks in loaded_documents.items()
        ])
        doc_section = f"Documents loaded and ready to query:\n{doc_lines}"
    else:
        doc_section = "No documents loaded yet."

    return f"""
You are Joi, a sharp voice assistant.
Keep every reply under 2 sentences.
Never use bullet points or markdown — you are speaking out loud.
Speak naturally and concisely.

You have two tools:
- search_web: for current events, news, prices, weather
- query_document: for questions about loaded PDF documents

{doc_section}

Rules:
- Use query_document when user asks about a loaded document — use exact label
- Use search_web for anything current or time-sensitive
- Never make up answers — use a tool if unsure
"""

# ── CONVERSATION HISTORY ──────────────────────────────────────────────────────
messages = [{"role": "system", "content": build_system_prompt()}]

def refresh_system_prompt():
    """Update system prompt only if documents are loaded."""
    if loaded_documents:
        messages[0] = {"role": "system", "content": build_system_prompt()}

# ── REPLY FUNCTIONS ───────────────────────────────────────────────────────────
def get_reply_non_streaming(user_text):
    """Get full reply — handles tool use internally."""
    messages.append({"role": "user", "content": user_text})

    try:
        # Call 1 — LLM decides whether to use a tool
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

            print(f"[Joi is using tool: {tool_name}]")

            # Run the right function from TOOL_MAP
            tool_result = TOOL_MAP[tool_name](tool_args)

            # Add tool call and result to history
            messages.append(message)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(tool_result)
            })

            # Call 2 — LLM answers using tool result
            response2 = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=150
            )

            reply = response2.choices[0].message.content
            messages.append({"role": "assistant", "content": reply})

        else:
            # No tool needed — direct reply
            reply = message.content
            messages.append({"role": "assistant", "content": reply})

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

