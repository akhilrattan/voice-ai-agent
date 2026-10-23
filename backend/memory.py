import json
import os
from vector_store import store_memory, recall_memories

MEMORY_FILE = "long_term_memory.json"
MAX_MESSAGES = 8


def load_facts():
    """Load persisted facts from disk."""
    if not os.path.exists(MEMORY_FILE):
        return {}
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)

def save_fact(key, value):
    """Save a fact to disk permanently."""
    facts = load_facts()
    facts[key] = value
    with open(MEMORY_FILE, "w") as f:
        json.dump(facts, f, indent=2)
    print(f"Memory saved: {key} = {value}")

def get_all_facts():
    """Return all saved facts as a formatted string."""
    facts = load_facts()
    if not facts:
        return ""
    return "\n".join([f"- {k}: {v}" for k, v in facts.items()])

def extract_and_save_facts(conversation_text, llm_client, model):
    """
    Ask LLM to extract memorable facts from conversation.
    Saves them to disk automatically.
    """
    if not conversation_text.strip():
        return

    prompt = f"""
Extract any important personal facts from this conversation that are worth remembering.
Examples: name, location, job, preferences, ongoing projects.
Return ONLY a JSON object like {{"name": "Akhil", "location": "Shimla"}} or {{}} if nothing memorable.
No explanation, just JSON.

Conversation:
{conversation_text}
"""
    try:
        response = llm_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0
        )
        raw = response.choices[0].message.content.strip()

        # Clean up response — remove markdown if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        facts = json.loads(raw)

        for key, value in facts.items():
            save_fact(key, value)
            store_memory(f"{key}: {value}")

    except Exception as e:
        print(f"Memory extraction failed: {e}")


def summarise_messages(messages, llm_client, model):
    """
    Summarise old messages that are about to be dropped from context.
    Stores summary in vector memory so nothing important is lost.
    Returns the summary string.
    """
    if not messages:
        return ""

    # Build conversation text from messages
    conversation = ""
    for m in messages:
        if m["role"] in ["user", "assistant"]:
            role = "User" if m["role"] == "user" else "Aria"
            content = m.get("content") or ""
            if content:
                conversation += f"{role}: {content}\n"

    if not conversation.strip():
        return ""

    prompt = f"""
Summarise this conversation in 2-3 sentences capturing the key points, 
decisions, and any important information discussed.
Be concise — this summary will be used as context for future conversations.

Conversation:
{conversation}
"""
    try:
        response = llm_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0
        )
        summary = response.choices[0].message.content.strip()

        # Store summary in vector memory
        store_memory(f"Past conversation summary: {summary}")
        print(f"Conversation summarised and stored.")

        return summary

    except Exception as e:
        print(f"Summarisation failed: {e}")
        return ""

def trim_messages(messages, llm_client, model, max_messages=MAX_MESSAGES):
    """
    Keep messages list bounded at max_messages.
    Summarises dropped messages before removing them.
    Always keeps messages[0] — the system prompt.
    Returns trimmed messages list.
    """
    # +1 because messages[0] is always system prompt
    if len(messages) <= max_messages + 1:
        return messages

    # Messages to drop — everything after system prompt up to the overflow
    overflow_count = len(messages) - max_messages - 1
    messages_to_drop = messages[1:overflow_count + 1]

    # Summarise before dropping
    summarise_messages(messages_to_drop, llm_client, model)

    # Keep system prompt + last max_messages messages
    trimmed = [messages[0]] + messages[overflow_count + 1:]

    print(f"Trimmed {overflow_count} messages from history.")
    return trimmed
