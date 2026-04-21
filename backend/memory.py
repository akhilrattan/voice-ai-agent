import json
import os
from vector_store import store_memory, recall_memories

MEMORY_FILE = "long_term_memory.json"

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