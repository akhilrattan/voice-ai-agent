import os
from voice.input import listen
from llm.joi import get_reply_streaming, refresh_system_prompt
from txt_to_speech.tts import speak
from web_search.tools import load_document

print("/Joi is ready.")
print("  Press Enter with no text     → voice input")
print("  Type a message               → chat with Joi")
print("  Type 'load /path as label'   → load a PDF")
print("  Type 'quit'                  → exit\n")

while True:
    try:
        user_input = input("[ Enter to speak | or type ]: ").strip()

        # ── QUIT ───────────────────────────────────────────────────────────
        if user_input.lower() == "quit":
            print("Joi: Goodbye!")
            break

        # ── LOAD FILE ──────────────────────────────────────────────────────
        elif user_input.lower().startswith("load ") and " as " in user_input:
            parts = user_input[5:].split(" as ")
            filepath = parts[0].strip()
            label = parts[1].strip()
            result = load_document(filepath, label)
            refresh_system_prompt()
            print(f"{result}\n")

        # ── VOICE INPUT ─────────────────────────────────────────────────────
        elif user_input == "":
            user_text = listen()

            if not user_text:
                print("Didn't catch that. Try again.\n")
                continue

            print(f"You said: {user_text}")
            print("Joi: ", end="", flush=True)

            for sentence in get_reply_streaming(user_text):
                print(sentence, end=" ", flush=True)
                speak(sentence)

            print("\n")

        # ── TEXT INPUT ──────────────────────────────────────────────────────
        else:
            print("Joi: ", end="", flush=True)

            for sentence in get_reply_streaming(user_input):
                print(sentence, end=" ", flush=True)
                speak(sentence)

            print("\n")

    except KeyboardInterrupt:
        print("/Joi: Goodbye!")
        break
    except Exception as e:
        print(f"Error: {e}\n")