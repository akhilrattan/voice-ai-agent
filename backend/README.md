# Joi — Voice AI Agent

A real-time voice AI assistant that listens, thinks, and speaks back.
Built in Python. The terminal app keeps its full local voice workflow, while the API can run in a lightweight, text-first deployment mode.

## What it does
- Speak or type to Joi — she replies with voice
- Searches the web in real time for current information
- Answers questions from your PDF documents
- Remembers full conversation context

## Tech stack
| Component | Technology |
|-----------|------------|
| Speech to text | OpenAI Whisper |
| LLM | GPT-4o-mini via OpenRouter |
| Text to speech | Kokoro TTS (local, offline) |
| Web search | Tavily API |
| Document reading | pypdf + keyword chunking |

## Architecture
```
Voice/Text input
      ↓
   Whisper
      ↓
  GPT-4o-mini ←→ Tools (web search, document query)
      ↓
   Kokoro TTS
      ↓
  Audio output
```

## Project structure
```
voice-agent/
├── voice/
│   └── input.py       # mic recording + Whisper transcription
├── llm/
│   └── joi.py        # LLM logic, tool use, conversation history
├── txt_to_speech/
│   └── tts.py         # Kokoro TTS voice output
├── web_search/
│   └── tools.py       # Tavily search + document tools
├── doc/
|   └──documents.py       # PDF loading and chunking
└── main.py            # main loop — voice and text input
```

## Setup
```bash
# Clone the repo
git clone https://github.com/akhilrattan/voice-agent
cd voice-agent

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Add your API keys
cp .env.example .env
# Edit .env with your keys

# Run
python main.py
```

## Deploy the API

The included Dockerfile is for a lightweight hosted API. It installs only the API dependencies, does not download Whisper, and starts with local server-side voice disabled. Text chat, Tavily search, and PDF querying remain available. Provide `OPENROUTER_API_KEY` and `TAVILY_API_KEY` as secrets in your hosting provider. Set `CORS_ORIGINS` to the exact deployed frontend origin (comma-separated if you have more than one).

For a Python service that installs from a requirements file, use `requirements-api.txt` and the start command `uvicorn main_api:app --host 0.0.0.0 --port $PORT` from the `backend` directory. The API listens on the provider's `PORT` when running in Docker.

The `/chat/voice` route and generated audio require the local Whisper and Kokoro packages and models. To enable them in an API deployment, set `ENABLE_LOCAL_VOICE=true` and install the full `requirements.txt` plus the system audio tools in the Dockerfile. This is resource-heavy and usually unsuitable for small free instances. The default hosted mode returns text replies without generating audio.

The terminal app remains unchanged: install the full `requirements.txt`, configure the same API keys in `.env`, and run `python main.py` to use microphone transcription and local Kokoro speech.

The web frontend uploads PDFs directly to `POST /load-document` (up to 25 MB). The backend extracts and holds their text in process memory, so documents need to be loaded again after a server restart.

## Environment variables
```
OPENROUTER_API_KEY=your-key
TAVILY_API_KEY=your-key
ELEVENLABS_API_KEY=your-key  # optional
```

## Usage
```
Press Enter              → speak to Joi
Type a message           → chat with Joi
load /path as label      → load a PDF document
quit                     → exit
```

## What I learned
- How LLMs work — context windows, token limits, system prompts
- Tool use and the two-call agent pattern
- Real-time audio — recording, transcription, synthesis
- Async Python and threading for concurrent audio
- RAG — chunking documents and retrieving relevant context
- Modular code structure — separation of concerns
