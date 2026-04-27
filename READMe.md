# JOI — Voice AI Agent

> A real-time voice AI assistant that listens, thinks, and speaks back.
> Live demo: [your-github-pages-url]

## Demo
[Add your demo video link 

## What it does
- 🎤 Speak or type — Aria responds with voice
- 🔍 Searches the web in real time for current information  
- 📄 Answers questions from your PDF documents
- 🧠 Remembers facts about you across sessions
- ⚡ Runs TTS and speech recognition locally — no voice API costs

## Architecture
Browser (GitHub Pages)
↓  HTTP

↓
┌───────────────────────┐
│ Whisper  — STT        │
│ GPT-4o-mini — LLM     │
│ Kokoro   — TTS        │
│ ChromaDB — Vector RAG │
│ Tavily   — Web search │
└───────────────────────┘

## Tech stack
| Layer | Technology |
|-------|-----------|
| Frontend | HTML, CSS, JavaScript |
| Backend | FastAPI, Python |
| Speech to text | OpenAI Whisper (local) |
| LLM | GPT-4o-mini via OpenRouter |
| Text to speech | Kokoro TTS (local, offline) |
| Vector database | ChromaDB |
| Web search | Tavily API |
| Deployment |  + GitHub Pages |

## Project structure
voice-agent-web/
├── backend/
│   ├── main_api.py        # FastAPI server
│   ├── llm/aria.py        # LLM + tool use + memory
│   ├── voice/input.py     # Whisper transcription
│   ├── txt_to_speech/     # Kokoro TTS
│   ├── web_search/        # Tavily + document tools
│   ├── vector_store.py    # ChromaDB operations
│   ├── memory.py          # Persistent memory
│   ├── documents.py       # PDF processing
│   └── Dockerfile
└── frontend/
└── index.html         # Single-file browser app

## Setup

### Backend
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
uvicorn main_api:app --reload
```

### Frontend
```bash
# Open frontend/index.html in browser
# Or serve with: python3 -m http.server 3000
```

## Environment variables
OPENROUTER_API_KEY=your-key
TAVILY_API_KEY=your-key

## What I learned
- FastAPI and REST API design
- Browser APIs — MediaRecorder, fetch, Web Audio
- Vector embeddings and semantic search
- Persistent agent memory architecture  
- Docker containerisation
- Full stack deployment