# Joi — Voice AI Agent

A real-time voice AI assistant that listens, thinks, and speaks back.
Built in Python, runs entirely on local machine.

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