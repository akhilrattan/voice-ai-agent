from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import shutil
import os
import uuid
import time
import logging
import tempfile
from dotenv import load_dotenv

load_dotenv()

from llm.joi import get_reply_non_streaming, refresh_system_prompt
from web_search.tools import load_document

# ── LOGGING SETUP ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Keep server deployments lightweight by default. main.py still uses the local
# Whisper and Kokoro models for full terminal voice support.
ENABLE_LOCAL_VOICE = os.getenv("ENABLE_LOCAL_VOICE", "false").lower() in {"1", "true", "yes"}
CORS_ORIGINS = [origin.strip() for origin in os.getenv(
    "CORS_ORIGINS", "http://localhost:3000,http://localhost:5173,http://127.0.0.1:5500"
).split(",") if origin.strip()]

# ── APP SETUP ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Aria Voice Agent API",
    description="Real-time voice AI agent — speech, reasoning, and TTS",
    version="1.0.0"
)

# CORS — allows browser to call this API
# Without this, browser requests get blocked
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve audio files as static files
#creates the folder if it doesn't exist. exist_ok=True means don't throw an error if it already exists. Always use this when your code depends on a folder existing.
os.makedirs("audio_responses", exist_ok=True)
#StaticFiles — mounts a folder as a static file server. Anything in audio_responses/ is now accessible at /audio/filename.wav. The browser can fetch it directly with a URL.
app.mount("/audio", StaticFiles(directory="audio_responses"), name="audio")

@app.middleware("http")
async def log_request(request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({duration:.2f}s)")
    return response



class TextInput(BaseModel):
    message: str

@app.get("/")
def root():
    return {"message" : "Joi api is running"}

@app.get("/health")
def health():
    """to check if the app is alive"""
    return {
        "status": "ok",
        "local_voice_enabled": ENABLE_LOCAL_VOICE,
        "whisper": "enabled" if ENABLE_LOCAL_VOICE else "disabled",
        "tts": "enabled" if ENABLE_LOCAL_VOICE else "disabled",
    }


def make_audio(reply, filename):
    """Generate local speech only when explicitly enabled for the API."""
    if not ENABLE_LOCAL_VOICE:
        return None
    from txt_to_speech.tts import generate_audio
    return generate_audio(reply, filename)

@app.post("/chat/text")
def chat_text(input: TextInput):
    """handles text input-no need to transcribe"""
    if not input.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    logger.info(f"Text input: '{input.message[:50]}'")

    try:
        reply = get_reply_non_streaming(input.message)
        filename = f"audio_responses/{uuid.uuid4()}.wav"
 #uuid.uuid4() — generates a unique random ID like a3f8c2d1-.... You use it for filenames so two simultaneous users don't overwrite each other's audio files. Essential for any multi-user system
        audio_path = make_audio(reply, filename)
        logger.info(f"reply generated : '{reply[:50]}'")
        return{
            "user_input": input.message,
            "reply" : reply,
            "audio_url": f"/audio/{os.path.basename(filename)}" if audio_path else None,
            "status": "ok"
        }
    except Exception as e:
        logger.error(f"text chat error : {e}")
        raise HTTPException(status_code = 500, detail = str(e))

#use async def when your function waits on I/O — file uploads, database calls, network requests. Use regular def when it's pure computation with no waiting.
@app.post("/chat/voice")
async def chat_voice(audio : UploadFile = File(...)):
    """handles voice input-- need for transcribing"""
    if not ENABLE_LOCAL_VOICE:
        raise HTTPException(status_code=503, detail="Server-side voice is disabled. Set ENABLE_LOCAL_VOICE=true to load Whisper and Kokoro.")
    from voice.input import transcribe
    if audio.content_type not in ["audio/wav", "audio/wave", "audio/webm", "audio/mp4", "application/octet-stream"]:
        logger.warning(f"Unexpected content type: {audio.content_type} — attempting anyway")
    logger.info(f"voice upload recieved: {audio.filename}({audio.content_type})")
    upload_path = f"audio_responses/uplaod_{uuid.uuid4()}.wav"

    try:
        with open(upload_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        user_text = transcribe(upload_path)
        if not user_text:
            raise HTTPException(status_code=422, detail="Could not transcribe audio — speak clearly and try again")

        logger.info(f"Transcribed: '{user_text}'")

        
        reply = get_reply_non_streaming(user_text)

        response_filename = f"audio_responses/{uuid.uuid4()}.wav"
        audio_path = make_audio(reply, response_filename)
        logger.info(f"Reply: '{reply[:50]}'")

        return{
            "user_input": user_text,
            "reply": reply,
            "audio_url": f"/audio/{os.path.basename(response_filename)}" if audio_path else None,
            "status": "ok"
        }
    except HTTPException:
        raise  # re-raise HTTP exceptions as-is

    except Exception as e:
        logger.error(f"Voice chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Always clean up upload — even if something failed
        if os.path.exists(upload_path):
            os.remove(upload_path)
                
@app.post("/load-document")
async def ld_doc(file: UploadFile = File(...), label: str = Form("")):
    """Upload a PDF from the browser and load its text for document queries."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    contents = await file.read(25 * 1024 * 1024 + 1)
    if len(contents) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="PDF must be 25 MB or smaller")

    document_label = label.strip() or os.path.basename(file.filename)[:-4]
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
            temp_pdf.write(contents)
            temp_path = temp_pdf.name

        logger.info(f"Loading uploaded PDF as {document_label}")
        result = load_document(temp_path, document_label)
        if result.startswith("Failed to read") or result.startswith("Could not extract"):
            raise HTTPException(status_code=422, detail=result)
        refresh_system_prompt()
        return {"message": result, "status": "ok"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document load error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await file.close()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.delete("/audio/cleanup")
def cleanup_audio():
    """delete all the previos audios generated"""
    try:
        list = os.listdir("audio_responses")
        for f in list:
            if f.endswith(".wav"):
                os.remove(f"audio_responses/{f}")
        logger.info(f"cleaned up {len(list)} audio files")
        return{"deleted": len(list), "status": "ok"}
    except Exception as e:
        raise HTTPException(status_code = 500, detail =str(e))
