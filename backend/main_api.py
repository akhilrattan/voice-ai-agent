from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import shutil
import os
import uuid
import time
import logging
from dotenv import load_dotenv

load_dotenv()

from llm.joi import get_reply_non_streaming, refresh_system_prompt
from txt_to_speech.tts import generate_audio
from voice.input import transcribe
from web_search.tools import load_document

# ── LOGGING SETUP ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "audio_responses")

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
    allow_origins=["*"],      # any frontend can call this for now
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve audio files as static files
#creates the folder if it doesn't exist. exist_ok=True means don't throw an error if it already exists. Always use this when your code depends on a folder existing.
os.makedirs(AUDIO_DIR, exist_ok=True)
#StaticFiles — mounts a folder as a static file server. Anything in audio_responses/ is now accessible at /audio/filename.wav. The browser can fetch it directly with a URL.
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

app.middleware("http")
async def log_request(request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logger.info(f"{request.method} {request.url_path} -> {request.status_code} ({duration:2f}s)")
    return response



class TextInput(BaseModel):
    message: str

class LoadDocInput(BaseModel):
    filepath: str
    label: str

@app.get("/")
def root():
    return {"message" : "Joi api is running"}

@app.get("/health")
def health():
    """Railway pings this to verify app is alive."""
    return {
        "status": "ok",
        "whisper": "loaded",
        "tts": "loaded",
        "version": "1.0.0"
    }

@app.post("/chat/text")
def chat_text(input: TextInput):
    """handles text input-no need to transcribe"""
    if not input.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    logger.info(f"Text input: '{input.message[:50]}'")

    try:
        reply = get_reply_non_streaming(input.message)
        filename = os.path.join(AUDIO_DIR, f"{uuid.uuid4()}.wav")
 #uuid.uuid4() — generates a unique random ID like a3f8c2d1-.... You use it for filenames so two simultaneous users don't overwrite each other's audio files. Essential for any multi-user system
        audio_path = generate_audio(reply, filename)
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
    if audio.content_type not in ["audio/wav", "audio/wave", "audio/webm", "audio/mp4", "application/octet-stream"]:
        logger.warning(f"Unexpected content type: {audio.content_type} — attempting anyway")
    logger.info(f"voice upload recieved: {audio.filename}({audio.content_type})")
    upload_path = os.path.join(AUDIO_DIR, f"upload_{uuid.uuid4()}.wav")

    try:
        with open(upload_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        user_text = transcribe(upload_path)
        if not user_text:
            raise HTTPException(status_code=422, detail="Could not transcribe audio — speak clearly and try again")

        logger.info(f"Transcribed: '{user_text}'")

        
        reply = get_reply_non_streaming(user_text)

        response_filename = os.path.join(AUDIO_DIR, f"{uuid.uuid4()}.wav")
        audio_path = generate_audio(reply, response_filename)
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
async def ld_doc(input : LoadDocInput):
    """load the pdf for rag querying"""
    if not os.path.exists(input.filepath):
        raise HTTPException(status_code=404, detail=f"file not found {input.filepath}")
    if not input.filepath.endswith("pdf"):
        raise HTTPException(statusa_code = 404 , detail="only pdf files are supported")
    logger.info(f"loading document{input.label} from {input.filepath}")

    try:
        result = load_document(input.filepath,input.label)
        refresh_system_prompt()
        return {"message": result, "status": "ok"}
    except Exception as e:
        logger.error(f"Document load error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/audio/cleanup")
def cleanup_audio():
    """delete all the previos audios generated"""
    try:
        files = os.listdir(AUDIO_DIR)
        for f in files:
            if f.endswith(".wav"):
                os.remove(os.path.join(AUDIO_DIR, f))
        logger.info(f"cleaned up {len(files)} audio files")
        return{"deleted": len(files), "status": "ok"}
    except Exception as e:
        raise HTTPException(status_code = 500, detail =str(e))
