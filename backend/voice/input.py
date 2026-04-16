import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import time

SAMPLE_RATE = 16000
DURATION = 5
TEMP_FILE = "temp_recording.wav"
SILENCE_THRESHOLD = 500

print("Loading Whisper model...")
model = whisper.load_model("base")  # faster than base on Apple Silicon
print("Voice module ready.")

def record_audio(duration=DURATION):
    """Record audio from mic and save to temp file."""
    print("Recording — speak now!")
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.int16
    )
    sd.wait()
    write(TEMP_FILE, SAMPLE_RATE, audio)
    return audio

 #np.abs(audio) takes the absolute value of every audio sample — audio waves go positive and negative, absolute value makes everything positive. .mean() averages all those values. The result is the average energy level of the recording. Low mean = quiet = probably silence. High mean = loud = speech detected
def is_silent(audio):
    """Check if recording is basically silence."""
    energy = np.abs(audio).mean()
    return energy < SILENCE_THRESHOLD

def transcribe(filepath):
    """Transcribe audio file. Returns text or empty string."""
    """Transcribe an existing audio file. Returns text string."""
    try:
        result = model.transcribe(filepath)
        return result["text"].strip()
    except Exception as e:
        print(f"Transcription failed: {e}")
        return ""
#result dictionary contains "text" (the transcript), "segments" (timestamps for each word), and "language" (detected language

def listen():
    """Record and transcribe in one step. Returns text string."""
    audio = record_audio()

    if is_silent(audio):
        print("No speech detected.")
        return ""

    try:
        text = transcribe(TEMP_FILE)
    except Exception as e:
        print(f"Transcription failed: {e}")
        return ""

    junk_phrases = ["thank you", "thanks for watching", "bye", "you"]
    if len(text) < 3 or text.lower().strip() in junk_phrases:
        print("Transcript too short or junk — try again.")
        return ""

    return text