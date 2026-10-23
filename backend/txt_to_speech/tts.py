import numpy as np
import sounddevice as sd
from kokoro import KPipeline
import threading
import queue
import soundfile as sf
import os

print("Loading Kokoro TTS model...")
pipeline = KPipeline(lang_code='a')
print("TTS ready.")

def speak(text):
    """Generate and play audio locally — for terminal use."""
    if not text:
        return
    try:
        audio_queue = queue.Queue()

        def generate():
            for _, _, audio in pipeline(text, voice='af_heart', speed=1.0):
                audio_queue.put(np.array(audio))
            audio_queue.put(None)

        thread = threading.Thread(target=generate)
        thread.start()

        while True:
            chunk = audio_queue.get()
            if chunk is None:
                break
            sd.play(chunk, samplerate=24000)
            sd.wait()

        thread.join()

    except Exception as e:
        print(f"TTS error: {e}")
        print(f"Joi (text only): {text}")


def generate_audio(text, output_path="response_audio.wav"):
    """
    Generate audio and save to file — for API use.
    Returns the file path on success, None on failure.
    """
    if not text:
        return None

    try:
        all_audio = []

        for _, _, audio in pipeline(text, voice='af_heart', speed=1.0):
            all_audio.append(np.array(audio))

        if not all_audio:
            return None

        # Concatenate all chunks into one audio array
        full_audio = np.concatenate(all_audio)

        # Save to WAV file
        sf.write(output_path, full_audio, samplerate=24000)

        return output_path

    except Exception as e:
        print(f"TTS generation error: {e}")
        return None