import numpy as np
import soundfile as sf
import os
from kokoro import KPipeline

print("Loading Kokoro TTS model...")
pipeline = KPipeline(lang_code='a')
print("TTS ready.")

def generate_audio(text, output_path="response_audio.wav"):
    """
    Generate audio and save to file — for API use.
    No speaker playback — works in Docker with no audio hardware.
    Returns file path on success, None on failure.
    """
    if not text:
        return None

    try:
        all_audio = []
        for _, _, audio in pipeline(text, voice='af_heart', speed=1.0):
            all_audio.append(np.array(audio))

        if not all_audio:
            return None

        full_audio = np.concatenate(all_audio)
        sf.write(output_path, full_audio, samplerate=24000)
        return output_path

    except Exception as e:
        print(f"TTS generation error: {e}")
        return None


def speak(text):
    """
    Play audio locally — terminal use only.
    Imports sounddevice only when called — not at module level.
    Fails gracefully in Docker where there's no audio hardware.
    """
    if not text:
        return
    try:
        import sounddevice as sd
        import threading
        import queue

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
        print(f"TTS playback error (expected in Docker): {e}")
        print(f"Aria (text only): {text}")