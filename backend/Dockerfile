FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    espeak-ng \
    git \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "import whisper; whisper.load_model('base')"

COPY . .

RUN mkdir -p audio_responses chroma_db

EXPOSE 8000

CMD sh -c 'uvicorn main_api:app --host 0.0.0.0 --port ${PORT:-8000}'
