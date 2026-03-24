# Recommended Dockerfile content (adjust Python version if needed)
FROM python:3.11-slim-buster 

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
