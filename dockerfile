# syntax=docker/dockerfile:1.6

# ===== Choose your Python 3.9 base =====
# Stable & supported:
  ARG PY_BASE=python:3.9-slim
  # If you MUST pin exactly 3.9.6 and the tag exists, uncomment this and comment the line above:
  # ARG PY_BASE=python:3.9.6-slim-bullseye
  
  FROM ${PY_BASE}
  
  ENV PYTHONDONTWRITEBYTECODE=1 \
      PYTHONUNBUFFERED=1 \
      PIP_NO_CACHE_DIR=1
  
  # ---- System deps (incl. Tesseract + libs OpenCV wheels expect) ----
  RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl \
      tesseract-ocr tesseract-ocr-eng tesseract-ocr-osd \
      libtesseract-dev libleptonica-dev \
      libgl1 libglib2.0-0 \
      && rm -rf /var/lib/apt/lists/*
  
  # Tesseract language data path (Debian/Ubuntu)
  ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
  
  # ---- App setup ----
  WORKDIR /app
  
  # Install Python deps first for better layer caching
  COPY requirements.txt .
  RUN python -m pip install --upgrade pip setuptools wheel \
   && pip install -r requirements.txt

  # spaCy small English model pinned to 3.6.0 (matches spacy==3.6.0)
RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.6.0/en_core_web_sm-3.6.0-py3-none-any.whl

# (optional) validate spaCy installs
RUN python -m spacy validate
  
  # Copy the rest of your app
  COPY . .
  
  # Healthcheck expects your app to serve /health on 8000
  ENV PORT=8000
  EXPOSE 8000
  
  HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${PORT}/health || exit 1
  
  # NOTE: change "backend:app" if your FastAPI module/object differ
  CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
  