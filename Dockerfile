FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# ---------------------------
# System dependencies
# ---------------------------
# libgl1 + libglib2.0-0 are REQUIRED for opencv / ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------
# Python dependencies
# ---------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------
# Application code
# ---------------------------
COPY app/ /app/

# ---------------------------
# Runtime config
# ---------------------------
ENV WORKERS=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# ---------------------------
# Start FastAPI (GPU-safe)
# ---------------------------
CMD ["gunicorn", "main:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "-b", "0.0.0.0:8000", \
     "-w", "1"]
