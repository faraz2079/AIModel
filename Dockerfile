FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install ultralytics fastapi uvicorn[standard] python-multipart opencv-python-headless

COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
