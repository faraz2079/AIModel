FROM python:3.10-slim

WORKDIR /app

RUN pip install requests pillow opencv-python-headless numpy

COPY loadgen_yolo.py /app/loadgen_yolo.py
COPY sample.jpg /app/image.jpg

CMD ["python3", "/app/loadgen_yolo.py"]
