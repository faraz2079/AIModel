FROM python:3.10-slim

RUN pip install pillow numpy requests

WORKDIR /app
COPY loadgen_resnet50.py /app/loadgen_resnet50.py
COPY kitten.jpg /app/image.jpg

CMD ["python3", "loadgen_resnet50.py"]
