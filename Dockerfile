# GPU-enabled TensorFlow Serving
FROM tensorflow/serving:2.14.0-gpu

# Copy the model into the container
COPY models/resnet50 /models/resnet50

ENTRYPOINT ["/usr/bin/tensorflow_model_server"]

CMD ["--port=8500", "--rest_api_port=8501", "--model_name=resnet50", "--model_base_path=/models/resnet50"]
