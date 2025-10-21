FROM tensorflow/serving:2.14.0
COPY models/resnet50 /models/resnet50
CMD ["--rest_api_port=8501","--model_name=resnet50","--model_base_path=/models/resnet50"]
