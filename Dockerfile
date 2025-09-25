FROM ubuntu:latest
LABEL authors="lucas"

# Use the exact NGC PyTorch image you pulled
FROM nvcr.io/nvidia/pytorch:25.08-py3

ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace

# install your deps first to leverage layer caching
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /tmp/requirements.txt

# We DO NOT copy your code here for dev; we'll mount it at runtime.
# (This way you rebuild only when requirements change.)
#docker run --gpus all -it --rm -v C:/Users/lucas/workspace/HRM:/workspace -w /workspace hrm_nvidia_dk:25.0 python your_script.py