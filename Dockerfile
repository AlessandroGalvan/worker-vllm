FROM nvidia/cuda:12.1.0-base-ubuntu22.04 

# Installazione Git, Git-LFS e dipendenze di sistema
RUN apt-get update -y \
    && apt-get install -y python3-pip git git-lfs ncdu \
    && git lfs install \
    && git config --global core.compression 0 \
    && git config --global http.postBuffer 1048576000 \
    && rm -rf /var/lib/apt/lists/*

RUN ldconfig /usr/local/cuda-12.1/compat/

# Install Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade -r /requirements.txt

# Install vLLM e FlashInfer 
RUN python3 -m pip install vllm==0.9.1 && \
    python3 -m pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3

# Setup for Option 2: Building the Image with the Model included
ARG MODEL_NAME=""
ARG TOKENIZER_NAME=""
ARG BASE_PATH="/runpod-volume"
ARG QUANTIZATION=""
ARG MODEL_REVISION=""
ARG TOKENIZER_REVISION=""

ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    BASE_PATH=$BASE_PATH \
    QUANTIZATION=$QUANTIZATION \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=0 

ENV PYTHONPATH="/:/vllm-workspace"

# Configurazione directory cache efficiente
RUN mkdir -p /workspace/cache && chmod 777 /workspace/cache
ENV TMPDIR=/workspace/cache/tmp \
    HF_HOME=/workspace/cache/huggingface

COPY src /src
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
    export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_NAME" ]; then \
    python3 /src/download_model.py; \
    fi

# Start the handler
CMD ["python3", "/src/handler.py"]
