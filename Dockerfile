# PyTorch is preinstalled in this image; requirements.txt intentionally keeps
# torch commented out to avoid replacing the CUDA-enabled PyTorch runtime.
# Override BASE_IMAGE when using an internal registry mirror:
#   docker build --build-arg BASE_IMAGE=<mirror>/pytorch:... -t ... .
ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_HOME=/tmp/torch \
    HF_HOME=/tmp/huggingface \
    Datasets_HOME=/tmp/huggingface/datasets \
    OMP_NUM_THREADS=1

# Install Python dependencies first so source-only edits do not invalidate the
# dependency layer.
WORKDIR /app

ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -i "${PIP_INDEX_URL}" -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt \
    && python -c "import torch; print(f'torch={torch.__version__} cuda={torch.version.cuda}')"

# Copy the training application. Dataset data is excluded by .dockerignore and
# must be mounted from OSS at runtime.
COPY . /app

# /app/out and /app/checkpoints are created at runtime as symlinks by
# dlc/train.sh, pointing to the mounted output storage.
RUN chmod +x /app/dlc/train.sh /app/dlc/smoke_test.sh \
    && python -m compileall -q /app

# PAI supplies a shell command. Using bash as the entrypoint makes that command
# execute exactly as written. The default CMD still supports `docker run IMAGE`.
ENTRYPOINT ["/bin/bash", "-lc"]
CMD ["exec /app/dlc/train.sh"]
