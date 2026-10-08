FROM python:3.12-slim
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# git: label-studio-ml is installed from GitHub
RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    apt-get -y update \
    && apt-get install -y git

# WORKERS=1: the embedding and mask caches in model.py are per process.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_CACHE_DIR=/.cache \
    PORT=9090 \
    WORKERS=1 \
    THREADS=4 \
    TF_CPP_MIN_LOG_LEVEL=1

# install base requirements
COPY requirements-base.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements-base.txt

# TensorFlow, pinned to isonoai's version
COPY requirements.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements.txt

COPY . ./

# NOTE: the model is NOT baked into the image. The two SavedModels from
# export_savedmodel.py are mounted at runtime from ./savedmodel (see
# docker-compose.yml), so swapping models doesn't need a rebuild.

CMD ["./start.sh"]
