FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    apt-get -y update \
    && apt-get install -y git ffmpeg libsm6 libxext6

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_CACHE_DIR=/.cache \
    PORT=9090 \
    WORKERS=2 \
    THREADS=4

# install base requirements
COPY requirements-base.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements-base.txt

# install segment-anything (MedSAM reuses Meta's SAM architecture/registry --
# only the checkpoint weights are MedSAM-specific, so no custom repo needed)
COPY requirements.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements.txt

COPY . ./

# NOTE: the checkpoint itself is NOT baked into the image -- see README.md.
# It's mounted at runtime from ./checkpoints, since MedSAM's checkpoint is only
# distributed via Google Drive and can't be reliably fetched in a Docker build.

CMD ["./start.sh"]
