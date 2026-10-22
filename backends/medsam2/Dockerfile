FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
ARG DEBIAN_FRONTEND=noninteractive
# The MedSAM2 commit this backend was tested against
ARG MEDSAM2_COMMIT=332f30d420f1d1b08e2a79b3ae6a602458808383

WORKDIR /app

# git: label-studio-ml and MedSAM2 come from GitHub
RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    apt-get -y update \
    && apt-get install -y --no-install-recommends git ca-certificates

# WORKERS=1: the model and the loaded cines live in the process.
# HF_HOME: where the checkpoint is downloaded on first use -- a volume, so it's kept (docker-compose.yml).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_CACHE_DIR=/.cache \
    PORT=9090 \
    WORKERS=1 \
    THREADS=4 \
    HF_HOME=/data/huggingface

COPY requirements-base.txt requirements.txt ./
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements-base.txt -r requirements.txt

# MedSAM2, as an editable install of a clone: its model configs are only found that way (the package
# doesn't ship them). Its build imports torch, already in the base image, hence --no-build-isolation;
# SAM2_BUILD_CUDA=0 skips SAM2's optional hole-filling CUDA extension.
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    git clone https://github.com/bowang-lab/MedSAM2.git /MedSAM2 \
    && git -C /MedSAM2 checkout ${MEDSAM2_COMMIT} \
    && SAM2_BUILD_CUDA=0 pip install --no-build-isolation -e /MedSAM2

COPY . ./

CMD ["./start.sh"]
