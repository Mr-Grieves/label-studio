FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN mamba update conda -y

RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    apt-get -y update \
    && apt-get install -y git curl build-essential ffmpeg libsm6 libxext6

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_CACHE_DIR=/.cache \
    PORT=9090 \
    WORKERS=2 \
    THREADS=4 \
    CUDA_HOME=/usr/local/cuda

RUN mamba install nvidia/label/cuda-12.4.0::cuda -y

ENV CUDA_HOME=/opt/conda \
    TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX;8.9;9.0"

# SonoBase's own documented install path uses uv (not plain pip) against its
# pyproject.toml + uv.lock -- that lockfile is the only place the exact
# nemo-automodel commit and its transitive deps are pinned, so we defer to it
# rather than hand-picking pip installs.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Clone SonoBase and install its dependencies exactly as its lockfile pins
# them. This includes nemo-automodel, pulled from NVIDIA's GitHub repo --
# expect this step to take a while and pull a lot of disk.
RUN git clone --depth 1 https://github.com/AlfredQin/sonobase.git /sonobase
WORKDIR /sonobase
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    uv sync

# Add the serving stack (label-studio-ml, gunicorn) into the SAME uv-managed
# venv rather than the container's system Python, so everything SonoBase
# needs (nemo_cv, the hydra-instantiated model, etc.) is importable at
# runtime without juggling two separate Python environments.
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    uv pip install gunicorn==22.0.0 "label-studio-ml @ git+https://github.com/HumanSignal/label-studio-ml-backend.git"

ENV PYTHONPATH="/sonobase/src"

# Download the checkpoint + its paired config from HuggingFace (public, direct
# CDN links -- unlike MedSAM's Google Drive folder, this can be scripted at
# build time). These two files are published together in the same HF repo;
# see model.py for the fallback if that pairing ever changes upstream.
RUN mkdir -p /sonobase_assets \
    && curl -L -o /sonobase_assets/sonobase_hiera_b_conv_s_conv_t.pt \
         https://huggingface.co/AlfredQin/sonobase/resolve/main/sonobase_hiera_b_conv_s_conv_t.pt \
    && curl -L -o /sonobase_assets/config.yaml \
         https://huggingface.co/AlfredQin/sonobase/resolve/main/config.yaml

WORKDIR /app
COPY . ./

CMD ["./start.sh"]
