# Ubuntu 24.04: pyconvert is built against its ICU (libicu74) and glibc, and
# ships Python 3.12, which the pyconvert wheels are built for.
FROM ubuntu:24.04
ARG DEBIAN_FRONTEND=noninteractive
ARG POETRY_VERSION=2.4.1

WORKDIR /app

# - python3.12 + venv
# - git: label-studio-ml and the private-repo Poetry plugin come from GitHub
# - pyconvert's shared libraries (GL, fonts, D-Bus, X11, ICU, GLib), and xvfb:
#   it needs a display, so the app runs under xvfb-run (see tsam/start.sh). Same set
#   as isonoai's image.
RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    apt-get -y update \
    && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv git ca-certificates \
        xvfb xauth \
        libglx0 libopengl0 libegl1 libfreetype6 libfontconfig1 \
        libdbus-1-3 libx11-6 libicu74 libglib2.0-0t64

# Poetry in its own venv; the app's venv is /app/.venv (poetry.toml: in-project).
# WORKERS=1: the embedding cache in tsam/model.py is per process.
RUN python3.12 -m venv /opt/poetry && /opt/poetry/bin/pip install --no-cache-dir "poetry==${POETRY_VERSION}"
ENV PATH=/app/.venv/bin:/opt/poetry/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=9090 \
    WORKERS=1 \
    THREADS=4 \
    TF_CPP_MIN_LOG_LEVEL=1 \
    QT_QPA_PLATFORM=offscreen

# Dependencies, as in isonoai's image: a GitHub token (build secret
# PRIVATE_REPO_GH_TOKEN, see the Makefile's docker-build) gives the private-repo plugin
# access to pyconvert; it's removed again before the layer is committed.
COPY pyproject.toml poetry.lock poetry.toml ./
RUN --mount=type=secret,id=PRIVATE_REPO_GH_TOKEN \
    --mount=type=cache,target=/root/.cache/pypoetry,sharing=locked \
    poetry config pypi-token.pyisonosw "$(cat /run/secrets/PRIVATE_REPO_GH_TOKEN)" && \
    git clone --depth 1 "https://$(cat /run/secrets/PRIVATE_REPO_GH_TOKEN)@github.com/clariusmhealth/pypi-private-repo" /tmp/pypi-private-repo && \
    poetry self add /tmp/pypi-private-repo && \
    poetry install --no-interaction --no-root && \
    rm -rf /tmp/pypi-private-repo && \
    poetry config --unset pypi-token.pyisonosw

COPY . ./

# NOTE: the model is NOT baked into the image. The backend downloads it from
# MLflow run $MLFLOW_ID at startup, so a new model doesn't need a rebuild.

CMD ["./tsam/start.sh"]
