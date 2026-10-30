.PHONY: help install secrets run replay docker-build docker-run

# The online-training config (base model, project, test set...); override with `make run CONFIG=<path>`
CONFIG ?= tsam/configs/foot-ankle.yaml
# Where the config's test tfrecords are (isonoai's /data)
DATA_DIR ?= /data
# The host port the backend listens on (Label Studio, or a tunnel, points here)
PORT ?= 9094
IMAGE ?= tsam-backend
# isonoai's checkout, for its sops-encrypted secrets
ISONOAI ?= ../isonoai
# The secrets files the backend needs (see tsam/main.py's check_env)
SECRETS := aws label_studio mlflow
# On macOS, point sops at ~/.config/sops/age/keys.txt (where isonoai's onboarding puts the key) unless
# your shell already sets it: there, sops otherwise only looks in ~/Library/Application Support/sops/age/.
# Elsewhere sops finds its key as it does for isonoai.
ifeq ($(shell uname),Darwin)
export SOPS_AGE_KEY_FILE ?= $(HOME)/.config/sops/age/keys.txt
endif

help: ## Show this help
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-14s %s\n", $$1, $$2}'

install: ## Install the Python environment into .venv
	poetry install --no-root

secrets: ## Decrypt the backend's secrets from isonoai's sops files into .env (git- and docker-ignored)
	@umask 077 && : > .env.tmp && for f in $(SECRETS); do \
		sops -d --output-type dotenv "$(ISONOAI)/secrets/$$f.enc.yaml" >> .env.tmp || { rm -f .env.tmp; exit 1; }; \
	done && mv .env.tmp .env

# run and docker-run use .env as it is, and only decrypt it when there's none. So on a machine whose
# sops key can't decrypt isonoai's secrets (e.g. a server), copy in a .env made with `make secrets`
# elsewhere; refresh it with `make secrets` when the secrets change.
.env:
	@$(MAKE) --no-print-directory secrets

run: .env ## Run the backend locally on PORT, with the online-training config CONFIG
	DATA_DIR=$(DATA_DIR) .venv/bin/dotenv -f .env run -- .venv/bin/python -m tsam.main --port $(PORT) --config $(CONFIG)

replay: .env ## Replay CONFIG's project's existing annotations through online training (tsam/replay.py)
	DATA_DIR=$(DATA_DIR) .venv/bin/dotenv -f .env run -- .venv/bin/python -m tsam.replay --config $(CONFIG)

# As isonoai's `make build`: $PRIVATE_REPO_GH_TOKEN, the token that can read the private pyconvert repo
# (isonoai's docs/dev/contributing.md), passed as a build secret, so it's never in the image or its history.
docker-build: ## Build the Docker image (needs $PRIVATE_REPO_GH_TOKEN, as for isonoai's image)
	@if [ -z "$$PRIVATE_REPO_GH_TOKEN" ]; then \
		echo "Error: PRIVATE_REPO_GH_TOKEN is not set (see isonoai's docs/dev/contributing.md)."; \
		exit 1; \
	fi
	@echo "$$PRIVATE_REPO_GH_TOKEN" | DOCKER_BUILDKIT=1 docker build \
		--secret id=PRIVATE_REPO_GH_TOKEN,src=/dev/stdin -t $(IMAGE) .

# The image's gunicorn reads basic auth from BASIC_AUTH_*; passed by name from your ML_SERVER_BASIC_AUTH_*
# (what `make run` uses), so the values never appear on the command line.
# --init: a minimal init as PID 1. xvfb-run (tsam/start.sh) hangs as PID 1, waiting for a signal from
# Xvfb that it never gets.
docker-run: .env ## Run the Docker image on PORT, with the online-training config CONFIG
	BASIC_AUTH_USER="$$ML_SERVER_BASIC_AUTH_USER" BASIC_AUTH_PASS="$$ML_SERVER_BASIC_AUTH_PASS" \
		docker run --init --rm -p $(PORT):9090 --env-file .env -e CONFIG=$(CONFIG) -v $(DATA_DIR):/data:ro \
		-e BASIC_AUTH_USER -e BASIC_AUTH_PASS -e SMOOTH_SIGMA -e EMBEDDING_CACHE_SIZE -e LOG_LEVEL $(IMAGE)
