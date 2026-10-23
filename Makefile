.PHONY: help install secrets run docker-build docker-run

# The isonoai training run whose SAM to serve; override with `make run MLFLOW_ID=<run id>`
MLFLOW_ID ?= fb98dac80d9340689e39421c433dc4bb
# The host port the backend listens on (Label Studio, or a tunnel, points here)
PORT ?= 9094
IMAGE ?= tsam-backend
# isonoai's checkout, for its sops-encrypted secrets
ISONOAI ?= ../isonoai
# The secrets files the backend needs (see tsam/main.py's check_env)
SECRETS := aws label_studio mlflow

help: ## Show this help
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-14s %s\n", $$1, $$2}'

install: ## Install the Python environment into .venv
	poetry install --no-root

secrets: ## Decrypt the backend's secrets from isonoai's sops files into .env (git- and docker-ignored)
	@umask 077 && : > .env.tmp && for f in $(SECRETS); do \
		sops -d --output-type dotenv "$(ISONOAI)/secrets/$$f.enc.yaml" >> .env.tmp || { rm -f .env.tmp; exit 1; }; \
	done && mv .env.tmp .env

run: secrets ## Run the backend locally on PORT, serving MLflow run MLFLOW_ID
	.venv/bin/dotenv -f .env run -- .venv/bin/python -m tsam.main --port $(PORT) --mlflow-id $(MLFLOW_ID)

docker-build: ## Build the Docker image (needs `gh auth login` with access to the private pyconvert repo)
	PRIVATE_REPO_GH_TOKEN="$$(gh auth token)" docker build \
		--secret id=PRIVATE_REPO_GH_TOKEN,env=PRIVATE_REPO_GH_TOKEN -t $(IMAGE) .

# The image's gunicorn reads basic auth from BASIC_AUTH_*; passed by name from your ML_SERVER_BASIC_AUTH_*
# (what `make run` uses), so the values never appear on the command line.
docker-run: secrets ## Run the Docker image on PORT, serving MLflow run MLFLOW_ID
	BASIC_AUTH_USER="$$ML_SERVER_BASIC_AUTH_USER" BASIC_AUTH_PASS="$$ML_SERVER_BASIC_AUTH_PASS" \
		docker run --rm -p $(PORT):9090 --env-file .env -e MLFLOW_ID=$(MLFLOW_ID) \
		-e BASIC_AUTH_USER -e BASIC_AUTH_PASS -e SMOOTH_SIGMA -e EMBEDDING_CACHE_SIZE -e LOG_LEVEL $(IMAGE)
