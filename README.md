# T-SAM Label Studio backend

A Label Studio ML backend that serves isonoai's box-only SAM for interactive segmentation:
draw a box with the smart rectangle tool, get a mask back. The model is loaded from an isonoai
MLflow training run at startup; nothing from isonoai is imported.

- `tsam/` — the backend: `main.py` (entry point), `model.py` (prediction), `utils/` (the SAM classes
  copied from isonoai, S3, pyconvert).
- `plugins/` — Label Studio Enterprise plugins for the labeling screen (see `plugins/README.md`,
  which also has the labeling config the backend expects).

Linear probes only: boxes are converted from post-scan to prescan by their two corners.

## Setup (once)

- **Python 3.12 and Poetry 2.4**, as in `.tool-versions` (`asdf install`).
- **The private package index**, for `pyconvert` (as in isonoai's `docs/dev/contributing.md`):
  ```bash
  poetry self add git+https://github.com/clariusmhealth/pypi-private-repo
  poetry config pypi-token.pyisonosw $(gh auth token)
  ```
- **isonoai checked out at `../isonoai`**, and **sops** able to decrypt its secrets (you can run
  isonoai's `make load_secrets`). The backend's secrets (AWS, Label Studio, MLflow) come from there.
- **For Docker:** Docker, and `PRIVATE_REPO_GH_TOKEN` exported: the same token as for isonoai's image,
  a fine-grained token that can read `clariusmhealth/pypi-private-repo` (isonoai's
  `docs/dev/contributing.md`).

`make help` lists the targets. The secrets go in `.env` (git- and docker-ignored): `make run` and
`make docker-run` decrypt it with `make secrets` if it's missing, and otherwise use it as it is, so
run `make secrets` again when isonoai's secrets change.

**On a machine that can't decrypt isonoai's secrets** (a server whose sops key isn't one of
isonoai's), copy `.env` from a machine that can, instead of decrypting:

```bash
make secrets                                   # on your machine
scp .env <server>:<path to this repo>/.env     # then, on the server: chmod 600 .env
```

## Run locally

```bash
make install                   # once, and after dependency changes
make run                       # serves on http://localhost:9094
make run MLFLOW_ID=<run id>    # a different training run
```

To debug, use the VS Code launch config **T-SAM backend (debug)**: it runs `make secrets` first
and loads `.env` (`.vscode/launch.json`, `.vscode/tasks.json`). Its MLflow run ID is set in
`launch.json`.

## Run in Docker

```bash
make docker-build              # once, and after code or dependency changes
make docker-run                # serves on http://localhost:9094
```

The model isn't baked into the image: it's downloaded from MLflow run `MLFLOW_ID` at startup, so
a new model only needs `make docker-run MLFLOW_ID=<run id>`.

## Connecting Label Studio

Label Studio (cloud) has to reach the backend over HTTPS. For a quick test, tunnel the port:

```bash
cloudflared tunnel --url http://localhost:9094
```

Then in the project: Settings → Model → Connect Model, with the tunnel's URL and interactive
preannotations on. To require a password, set `ML_SERVER_BASIC_AUTH_USER` and
`ML_SERVER_BASIC_AUTH_PASS` in your environment before `make run` / `make docker-run`, and choose
Basic Authentication in Label Studio with the same values.

## Settings

| Variable / `make` option | Default | |
|---|---|---|
| `MLFLOW_ID` | see `Makefile` | The isonoai training run to serve |
| `PORT` | `9094` | Host port |
| `SMOOTH_SIGMA` | `2.5` | Gaussian smoothing of the mask logits, in pixels of the model's 256 frame; 0 = off |
| `EMBEDDING_CACHE_SIZE` | `16` | Images whose embeddings are kept, so a new box on them only runs the decoder |
| `LOG_LEVEL` | `INFO` | |

`MLFLOW_ID` and `PORT` are `make` options (`make run PORT=9095`); the rest are environment
variables, passed through to Docker too (`SMOOTH_SIGMA=3 make docker-run`).
The embedding cache is per process, so the Docker image runs one gunicorn worker.
