#!/bin/bash

# Run inside SonoBase's own uv-managed venv (see Dockerfile) so nemo_cv and
# its hydra-instantiated model are importable.
exec uv run --project /sonobase gunicorn --bind :${PORT:-9090} --workers ${WORKERS:-1} --threads ${THREADS:-4} --timeout 0 --pythonpath '/app' _wsgi:app
