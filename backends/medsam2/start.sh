#!/bin/bash

# One worker: the model and the loaded cines live in the process.
exec gunicorn --bind :${PORT:-9090} --workers ${WORKERS:-1} --threads ${THREADS:-4} --timeout 0 --pythonpath '/app' _wsgi:app
