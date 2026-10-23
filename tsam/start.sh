#!/bin/bash

# Execute the gunicorn command. Under xvfb-run, because pyconvert needs a display
# (isonoai's portal runs the same way).
exec xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24 -ac +extension RANDR +render -noreset -nolock' \
    gunicorn --bind :${PORT:-9090} --workers ${WORKERS:-1} --threads ${THREADS:-4} --timeout 0 --pythonpath '/app' tsam.main:app
