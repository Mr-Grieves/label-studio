# Label Studio + interactive segmentation backends

Label Studio plus three interchangeable ML backends for interactive image
segmentation (SAM2, MedSAM, SonoBase), all run via a single `docker-compose.yml`.
Both Label Studio and its backends run in Docker now -- nothing needs a native
Python install or virtualenv anymore.

## Why one container, not several -- and why not *all* in one container either

Label Studio and each backend are still separate containers (that part wasn't
merged) -- combining them into a single container was considered and dropped:
the three backends already have very different base images and install steps
(SAM2 clones facebookresearch/sam2 at build time, MedSAM needs a manually
downloaded checkpoint, SonoBase clones NVIDIA's nemo-automodel and needs a
GPU), so cramming any of them into Label Studio's own image would mean running
multiple long-lived processes in one container via something like supervisord --
more fragile, and it would stop you from swapping which backend is active
without rebuilding Label Studio itself too. Keeping them separate but wiring
everything into one `docker-compose.yml` gets you the "one repo, one command"
convenience you're after without that downside -- and as a bonus, since Label
Studio and the backends now share Compose's built-in network, backends reach
it at `http://label-studio:8080` by service name, on any machine, with no more
LAN-IP or host.docker.internal juggling.

## Setup

```bash
cp .env.example .env
```

Bring up just Label Studio first, create your account, and generate an API key
under Account & Settings:

```bash
docker-compose up -d label-studio
```

Fill that key into `.env` as `LABEL_STUDIO_API_KEY`.

Note: the basic-auth env vars (`BASIC_AUTH_USER`/`PASS`) you may have seen in
older versions of these backends' docker-compose files don't actually do
anything -- `start.sh` runs the backend through gunicorn importing `_wsgi.py`
as a module, which skips the code path in `label-studio-ml-backend` that
reads those values (it only runs when `_wsgi.py` is executed directly via
`python`, not imported). So they've been dropped here rather than left in
looking like a working security control. The backend endpoints are only
reachable over Compose's internal network plus whatever ports you publish to
the host -- not a public API with auth in front of it.

## Running a backend

**On the Mac** (CPU only):

```bash
docker-compose --profile mac up -d label-studio sam2-backend
# or:
docker-compose --profile mac up -d label-studio medsam-backend
# or both at once, to compare:
docker-compose --profile mac up -d label-studio sam2-backend medsam-backend
```

**On the Linux GPU server:**

```bash
docker-compose --profile server up -d label-studio sonobase-backend
```

Then in Label Studio (Settings -> Machine Learning -> Add Model), point at
`http://localhost:9090` (SAM2), `9091` (MedSAM), or `9092` (SonoBase) -- from
the browser on whichever machine it's actually running on, or through an SSH
tunnel to that machine otherwise.

## The three backends, briefly

- **`backends/sam2/`** -- Meta's SAM2, general purpose, supports point and box
  prompts. The known-working baseline.
- **`backends/medsam/`** -- fine-tuned for medical imaging, but **box prompts
  only** -- point clicks are ignored (see the comments in `model.py`). Needs
  its checkpoint downloaded by hand from Google Drive (linked in
  `Dockerfile`'s comments) and placed at `backends/medsam/checkpoints/medsam_vit_b.pth`
  before building, since that link can't be scripted into the build.
- **`backends/sonobase/`** -- a newer (Sept. 2026) ultrasound-specific
  foundation model built on SAM2, supporting both point and box prompts.
  **GPU-only**, and the integration is less proven than the other two -- see
  the detailed caveats in `backends/sonobase/Dockerfile` and `model.py`
  (config-path fallback, heavy NVIDIA dependency, license). Model weights are
  CC BY-NC 4.0 (non-commercial).

## Cleaning up the old native install

This repo used to run Label Studio natively via a virtualenv. That's no
longer needed -- `requirements.txt` and the `venv/` folder here are unused
leftovers, safe to delete whenever you like.
