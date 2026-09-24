# Label Studio + interactive segmentation backends

Label Studio plus three interchangeable ML backends for interactive image
segmentation (SAM2, MedSAM, SonoBase), all run via a single `docker-compose.yml`
on the Linux GPU server. This repo assumes that's the only place it runs --
there's no CPU/Mac path anymore, everything here expects a GPU.

Label Studio and each backend run in separate containers: the three backends
already have very different base images and install steps (SAM2 clones
facebookresearch/sam2 at build time, MedSAM needs a manually downloaded
checkpoint, SonoBase clones NVIDIA's nemo-automodel), so cramming any of them
into Label Studio's own image would mean running multiple long-lived
processes in one container via something like supervisord -- more fragile,
and it would stop you from swapping which backend is active without
rebuilding Label Studio itself too. Keeping them separate but wiring
everything into one `docker-compose.yml` gets you the "one repo, one command"
convenience without that downside -- and since Label Studio and the backends
share Compose's built-in network, backends reach it at
`http://label-studio:8080` by service name, with no LAN-IP or
host.docker.internal juggling.

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

## The ideal workflow: where images live, end to end

**`./images/` is the single source of truth.** Everything else -- Label
Studio's project, and every backend's view of your pictures -- is derived
from what's in there. Nothing else should hold a second copy you maintain by
hand.

1. Drop image files into `./images/` (subfolders are fine).
2. In Label Studio: your project -> Settings -> Cloud Storage -> Add Source
   Storage -> **Local Files** -> absolute local path `/label-studio/images`
   (that's the path *inside* the container -- `./images/` is mounted there by
   `docker-compose.yml`) -> Test Connection -> **Save & Sync**.
3. That creates one task per image, referencing it in place -- Label Studio
   does **not** copy the file into `./data/`. Add more images later and hit
   Sync again to pick up just the new ones.
4. Never use the browser's drag-and-drop "Import" button for images that are
   already in `./images/` -- Import always copies the file into
   `./data/media/upload/`, which is exactly the duplication you ran into
   before. Import is only for one-off files you don't want to manage through
   `./images/` at all.

**How the ML backends reach those same files (the "mounted in a particular
location" part):** the `label-studio-ml-backend` SDK resolves a task's image
two possible ways, and it's automatic -- nothing in `model.py` decides this:

- If the image URL is a Local Storage reference (`/data/local-files/?d=...`)
  *and* the backend container has a file at
  `$LOCAL_FILES_DOCUMENT_ROOT/<that path>`, it reads it straight off disk.
  That's why every backend service in `docker-compose.yml` mounts
  `./images:/label-studio/images:ro` and sets
  `LOCAL_FILES_DOCUMENT_ROOT=/label-studio/images` -- identical to how
  `label-studio` itself is set up, so the paths line up and this always
  hits.
- If that lookup misses (file not mounted, or not synced yet), it
  transparently falls back to downloading the image over HTTP from
  `LABEL_STUDIO_URL`, authenticated with `LABEL_STUDIO_API_KEY` (also already
  set for every backend), then caches it. So a backend that *isn't* mounted
  correctly doesn't break -- it just does a slower HTTP round-trip per image
  instead of a local read. With the mounts in place as shipped here, you get
  the fast path by default.

Net effect: add an image once to `./images/`, hit Sync in Label Studio once,
and every backend can already see it -- no separate copying, no per-backend
setup.

## Running a backend

Every service requests a GPU (`deploy.resources.reservations.devices`) and
runs with `DEVICE=cuda` -- there's no CPU fallback to worry about. Bring up
everything at once:

```bash
docker-compose up -d
```

Or just Label Studio plus whichever backend(s) you want running right now,
to save GPU memory if you're not using all three:

```bash
docker-compose up -d label-studio sam2-backend
# or:
docker-compose up -d label-studio medsam-backend sonobase-backend
# etc.
```

Then in Label Studio (Settings -> Machine Learning -> Add Model), use the
Compose **service name**, not `localhost`, and each backend's *internal*
container port, which is always `9090` regardless of what it's published as
on the host:

- SAM2: `http://sam2-backend:9090`
- MedSAM: `http://medsam-backend:9090`
- SonoBase: `http://sonobase-backend:9090`

This matters because Label Studio's own container is the one making that
health-check request, not your browser -- `label-studio` and the backend
containers all share Compose's default network and can already reach each
other by service name, the same way the backends reach Label Studio at
`http://label-studio:8080`. `localhost` from inside the `label-studio`
container means the `label-studio` container itself, which is why
`http://localhost:9090` fails with connection refused even though the
backend is clearly running -- it's not a firewall or startup-order issue.

The published host ports (`localhost:9090` / `9091` / `9092`) are still
useful, just for a different purpose: hitting a backend directly yourself,
e.g. `curl http://localhost:9090/health` from a terminal on the server, or
through an SSH tunnel to it if you're elsewhere. Don't put those into the Add
Model form.

## The three backends, briefly

- **`backends/sam2/`** -- Meta's SAM2, general purpose, supports point and box
  prompts. The known-working baseline.
- **`backends/medsam/`** -- fine-tuned for medical imaging, but **box prompts
  only** -- point clicks are ignored (see the comments in `model.py`). Needs
  its checkpoint downloaded by hand from Google Drive (linked in
  `Dockerfile`'s comments) and placed at `backends/medsam/checkpoints/medsam_vit_b.pth`
  before building, since that link can't be scripted into the build.
- **`backends/sonobase/`** -- a newer (Sept. 2026) ultrasound-specific
  foundation model built on SAM2, supporting both point and box prompts. The
  integration is less proven than the other two -- see the detailed caveats
  in `backends/sonobase/Dockerfile` and `model.py` (upstream config bakes in
  a hardcoded checkpoint path from the original author's machine, which we
  patch around; heavy NVIDIA dependency; license). Model weights are CC
  BY-NC 4.0 (non-commercial).
