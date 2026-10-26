# medsam2: propagate a hand-labelled mask through a cine

Label one frame of a cine by hand, and [MedSAM2](https://github.com/bowang-lab/MedSAM2) (SAM2.1-tiny
fine-tuned on medical images, used out of the box) propagates it to every other frame. Corrective
clicks on any frame steer it.

## Project setup

Label Studio's `<Video>` tag only does boxes, so a cine is a **list of frame images**, one per page:
each task's `images` holds the frames' URLs. For example:

```xml
<View>
  <Image name="image" valueList="$images" zoom="true"/>

  <!-- Hand labels: polygons (or a <BrushLabels>). The suffix says where the structure is visible;
       see "Structures that come and go" -->
  <PolygonLabels name="polygon-labels" toName="image">
    <Label value="Effusion [First]"/>
    <Label value="Effusion [Middle]"/>
    <Label value="Effusion [Last]"/>
  </PolygonLabels>

  <!-- Clicks that run MedSAM2 -->
  <KeyPointLabels name="clicks" toName="image" smart="true" smartOnly="true">
    <Label value="Effusion [SAM]"/>
  </KeyPointLabels>

  <!-- Only there so Label Studio can show MedSAM2's masks -->
  <View style="display:none">
    <BrushLabels name="brush-labels" toName="image">
      <Label value="Effusion [SAM]"/>
    </BrushLabels>
  </View>
</View>
```

- The click labels and the hidden brush labels need **the same values**: masks get the click's label.
- A click's label matches the hand labels with the same name, ignoring a bracketed suffix
  (`Effusion [SAM]` uses the `Effusion [First]`, `[Middle]` and `[Last]` hand labels).

## Labelling a cine

Use **auto-accept**, with the `plugins/replace-propagated.js` plugin in the project. Without
auto-accept, Label Studio draws every pending mask of a frame list on whichever frame you're on
(its pending predictions aren't filtered by frame), which is unreadable and slow for a cine.

1. On the center frame (or any frame), label the object by hand, as a polygon or mask.
2. **Let the draft save** (Label Studio autosaves it a few seconds after a change), or submit. Label
   Studio only sends the backend its smart regions, so the backend reads the hand labels from the
   task's most recently updated draft or submitted annotation. Submit once you're happy with the
   propagated masks.
3. Turn on Auto-Annotation and auto-accept, and click with the smart keypoint tool:
   - on the hand-labelled frame: propagates the hand label through the cine, from scratch;
   - on another frame: a corrective click there (alt-click for background), then propagates again
     with it and every correction since.
   Clicking on an accepted mask selects it instead of placing a click: hold **Cmd** (Ctrl on
   Windows/Linux) to click through it -- Cmd+click for foreground, **Cmd+Alt+click** for background
   on a leak.
4. Each run is accepted as a mask on every frame that isn't hand-labelled; the plugin deletes the
   previous run's masks, so each frame keeps the newest.

Auto-accept deletes the clicks after each run, so the backend remembers your corrections for each
task and label (its log lists them). They're dropped when you click a hand-labelled frame, or when
the saved or drafted hand labels change.

Every hand-labelled frame and every corrected frame is a prompt that MedSAM2 keeps in memory for
all the others (a SAM2 "conditioning frame"), so any number of them work together in one continuous
pass -- if the masks start to drift, hand-label (or correct) a frame where they do. Hand labels are
the regions you drew (polygons, or masks on a visible brush label); the propagated masks, on the
hidden brush label, never count, so accepting them doesn't turn them into prompts. To fix a frame
by hand, draw a polygon on it.

A click on a hand-labelled frame only starts propagation: the backend drops it, since in SAM2 clicks
on a frame replace its mask prompt. To correct a hand-labelled frame, edit the label and save.

A background click on its own, far from the object, makes SAM2 drop the object on that frame:
correct with a foreground click, or a background click on the leak itself.

## Structures that come and go

In a sweep, a structure can enter and leave the view. The hand labels' suffixes say where it's
visible:

- `[First]`: hand-label the **first frame it's visible on**, when it appears partway through;
- `[Last]`: the **last frame it's visible on**, when it disappears partway through;
- `[Middle]` (or any other suffix): any other hand label -- optional, to steer it.

| Hand labels | Masks on |
|---|---|
| only `[Middle]` | the whole cine |
| `[First]` ... `[Last]` | that range |
| `[Last]` ... `[First]` (a gap) | the start to `[Last]`, then `[First]` to the end |
| `[First]` ... `[Middle]` | `[First]` to the end |

Each stretch is propagated on its own, over exactly its frames, from the hand labels in it, so the
boundary frames are always prompts. Frames outside every stretch get no mask and aren't tracked, and
clicks there are ignored. If the labels don't fit -- a second `[First]` before any `[Last]`, or a
`[Middle]` after a `[Last]` with no `[First]` -- the backend makes a guess and logs a warning. The log
lists the stretches and each hand label's role on every run.

## Running it

Locally (Apple Silicon runs it on the GPU via MPS): create the venv once,

```bash
cd backends/medsam2
```
```bash
python3.12 -m venv .venv && .venv/bin/pip install torch torchvision -r requirements-base.txt -r requirements.txt
```
```bash
curl -L https://codeload.github.com/bowang-lab/MedSAM2/tar.gz/332f30d420f1d1b08e2a79b3ae6a602458808383 | tar xz && mv MedSAM2-332f30d420f1d1b08e2a79b3ae6a602458808383 MedSAM2
```
```bash
SAM2_BUILD_CUDA=0 .venv/bin/pip install --no-build-isolation -e ./MedSAM2
```

then start **MedSAM2 backend (debug)** in VS Code (port 9095). In Docker on the GPU server:
`docker compose up -d medsam2-backend` (port 9094).

The checkpoint is downloaded from Hugging Face on the first request. Settings (environment):

| Variable | Default | |
|---|---|---|
| `MEDSAM2_CHECKPOINT` | `MedSAM2_US_Heart.pt` | from `wanglab/MedSAM2`; `MedSAM2_latest.pt` is the general model |
| `DEVICE` | CUDA, else MPS, else CPU | |
| `KEEP_LARGEST` | `true` | keep each frame's largest piece |
| `STATE_CACHE_SIZE` | `2` | cines kept loaded on the device |
