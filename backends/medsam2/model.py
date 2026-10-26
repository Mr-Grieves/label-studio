"""Label Studio ML backend for MedSAM2: propagate a hand-labelled mask through a cine.

A task holds a cine as a list of frame images (`images`), shown one frame per page with
<Image name="image" valueList="$images">; every region carries its frame as `item_index`.

1. Hand-label the object on one frame (normally the center one) as a mask or polygon, and submit
   or update the annotation: Label Studio only sends the backend its smart regions, so the backend
   reads the hand labels from the task's saved annotation.
2. Click with the smart KeyPointLabels tool:
   - on a hand-labelled frame: just starts propagation from it;
   - on any other frame: a corrective click there (alt-click for background), and propagation
     reruns with it.
3. The backend returns one prediction: a BrushLabels mask, with the click's label, on every frame
   that isn't hand-labelled.

Hand labels are the saved annotation's (or draft's) brush and polygon regions whose label matches
the click's, ignoring a bracketed suffix ("Effusion [SAM]" matches "Effusion [Polygon]"), except the
backend's own masks: anything on the hidden BrushLabels it returns its masks on.

Where the structure is visible comes from the hand labels' suffixes: "<label> [First]" on the first
frame it's visible, "<label> [Last]" on the last, "<label> [Middle]" (or any other suffix) anywhere in
between. They split the cine into stretches (see stretches): only [Middle] labels -> the whole cine;
[First] ... [Last] -> that range; [Last] ... [First] -> the start to [Last], then [First] to the end.
Each stretch is propagated on its own, from the hand labels in it; frames outside every stretch get
no mask and aren't tracked.

MedSAM2 (github.com/bowang-lab/MedSAM2: SAM2.1-tiny fine-tuned on medical images) runs out of the
box on the frames as Label Studio shows them. It's SAM2's video predictor: hand-labelled masks and
corrective clicks are prompts on their frames, propagated forwards and backwards through the cine.
"""

import json
import logging
import os
import shutil
import tempfile
import threading
from collections import OrderedDict
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from uuid import uuid4

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk import LabelStudio
from label_studio_sdk.converter import brush
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

logger = logging.getLogger(__name__)

# The checkpoint, from huggingface.co/wanglab/MedSAM2: MedSAM2_US_Heart.pt is fine-tuned on
# echocardiography videos; MedSAM2_latest.pt is the general model.
CHECKPOINT = os.getenv("MEDSAM2_CHECKPOINT", "MedSAM2_US_Heart.pt")
MODEL_CONFIG = os.getenv("MEDSAM2_CONFIG", "configs/sam2.1_hiera_t512.yaml")  # inside MedSAM2's sam2 package
DEVICE = os.getenv("DEVICE") or None  # None: CUDA, else Apple MPS, else CPU
STATE_CACHE_SIZE = int(os.getenv("STATE_CACHE_SIZE", 2))  # cines kept loaded on the device
CORRECTIONS_CACHE_SIZE = 256  # (task, label) pairs whose corrective clicks are remembered
DOWNLOAD_THREADS = int(os.getenv("DOWNLOAD_THREADS", 16))
KEEP_LARGEST = os.getenv("KEEP_LARGEST", "true").lower() == "true"  # one piece per frame

OBJ_ID = 1  # one object per request: the clicked label's


def base_label(label: str) -> str:
    """A label without its bracketed suffix: "Effusion [SAM]" -> "Effusion"."""
    return label.rsplit(" [", 1)[0] if label.endswith("]") else label


FIRST, MIDDLE, LAST = "[First]", "[Middle]", "[Last]"


def role(label: str) -> str:
    """A hand label's role from its suffix: "LV [First]" -> "[First]"; any other suffix -> "[Middle]"."""
    suffix = label[len(base_label(label)):].strip()
    return suffix if suffix in (FIRST, LAST) else MIDDLE


def stretches(roles: dict, n_frames: int) -> tuple[list[tuple[int, int]], list[str]]:
    """The stretches of frames where the structure is visible, as (first, last) pairs, from the hand
    labels' roles ({frame: {"[First]", "[Middle]", "[Last]"}}), plus warnings about inconsistencies.

    Walking through the frames: a stretch starts at a [First] -- or at frame 0 if the first hand label
    is a [Middle] or [Last] -- and ends at a [Last], or at the end of the cine. So only [Middle] labels
    give the whole cine; [Last] then [First] gives the start to [Last], then [First] to the end.
    A [First] inside a stretch is treated as a [Middle]; a [Middle] or [Last] after a stretch has ended
    starts a new one just after that end.
    """
    order = {FIRST: 0, MIDDLE: 1, LAST: 2}  # on one frame: a stretch that starts and ends there
    events = sorted((f, order[r], r) for f, rs in roles.items() for r in rs)
    result, warnings = [], []
    start, last_end = None, -1
    for i, (frame, _, r) in enumerate(events):
        if start is None:
            if r == FIRST:
                start = frame
            else:
                # The stretch began before this label: at the start of the cine, or after the last end
                start = last_end + 1
                if i:  # not the first label: the structure came back without a [First]
                    warnings.append(f"{r} on frame {frame} after the [Last] on frame {last_end}: assuming it reappears on frame {start}")
        elif r == FIRST:
            warnings.append(f"[First] on frame {frame} while already visible since frame {start}: treated as [Middle]")
        if r == LAST:
            result.append((start, frame))
            start, last_end = None, frame
    if start is not None:
        result.append((start, n_frames - 1))
    return result, warnings


def largest_component(mask: np.ndarray) -> np.ndarray:
    """The largest 8-connected component of a binary mask, as a bool mask (all False if empty)."""
    n, components, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros(mask.shape, bool)
    return components == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])


def result_to_mask(result: dict, width: int, height: int) -> np.ndarray | None:
    """A saved brush or polygon region -> its (height, width) bool mask on its frame."""
    value = result["value"]
    if result["type"] == "polygonlabels":
        points = np.array(value["points"], dtype=np.float32) / 100 * [width, height]
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
        return mask.astype(bool)
    if result["type"] == "brushlabels" and value.get("format") == "rle":
        w, h = result["original_width"], result["original_height"]
        mask = brush.decode_rle(value["rle"]).reshape(h, w, 4)[..., 3] > 0  # RGBA; the alpha marks the mask
        if (w, h) != (width, height):
            mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
        return mask
    return None


class LRUCache:
    """A small least-recently-used cache. Not thread-safe: used under MedSam2Handler's lock."""

    def __init__(self, size: int):
        self.size = size
        self.items = OrderedDict()

    def get(self, key):
        if key not in self.items:
            return None
        self.items.move_to_end(key)
        return self.items[key]

    def put(self, key, value):
        self.items[key] = value
        self.items.move_to_end(key)
        while len(self.items) > self.size:
            self.items.popitem(last=False)


class MedSam2Handler:
    """The model and the loaded cines, shared by every request (label-studio-ml makes a NewModel per request).

    The predictor and its inference states aren't thread-safe, so a request holds `lock` while it
    runs the model.
    """

    def __init__(self):
        checkpoint = hf_hub_download(repo_id="wanglab/MedSAM2", filename=CHECKPOINT)
        self.predictor = build_sam2_video_predictor(
            MODEL_CONFIG,
            checkpoint,
            device=DEVICE,
            # A frame given corrective clicks becomes a conditioning frame, kept in memory for every frame
            # like a hand label, as MedSAM2 was fine-tuned (sam2.1_hiera_tiny_finetune512.yaml). SAM2's
            # inference default only keeps it among the last few frames tracked.
            hydra_overrides_extra=["++model.add_all_frames_to_correct_as_cond=true"],
        )
        self.device = self.predictor.device
        logger.info(f"MedSAM2 {CHECKPOINT} loaded on {self.device}")
        self.states = LRUCache(STATE_CACHE_SIZE)  # task id -> the cine's inference state
        # (task id, label) -> (the hand labels' fingerprint, {(frame, x %, y %, positive), ...}): the
        # corrective clicks so far. Auto-accept deletes the clicks after each prediction, so Label
        # Studio only ever sends the newest one.
        self.corrections = LRUCache(CORRECTIONS_CACHE_SIZE)
        self.lock = threading.Lock()

    def load_cine(self, task_id: int, frame_paths: list[str]) -> dict:
        """The cine's inference state, cached per task: its frames, resized and normalized on the device."""
        state = self.states.get(task_id)
        if state is None:
            # SAM2 reads a cine from a folder of "<frame index>.jpg" files.
            with tempfile.TemporaryDirectory() as folder:
                for i, path in enumerate(frame_paths):
                    target = os.path.join(folder, f"{i:05d}.jpg")
                    if path.lower().endswith((".jpg", ".jpeg")):
                        shutil.copyfile(path, target)
                    else:
                        Image.open(path).convert("RGB").save(target, quality=95)
                state = self.predictor.init_state(
                    video_path=folder, offload_video_to_cpu=self.device.type != "cuda"
                )
            self.states.put(task_id, state)
        return state

    def propagate(self, state: dict, seeds: dict, clicks: dict, track: tuple = (None, None)) -> dict:
        """Masks for the tracked frames: {frame: (H, W) bool mask}.

        seeds: {frame: (H, W) bool mask}, the hand labels; clicks: {frame: (points (N, 2) in pixels,
        labels (N,): 1 foreground, 0 background)}; track: (first, last) frames to track, None for the
        start/end of the cine.
        """
        # bfloat16 on CUDA, as MedSAM2's own inference scripts run; full precision elsewhere.
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else nullcontext()
        with torch.inference_mode(), autocast:
            self.predictor.reset_state(state)
            for frame, mask in seeds.items():
                self.predictor.add_new_mask(state, frame_idx=frame, obj_id=OBJ_ID, mask=mask)
            masks = self._propagate_both_ways(state, min(seeds), *track)
            if clicks:
                # Clicks go in after a first propagation: SAM2 refines a frame it has tracked (from its
                # prediction there and its memory of the hand labels). On an untracked frame it would
                # segment from the clicks alone, and that frame would then steer the propagation.
                for frame, (points, labels) in clicks.items():
                    self.predictor.add_new_points_or_box(
                        state, frame_idx=frame, obj_id=OBJ_ID, points=points, labels=labels
                    )
                masks = self._propagate_both_ways(state, min(seeds), *track)
        return masks

    def _propagate_both_ways(self, state: dict, start: int, first: int | None = None, last: int | None = None) -> dict:
        """One continuous forward pass from `start` (the first hand-labelled frame), and a backward pass
        from it for the frames before it.

        Every prompted frame -- hand-labelled or clicked -- is a conditioning frame, and SAM2 attends to
        all of them (before and after) on every frame it tracks, so one forward pass already uses every
        prompt at once. The two passes only meet on a hand-labelled frame, so masks don't jump where one
        pass hands over to the other.
        """
        masks = {}
        for reverse in (False, True):
            # Stop at `last` going forwards and `first` going backwards (None: the end of the cine)
            end = first if reverse else last
            for frame, _, logits in self.predictor.propagate_in_video(
                state,
                start_frame_idx=start,
                max_frame_num_to_track=None if end is None else abs(start - end),
                reverse=reverse,
            ):
                masks.setdefault(frame, (logits[0, 0] > 0).cpu().numpy())
        return masks


class NewModel(LabelStudioMLBase):
    """MedSAM2: propagates the hand-labelled mask through the cine, steered by corrective clicks."""

    handler: MedSam2Handler | None = None  # created on the first request, then shared

    def setup(self):
        self.set("model_version", f"medsam2-{os.path.splitext(CHECKPOINT)[0]}")

    @classmethod
    def get_handler(cls) -> MedSam2Handler:
        if cls.handler is None:
            cls.handler = MedSam2Handler()
        return cls.handler

    def predict(self, tasks: list[dict], context: dict | None = None, **kwargs) -> ModelResponse:
        """Masks for every frame that isn't hand-labelled, from the hand labels and the smart clicks."""
        if not context or not context.get("result"):
            logger.info("No smart clicks: nothing to do (MedSAM2 only runs interactively).")
            return ModelResponse(predictions=[])
        clicks = [r for r in context["result"] if r["type"] == "keypointlabels"]
        if not clicks:
            logger.info("MedSAM2 is started by a smart click; other smart regions are ignored.")
            return ModelResponse(predictions=[])

        from_name, to_name, value = self.get_first_tag_occurence("BrushLabels", "Image")
        task = tasks[0]
        urls = task["data"][value]
        if not isinstance(urls, list):
            logger.warning(f"Task {task['id']}'s {value!r} isn't a list of frames; MedSAM2 needs a cine.")
            return ModelResponse(predictions=[])
        label = clicks[-1]["value"]["keypointlabels"][0]

        # The hand labels and presence marks, from the saved annotation or draft (Label Studio sends
        # only the smart clicks).
        saved = self.saved_result(task["id"])
        hand_labels = self.hand_labels(saved, base_label(label), prediction_tag=from_name)
        if not hand_labels:
            logger.warning(
                f"Task {task['id']} has no saved or drafted hand-labelled mask or polygon for {base_label(label)!r}: "
                "label a frame, and let the draft save (or submit) first."
            )
            return ModelResponse(predictions=[])

        frame_paths = self.download_frames(urls, task["id"])
        handler = self.get_handler()
        with handler.lock:
            state = handler.load_cine(task["id"], frame_paths)
            width, height = state["video_width"], state["video_height"]

            seeds, roles = {}, {}
            for result in hand_labels:
                mask = result_to_mask(result, width, height)
                if mask is not None:
                    frame = result.get("item_index", 0)
                    seeds[frame] = seeds.get(frame, np.zeros((height, width), bool)) | mask
                    roles.setdefault(frame, set()).update(role(l) for l in result["value"][result["type"]])
            visible, warnings = stretches(roles, len(urls))
            for warning in warnings:
                logger.warning(f"{base_label(label)!r}: {warning}")
            in_stretch = lambda f: any(a <= f <= b for a, b in visible)

            corrections = self.corrections(handler, task["id"], label, hand_labels, clicks, set(seeds))
            prompts = {}
            for frame, x, y, positive in corrections:
                if not in_stretch(frame):
                    continue  # outside where the structure is visible
                points, point_labels = prompts.setdefault(frame, ([], []))
                points.append([x / 100 * width, y / 100 * height])
                point_labels.append(1 if positive else 0)
            prompts = {f: (np.array(p, np.float32), np.array(l, np.int32)) for f, (p, l) in prompts.items()}

            logger.info(
                f"Propagating {base_label(label)!r} through {len(urls)} frames: visible on "
                f"{', '.join(f'{a}-{b}' for a, b in visible)}; hand labels on frames "
                f"{', '.join(f'{f}{sorted(roles[f])}' for f in sorted(roles))}; corrective clicks on frames {sorted(prompts)}"
            )
            # Each stretch is propagated on its own, over exactly its frames, from its own hand labels:
            # tracking through a gap, SAM2 loses the structure and doesn't pick it up again after, even
            # with a hand label there (it only takes the object's summary from prompts it has passed).
            masks = {}
            for first, last in visible:
                stretch_seeds = {f: m for f, m in seeds.items() if first <= f <= last}
                stretch_clicks = {f: c for f, c in prompts.items() if first <= f <= last}
                masks.update(handler.propagate(state, stretch_seeds, stretch_clicks, (first, last)))

        results = []
        for frame, mask in sorted(masks.items()):
            if frame in seeds or not in_stretch(frame):
                continue  # hand-labelled: left as drawn; outside the stretches: no mask
            if KEEP_LARGEST:
                mask = largest_component(mask)
            if not mask.any():
                continue
            results.append({
                "id": str(uuid4())[:8],
                "from_name": from_name,
                "to_name": to_name,
                "item_index": frame,
                "original_width": width,
                "original_height": height,
                "image_rotation": 0,
                "value": {"format": "rle", "rle": brush.mask2rle(mask.astype(np.uint8) * 255), "brushlabels": [label]},
                "type": "brushlabels",
                "readonly": False,
            })
        return ModelResponse(predictions=[{"result": results, "model_version": self.get("model_version")}])

    @staticmethod
    def corrections(
        handler: MedSam2Handler, task_id: int, label: str, hand_labels: list[dict], clicks: list[dict], seeds: set
    ) -> list[tuple]:
        """The corrective clicks to use, as (frame, x %, y %, positive): the ones remembered for this
        task and label, plus this request's. Call under handler.lock.

        With auto-accept, Label Studio deletes the smart clicks after each prediction, so a request
        carries only the newest click; earlier corrections are remembered here instead. Without it,
        every click is sent each time, and the duplicates collapse.

        Starting again: if the newest click is on a hand-labelled frame, or the saved hand labels have
        changed, the remembered corrections are dropped. Clicks on hand-labelled frames are never
        corrections -- in SAM2 they'd replace the frame's hand-labelled mask.
        """
        fingerprint = "\n".join(sorted(json.dumps([r.get("item_index"), r["value"]], sort_keys=True) for r in hand_labels))
        key = (task_id, base_label(label))
        remembered = handler.corrections.get(key)
        if remembered is None or remembered[0] != fingerprint or clicks[-1].get("item_index", 0) in seeds:
            remembered = (fingerprint, {})
        corrections = dict(remembered[1])  # an insertion-ordered set
        for click in clicks:
            frame = click.get("item_index", 0)
            if frame not in seeds:
                v = click["value"]
                corrections[(frame, round(v["x"], 3), round(v["y"], 3), click.get("is_positive", True))] = None
        handler.corrections.put(key, (fingerprint, corrections))
        return list(corrections)

    @staticmethod
    def saved_result(task_id: int) -> list[dict]:
        """The result of the task's most recently updated annotation or draft.

        Drafts count, so a cine can be propagated and corrected before the annotation is submitted.
        Label Studio autosaves the draft as you label; the backend can't tell who's clicking, so with
        several people on one task it uses whoever saved last.
        """
        client = LabelStudio(base_url=os.environ["LABEL_STUDIO_URL"], api_key=os.environ["LABEL_STUDIO_API_KEY"])
        saved = [(a.updated_at, a.result) for a in client.annotations.list(id=task_id) if not a.was_cancelled]
        # The SDK has no call for a task's drafts (and the task's own `drafts` field leaves them out),
        # so they're fetched with its authenticated HTTP client.
        drafts = client._client_wrapper.httpx_client.request(f"api/tasks/{task_id}/drafts", method="GET")
        drafts.raise_for_status()
        saved += [(datetime.fromisoformat(d["updated_at"]), d.get("result")) for d in drafts.json()]
        if not saved:
            return []
        _, latest = max(saved, key=lambda s: s[0])
        return latest or []

    @staticmethod
    def hand_labels(saved: list[dict], label: str, prediction_tag: str) -> list[dict]:
        """The hand-drawn brush and polygon regions labelled `label` (without suffix) in a saved result.

        Propagated masks are told apart by their tag, `prediction_tag` (the hidden BrushLabels the
        backend returns its masks on, which nobody draws with), not by their origin: Label Studio saves
        accepted interactive predictions with origin "manual", like hand-drawn regions.
        """
        return [
            r for r in saved
            if r.get("type") in ("brushlabels", "polygonlabels")
            and r.get("from_name") != prediction_tag
            and any(base_label(l) == label for l in r["value"].get(r["type"], []))
        ]

    def download_frames(self, urls: list[str], task_id: int) -> list[str]:
        """The cine's frames as local files, fetched through Label Studio in parallel (and cached by it)."""
        with ThreadPoolExecutor(DOWNLOAD_THREADS) as pool:
            return list(pool.map(lambda url: self.get_local_path(url, task_id=task_id), urls))
