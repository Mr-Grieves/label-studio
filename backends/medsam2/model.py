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

Hand labels are the saved annotation's brush and polygon regions whose label matches the click's,
ignoring a bracketed suffix ("Effusion [SAM]" matches "Effusion [Polygon]"), and that were drawn by
hand: accepted predictions don't count, unless they were edited afterwards.

MedSAM2 (github.com/bowang-lab/MedSAM2: SAM2.1-tiny fine-tuned on medical images) runs out of the
box on the frames as Label Studio shows them. It's SAM2's video predictor: hand-labelled masks and
corrective clicks are prompts on their frames, propagated forwards and backwards through the cine.
"""

import logging
import os
import shutil
import tempfile
import threading
from collections import OrderedDict
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
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
DOWNLOAD_THREADS = int(os.getenv("DOWNLOAD_THREADS", 16))
KEEP_LARGEST = os.getenv("KEEP_LARGEST", "true").lower() == "true"  # one piece per frame
HAND_LABEL_ORIGINS = {"manual", "prediction-changed"}  # an accepted prediction you edited counts as yours

OBJ_ID = 1  # one object per request: the clicked label's


def base_label(label: str) -> str:
    """A label without its bracketed suffix: "Effusion [SAM]" -> "Effusion"."""
    return label.rsplit(" [", 1)[0] if label.endswith("]") else label


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
        self.predictor = build_sam2_video_predictor(MODEL_CONFIG, checkpoint, device=DEVICE)
        self.device = self.predictor.device
        logger.info(f"MedSAM2 {CHECKPOINT} loaded on {self.device}")
        self.states = LRUCache(STATE_CACHE_SIZE)  # task id -> the cine's inference state
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

    def propagate(self, state: dict, seeds: dict, clicks: dict) -> dict:
        """Masks for every frame: {frame: (H, W) bool mask}.

        seeds: {frame: (H, W) bool mask}, the hand labels; clicks: {frame: (points (N, 2) in pixels,
        labels (N,): 1 foreground, 0 background)}.
        """
        # bfloat16 on CUDA, as MedSAM2's own inference scripts run; full precision elsewhere.
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else nullcontext()
        with torch.inference_mode(), autocast:
            self.predictor.reset_state(state)
            for frame, mask in seeds.items():
                self.predictor.add_new_mask(state, frame_idx=frame, obj_id=OBJ_ID, mask=mask)
            masks = self._propagate_both_ways(state)
            if clicks:
                # Clicks go in after a first propagation: SAM2 refines a frame it has tracked (from its
                # prediction there and its memory of the hand labels). On an untracked frame it would
                # segment from the clicks alone, and that frame would then steer the propagation.
                for frame, (points, labels) in clicks.items():
                    self.predictor.add_new_points_or_box(
                        state, frame_idx=frame, obj_id=OBJ_ID, points=points, labels=labels
                    )
                masks = self._propagate_both_ways(state)
        return masks

    def _propagate_both_ways(self, state: dict) -> dict:
        masks = {}
        # Forwards from the first prompted frame, then backwards from the last.
        for reverse in (False, True):
            for frame, _, logits in self.predictor.propagate_in_video(state, reverse=reverse):
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

        # The hand labels, from the saved annotation (Label Studio sends only the smart clicks).
        hand_labels = self.hand_labels(task["id"], base_label(label))
        if not hand_labels:
            logger.warning(
                f"Task {task['id']} has no saved hand-labelled mask or polygon for {base_label(label)!r}: "
                "label a frame and submit or update the annotation first."
            )
            return ModelResponse(predictions=[])

        frame_paths = self.download_frames(urls, task["id"])
        handler = self.get_handler()
        with handler.lock:
            state = handler.load_cine(task["id"], frame_paths)
            width, height = state["video_width"], state["video_height"]

            seeds = {}
            for result in hand_labels:
                mask = result_to_mask(result, width, height)
                if mask is not None:
                    frame = result.get("item_index", 0)
                    seeds[frame] = seeds.get(frame, np.zeros((height, width), bool)) | mask

            # Corrective clicks, by frame; a click on a hand-labelled frame only starts propagation.
            prompts = {}
            for click in clicks:
                frame = click.get("item_index", 0)
                if frame in seeds:
                    continue
                points, point_labels = prompts.setdefault(frame, ([], []))
                points.append([click["value"]["x"] / 100 * width, click["value"]["y"] / 100 * height])
                point_labels.append(1 if click.get("is_positive", True) else 0)
            prompts = {f: (np.array(p, np.float32), np.array(l, np.int32)) for f, (p, l) in prompts.items()}

            logger.info(
                f"Propagating {base_label(label)!r} through {len(urls)} frames from hand labels on "
                f"frames {sorted(seeds)} and clicks on frames {sorted(prompts)}"
            )
            masks = handler.propagate(state, seeds, prompts)

        results = []
        for frame, mask in sorted(masks.items()):
            if frame in seeds:
                continue  # hand-labelled: left as drawn
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

    def hand_labels(self, task_id: int, label: str) -> list[dict]:
        """The saved hand-drawn brush and polygon regions labelled `label` (without suffix), from the
        task's most recently updated annotation."""
        client = LabelStudio(base_url=os.environ["LABEL_STUDIO_URL"], api_key=os.environ["LABEL_STUDIO_API_KEY"])
        annotations = [a for a in client.annotations.list(id=task_id) if not a.was_cancelled]
        if not annotations:
            return []
        latest = max(annotations, key=lambda a: a.updated_at)
        return [
            r for r in latest.result or []
            if r.get("type") in ("brushlabels", "polygonlabels")
            and r.get("origin", "manual") in HAND_LABEL_ORIGINS
            and any(base_label(l) == label for l in r["value"].get(r["type"], []))
        ]

    def download_frames(self, urls: list[str], task_id: int) -> list[str]:
        """The cine's frames as local files, fetched through Label Studio in parallel (and cached by it)."""
        with ThreadPoolExecutor(DOWNLOAD_THREADS) as pool:
            return list(pool.map(lambda url: self.get_local_path(url, task_id=task_id), urls))
