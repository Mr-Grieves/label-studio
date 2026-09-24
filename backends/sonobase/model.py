import torch
import numpy as np
import os
import glob
from typing import List, Dict, Optional
from uuid import uuid4
import hydra
from omegaconf import OmegaConf
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.converter import brush
from PIL import Image

# SonoBase (https://github.com/AlfredQin/sonobase) adapts SAM2, keeping SAM2's
# prompt encoder / mask decoder / SAM2ImagePredictor interface -- only the image
# encoder is swapped for their Hiera+ConvNeXt hybrid. Their own fork of the
# predictor class lives under the repo's "nemo_cv" package (see README), which
# is why this import differs from the plain SAM2 example's `sam2.*` imports.
from nemo_cv.components.models.sam2.sam2_image_predictor import SAM2ImagePredictor

ROOT_DIR = os.getcwd()

# Built for GPU deployment -- SonoBase's dependency chain (nemo-automodel) and
# reference code assume CUDA. This will NOT run on a Mac (no CUDA at all,
# Apple Silicon or otherwise) -- it's meant to be built/run on the GPU server.
DEVICE = os.getenv('DEVICE', 'cuda')

# These two files are downloaded together from the same HuggingFace repo at
# build time (see Dockerfile) since they're published as a matched pair --
# unlike the code repo's own README, which points at a config path
# (configs/analysis/model/sam2.yaml) that doesn't actually exist in the repo
# as of this writing. If HuggingFace ever changes this, MODEL_CONFIG below
# falls back to searching the cloned code repo for a similarly-named config.
MODEL_CONFIG = os.getenv('MODEL_CONFIG', '/sonobase_assets/config.yaml')
MODEL_CHECKPOINT = os.getenv('MODEL_CHECKPOINT', '/sonobase_assets/sonobase_hiera_b_conv_s_conv_t.pt')
SONOBASE_SRC = os.getenv('SONOBASE_SRC', '/sonobase/src')


def _resolve_config_path(path):
    if os.path.exists(path):
        return path
    # Fallback: the shipped config went missing/renamed -- search the cloned
    # code repo instead of failing outright.
    candidates = glob.glob(os.path.join(SONOBASE_SRC, '..', '**', '*sam2*.yaml'), recursive=True)
    if candidates:
        print(f'MODEL_CONFIG not found at {path}, falling back to {candidates[0]}')
        return candidates[0]
    raise FileNotFoundError(
        f'Could not find a SonoBase model config at {path} or anywhere under {SONOBASE_SRC}/.. '
        f'-- set MODEL_CONFIG explicitly.'
    )


config_path = _resolve_config_path(MODEL_CONFIG)
cfg = OmegaConf.load(config_path)
sonobase_model = hydra.utils.instantiate(cfg, _recursive_=True, _convert_="all")

state_dict = torch.load(MODEL_CHECKPOINT, map_location=torch.device(DEVICE), weights_only=True)
if isinstance(state_dict, dict) and 'model' in state_dict:
    state_dict = state_dict['model']
sonobase_model.load_state_dict(state_dict, strict=False)
sonobase_model = sonobase_model.to(DEVICE).eval()

predictor = SAM2ImagePredictor(sonobase_model)


class NewModel(LabelStudioMLBase):
    """SonoBase ML Backend model.

    Unlike MedSAM, SonoBase supports both point and box prompts (it keeps
    SAM2's SAM2ImagePredictor-style interface), so this mirrors the SAM2
    example's predict() logic almost exactly -- only model loading differs.
    """

    def get_results(self, masks, probs, width, height, from_name, to_name, label):
        results = []
        total_prob = 0
        for mask, prob in zip(masks, probs):
            label_id = str(uuid4())[:4]
            mask = mask * 255
            rle = brush.mask2rle(mask)
            total_prob += prob
            results.append({
                'id': label_id,
                'from_name': from_name,
                'to_name': to_name,
                'original_width': width,
                'original_height': height,
                'image_rotation': 0,
                'value': {
                    'format': 'rle',
                    'rle': rle,
                    'brushlabels': [label],
                },
                'score': prob,
                'type': 'brushlabels',
                'readonly': False
            })

        return [{
            'result': results,
            'model_version': self.get('model_version'),
            'score': total_prob / max(len(results), 1)
        }]

    def set_image(self, image_url, task_id):
        image_path = self.get_local_path(image_url, task_id=task_id)
        image = Image.open(image_path)
        image = np.array(image.convert("RGB"))
        predictor.set_image(image)

    def _sonobase_predict(self, img_url, point_coords=None, point_labels=None, input_box=None, task=None):
        self.set_image(img_url, task.get('id'))
        point_coords = np.array(point_coords, dtype=np.float32) if point_coords else None
        point_labels = np.array(point_labels, dtype=np.float32) if point_labels else None
        input_box = np.array(input_box, dtype=np.float32) if input_box else None

        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=input_box,
            multimask_output=True
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        mask = masks[0, :, :].astype(np.uint8)
        prob = float(scores[0])
        return {
            'masks': [mask],
            'probs': [prob]
        }

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """Returns the predicted mask for a smart keypoint/box that has been placed."""

        from_name, to_name, value = self.get_first_tag_occurence('BrushLabels', 'Image')

        if not context or not context.get('result'):
            return ModelResponse(predictions=[])

        image_width = context['result'][0]['original_width']
        image_height = context['result'][0]['original_height']

        point_coords = []
        point_labels = []
        input_box = None
        selected_label = None
        for ctx in context['result']:
            x = ctx['value']['x'] * image_width / 100
            y = ctx['value']['y'] * image_height / 100
            ctx_type = ctx['type']
            selected_label = ctx['value'][ctx_type][0]
            if ctx_type == 'keypointlabels':
                point_labels.append(int(ctx.get('is_positive', 0)))
                point_coords.append([int(x), int(y)])
            elif ctx_type == 'rectanglelabels':
                box_width = ctx['value']['width'] * image_width / 100
                box_height = ctx['value']['height'] * image_height / 100
                input_box = [int(x), int(y), int(box_width + x), int(box_height + y)]

        img_url = tasks[0]['data'][value]
        predictor_results = self._sonobase_predict(
            img_url=img_url,
            point_coords=point_coords or None,
            point_labels=point_labels or None,
            input_box=input_box,
            task=tasks[0]
        )

        predictions = self.get_results(
            masks=predictor_results['masks'],
            probs=predictor_results['probs'],
            width=image_width,
            height=image_height,
            from_name=from_name,
            to_name=to_name,
            label=selected_label)

        return ModelResponse(predictions=predictions)
