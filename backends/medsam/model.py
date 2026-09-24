import torch
import numpy as np
import os
from typing import List, Dict, Optional
from uuid import uuid4
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.converter import brush
from PIL import Image
from skimage import transform
import torch.nn.functional as F

from segment_anything import sam_model_registry

ROOT_DIR = os.getcwd()

DEVICE = os.getenv('DEVICE', 'cpu')
# MedSAM's public checkpoint is a fine-tuned SAM ViT-B -- there is no other
# released model size, so this isn't meant to be swapped like SAM2's configs.
MODEL_TYPE = os.getenv('MODEL_TYPE', 'vit_b')
MODEL_CHECKPOINT = os.getenv('MODEL_CHECKPOINT', 'checkpoints/medsam_vit_b.pth')

medsam_checkpoint = str(os.path.join(ROOT_DIR, MODEL_CHECKPOINT))

# segment_anything's build_sam_vit_b() loads the checkpoint internally with a
# bare torch.load() (no map_location), so it assumes whatever device the
# checkpoint was saved on -- which fails on a CPU-only machine if the
# checkpoint was saved from a GPU tensor. Build the architecture first, then
# load the state dict ourselves with an explicit map_location.
medsam_model = sam_model_registry[MODEL_TYPE](checkpoint=None)
state_dict = torch.load(medsam_checkpoint, map_location=torch.device(DEVICE))
medsam_model.load_state_dict(state_dict)
medsam_model = medsam_model.to(DEVICE)
medsam_model.eval()


class NewModel(LabelStudioMLBase):
    """MedSAM ML Backend model.

    Unlike SAM/SAM2, the public MedSAM checkpoint was fine-tuned on bounding-box
    prompts only (see https://github.com/bowang-lab/MedSAM) -- it was not trained
    on point clicks and does not segment reliably from them. This backend only
    acts on the RectangleLabels ("smart" box) interaction; KeyPointLabels clicks
    are ignored if present, since forwarding them through the prompt encoder as
    point prompts would just produce noise for this checkpoint.
    """

    @torch.no_grad()
    def _embed_image(self, image_url, task_id):
        image_path = self.get_local_path(image_url, task_id=task_id)
        image = Image.open(image_path)
        image = np.array(image.convert("RGB"))
        H, W, _ = image.shape

        # MedSAM's own preprocessing: resize to 1024x1024 and min-max normalize
        # to [0, 1] -- this differs from the standard SAM predictor's preprocessing,
        # and matters for mask quality since the checkpoint was fine-tuned on it.
        img_1024 = transform.resize(
            image, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )
        img_1024_tensor = (
            torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        )
        image_embedding = medsam_model.image_encoder(img_1024_tensor)
        return image_embedding, H, W

    @torch.no_grad()
    def _medsam_predict(self, image_embedding, input_box, H, W):
        box_np = np.array([input_box])
        box_1024 = box_np / np.array([W, H, W, H]) * 1024
        box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=DEVICE)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :]  # (B, 1, 4)

        sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        low_res_logits, _ = medsam_model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=medsam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        low_res_pred = torch.sigmoid(low_res_logits)
        low_res_pred = F.interpolate(
            low_res_pred, size=(H, W), mode="bilinear", align_corners=False,
        )
        low_res_pred = low_res_pred.squeeze().cpu().numpy()
        mask = (low_res_pred > 0.5).astype(np.uint8)

        # MedSAM doesn't emit an IoU-style confidence the way SAM/SAM2 does;
        # approximate one as the mean predicted probability inside the mask.
        prob = float(low_res_pred[mask > 0].mean()) if mask.sum() > 0 else 0.0
        return mask, prob

    def get_results(self, mask, prob, width, height, from_name, to_name, label):
        label_id = str(uuid4())[:4]
        rle = brush.mask2rle(mask * 255)
        return [{
            'result': [{
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
            }],
            'model_version': self.get('model_version'),
            'score': prob
        }]

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """Returns the predicted mask for a smart bounding box that has been drawn."""

        from_name, to_name, value = self.get_first_tag_occurence('BrushLabels', 'Image')

        if not context or not context.get('result'):
            # no interaction has happened yet
            return ModelResponse(predictions=[])

        image_width = context['result'][0]['original_width']
        image_height = context['result'][0]['original_height']

        input_box = None
        selected_label = None
        saw_point = False
        for ctx in context['result']:
            ctx_type = ctx['type']
            selected_label = ctx['value'][ctx_type][0]
            if ctx_type == 'rectanglelabels':
                x = ctx['value']['x'] * image_width / 100
                y = ctx['value']['y'] * image_height / 100
                box_width = ctx['value']['width'] * image_width / 100
                box_height = ctx['value']['height'] * image_height / 100
                input_box = [int(x), int(y), int(box_width + x), int(box_height + y)]
            elif ctx_type == 'keypointlabels':
                saw_point = True

        if input_box is None:
            if saw_point:
                print('MedSAM only supports box prompts -- draw a rectangle instead of a point click.')
            return ModelResponse(predictions=[])

        img_url = tasks[0]['data'][value]
        image_embedding, H, W = self._embed_image(img_url, tasks[0].get('id'))
        mask, prob = self._medsam_predict(image_embedding, input_box, H, W)

        predictions = self.get_results(
            mask=mask,
            prob=prob,
            width=image_width,
            height=image_height,
            from_name=from_name,
            to_name=to_name,
            label=selected_label)

        return ModelResponse(predictions=predictions)
