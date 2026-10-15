import json
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def decode_images_json(images_json, timestamp: int | None = None) -> dict[str, Any]:
    """Decode raw_json_images string from metadata tensor.

    Args:
        images_json: Decoded JSON string from the metadata tensor

    Returns:
        images: decoded JSON string with parent key removed
    """
    # An image-blob_<uuid>.json file holds one image's JSON, encoded as a string
    if isinstance(images_json, str):
        return json.loads(images_json)

    # Prune parent key in images_json
    if len(images_json) == 1:
        return json.loads(next(iter(images_json.values())))

    if timestamp is not None:
        ts_str = str(timestamp)
        # Find the image containing the timestamp
        for image in images_json.values():
            if ts_str in image:
                return json.loads(image)

    return images_json


def get_prescan_size(images_json: dict[str, Any]) -> tuple[int, int]:
    """Read pre-scan width/height from decoded images JSON."""
    stream_0 = next(
        iter(
            [
                stream
                for section in images_json["sections"]
                for stream in section["streams"]
                if stream["info"]["type"] == "B pre-scan"
            ]
        )
    )
    prescan_w = int(stream_0["info"]["size"]["samples per line"])
    prescan_h = int(stream_0["info"]["size"]["number of lines"])
    return prescan_w, prescan_h


def convert_point_to_prescan(post_point: np.ndarray, ts: int, w: int, h: int, convert_utils: Any) -> np.ndarray | None:
    """Convert a single post-scan point into pre-scan coordinates."""
    # postScanToPreScan(points (N, 2), timestamp, width, height, bOnly) -> (N, 2), where
    # width/height are the post-scan frame's. It needs 2-D points, hence the reshape.
    points = np.asarray(post_point, dtype=np.float64).reshape(1, 2)
    try:
        pre_points = convert_utils.postScanToPreScan(points, ts, w, h, True)
    except RuntimeError:
        logger.exception("pyconvert.postScanToPreScan(): Frame contains no B/Color images")
        return None

    return float(pre_points[0][0]), float(pre_points[0][1])


def convert_mask_to_postscan(mask: np.ndarray, ts: int, w: int, h: int, convert_utils: Any) -> np.ndarray:
    """Convert a mask from the pre-scan image's coordinates to the model's post-scan frame."""
    try:
        mask = (mask * 255.0).astype(np.uint8)
        postscan_mask = convert_utils.convertMask(mask, timestamp=ts, width=w, height=h, bOnly=True)
        postscan_mask = np.squeeze(postscan_mask)
        return (postscan_mask > 127).astype(np.uint8)
    except Exception as _:
        logger.exception("Error converting mask to postscan coordinates")
    return None
