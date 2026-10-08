"""Export isonoai's SAM from MLflow as two TF SavedModels that run with TensorFlow alone.

Run from the isonoai environment (it needs isonoai to load the model, once):

    python export_savedmodel.py [RUN_ID] [OUTPUT_DIR]

Loading needs no isonoai code, no custom Keras classes and no MLflow:

    encoder = tf.saved_model.load(f"{OUTPUT_DIR}/encoder")
    decoder = tf.saved_model.load(f"{OUTPUT_DIR}/decoder")
    embeddings = encoder.serve(images)                    # (B, S, S, 1) -> (B, E, E, dim)
    out = decoder.serve({"embeddings": embeddings, "points": points, "labels": labels,
                         "pred_mask": pred_mask, "has_mask": has_mask})
    logits, iou = out["logits"], out["iou"]               # (B, 4E, 4E), (B,)

points are (B, N, 2) clicks in pixels of the S x S input frame, labels (B, N) 1/0.
pred_mask is the previous click's logits, with has_mask 1; on the first click pass
zeros and has_mask 0.
"""
import os
import sys

import keras
import mlflow
import mlflow.tensorflow

from isonoai.models.sam import VIT_LAYERS  # also registers isonoai's SAM classes

RUN_ID = sys.argv[1] if len(sys.argv) > 1 else "2075f77bd1ee4e069b55b156d990d059"
OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "effsam_savedmodel"


def build_decoder(model) -> keras.Model:
    """SAM.decode as a functional model: one fixed set of inputs, mask prompt switched by `has_mask`."""
    grid, dim = model.prompt_encoder.embedding_size, model.prompt_encoder.dim
    inputs = {
        "embeddings": keras.Input((grid, grid, dim), name="embeddings"),
        "points": keras.Input((None, 2), name="points"),
        "labels": keras.Input((None,), name="labels"),
        "pred_mask": keras.Input((4 * grid, 4 * grid), name="pred_mask"),
        "has_mask": keras.Input((), name="has_mask"),
    }
    clicks = inputs["embeddings"], inputs["points"], inputs["labels"]
    logits, iou = model.decode(*clicks)
    looped_logits, looped_iou = model.decode(*clicks, inputs["pred_mask"])
    use_mask = inputs["has_mask"] > 0
    return keras.Model(
        inputs,
        {
            "logits": keras.ops.where(use_mask[:, None, None], looped_logits, logits),
            "iou": keras.ops.where(use_mask, looped_iou, iou),
        },
        name="decoder",
    )


def export(model, output_dir: str) -> None:
    model.image_encoder.export(os.path.join(output_dir, "encoder"))
    build_decoder(model).export(os.path.join(output_dir, "decoder"))


if __name__ == "__main__":
    mlflow.set_tracking_uri("https://mlflow.clari.us")  # credentials from MLFLOW_TRACKING_USERNAME / _PASSWORD
    print(f"Loading runs:/{RUN_ID}/model")
    sam = mlflow.tensorflow.load_model(
        f"runs:/{RUN_ID}/model", keras_model_kwargs={"compile": False, "custom_objects": VIT_LAYERS}
    )
    export(sam, OUTPUT_DIR)
    print(f"Exported encoder/ and decoder/ to {os.path.abspath(OUTPUT_DIR)}")
