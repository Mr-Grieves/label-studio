"""isonoai's SAM, copied from isonoai/models/sam.py: just what Keras needs to load a trained SAM from MLflow.

Every class keeps its `isonoai` registration name -- the saved model refers to them by it -- and their code is
unchanged, so keep them in sync with isonoai if the architecture changes: they must match the code that trained the
model being loaded. Currently isonoai's box-only SAM (a box in; no clicks, no mask prompt). Building a SAM from a
backbone run is left out; `freeze_backbone` stays, since `SAM` calls it when it's built.
"""

from __future__ import annotations

import math
from typing import Any

import keras
import tensorflow as tf

from tsam.utils.model_layers import (
    AddCLSToken,
    AddPositionalEmbeddings,
    Mlp,
    TransformerBlock,
)

# The pretrained ViT's layers aren't registered with Keras, so every load of a saved SAM needs them
VIT_LAYERS = {
    "AddPositionalEmbeddings": AddPositionalEmbeddings,
    "AddCLSToken": AddCLSToken,
    "TransformerBlock": TransformerBlock,
}

SAM_NECK_LAYERNORM_EPS = 1e-6
CONV_LAYERNORM_EPS = 1e-6  # prompt encoder's mask downscaler and mask decoder's upscaling
TRANSFORMER_LAYERNORM_EPS = 1e-5  # mask decoder's two-way transformer


def _fourier_features(coords: tf.Tensor, gaussian_matrix: tf.Tensor) -> tf.Tensor:
    """SAM's random Fourier positional encoding of (x, y) coordinates in [0, 1]: (..., 2) -> (..., 2 * frequencies)."""
    projected = keras.ops.matmul(2.0 * coords - 1.0, gaussian_matrix) * (2.0 * math.pi)
    return keras.ops.concatenate([keras.ops.sin(projected), keras.ops.cos(projected)], axis=-1)


@keras.saving.register_keras_serializable(package="isonoai")
class SAM(keras.Model):
    """The full SAM model: image encoder, prompt encoder and mask decoder, each reachable as an attribute.

    Built as a functional model of the whole pass, so it can be exported as is: `images` (B, S, S, 1) and `boxes`
    (B, 4) in, as (x0, y0, x1, y1) pixels; the `mask` logits (B, 4E, 4E) and its predicted `iou` (B,), an unbounded
    regression, out.

    Args:
        image_encoder: Maps images (B, H, W, 1) to embeddings (B, E, E, dim).
        prompt_encoder: A `PromptEncoder`.
        mask_decoder: A `MaskDecoder`.
        backbone_trainable: Whether the image encoder's pretrained patch embedder and ViT train; if not, only its neck
            does. Saved with the model, so a reloaded model trains the same parts.
    """

    MODULES = ("image_encoder", "prompt_encoder", "mask_decoder")

    def __init__(
        self,
        image_encoder: keras.Model,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        backbone_trainable: bool = False,
        **kwargs,
    ) -> None:
        size = prompt_encoder.input_image_size
        inputs = {"images": keras.Input((size, size, 1), name="images"), "boxes": keras.Input((4,), name="boxes")}
        prompts = prompt_encoder({"boxes": inputs["boxes"]})
        outputs = mask_decoder({"image_embeddings": image_encoder(inputs["images"]), **prompts})
        super().__init__(inputs=inputs, outputs={"mask": outputs["mask"][:, 0], "iou": outputs["iou"][:, 0]}, **kwargs)

        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.backbone_trainable = backbone_trainable
        if not backbone_trainable:
            freeze_backbone(image_encoder)  # Here, not left to Keras: it doesn't restore a nested model's `trainable`

    def get_config(self) -> dict[str, Any]:
        """Return serialisable model config: the modules, not the functional graph built from them."""
        modules = {name: keras.saving.serialize_keras_object(getattr(self, name)) for name in self.MODULES}
        return {
            "name": self.name,
            "trainable": self.trainable,
            "backbone_trainable": self.backbone_trainable,
            **modules,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SAM:
        """Rebuild the model, and each of its modules, from its config."""
        with keras.saving.custom_object_scope(VIT_LAYERS):
            modules = {name: keras.saving.deserialize_keras_object(config.pop(name)) for name in cls.MODULES}
        return cls(**modules, **config)


def freeze_backbone(image_encoder: keras.Model) -> None:
    """Freeze the image encoder's pretrained patch embedder and ViT, leaving only the neck to train."""
    for layer in image_encoder.layers:
        layer.trainable = layer.name == "sam_neck"
        for variable in layer.weights:  # On reload, the ViT's blocks create their variables after `trainable` is set
            variable.trainable = layer.trainable


@keras.saving.register_keras_serializable(package="isonoai")
class TokensToGrid(keras.layers.Layer):
    """Drops the CLS token and reshapes the patch tokens to their square grid: (B, 1 + g*g, C) -> (B, g, g, C)."""

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Drop the CLS token, then reshape the patch tokens."""
        tokens = x[:, 1:, :]
        grid = tf.cast(tf.math.round(tf.math.sqrt(tf.cast(tf.shape(tokens)[1], tf.float32))), tf.int32)
        return tf.reshape(tokens, [tf.shape(tokens)[0], grid, grid, tokens.shape[-1]])

    def compute_output_shape(self, input_shape: tuple[int | None, ...]) -> tuple[int | None, ...]:
        """The grid size depends on the number of tokens, so only batch and channels are static."""
        return (input_shape[0], None, None, input_shape[-1])


@keras.saving.register_keras_serializable(package="isonoai")
class SamNeck(keras.layers.Layer):
    """SAM's neck on the token grid, then 2x transposed convs up to the embedding grid.

    SAM's own neck (1x1 conv -> LN -> 3x3 conv -> LN) projects to `neck_dim` and mixes neighbouring tokens, and its
    final LayerNorms put the embeddings on the scale MedSAM's mask decoder was trained on. Running the 3x3 conv before
    upsampling costs a quarter of running it after.

    Args:
        neck_dim: Output channels.
        upsample: Factor by which to upsample the grid: 1, 2 or 4.
    """

    def __init__(self, neck_dim: int, upsample: int = 2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.neck_dim = neck_dim
        self.upsample = upsample

        self.neck = [
            keras.layers.Conv2D(neck_dim, 1, use_bias=False, name="conv_1"),
            keras.layers.LayerNormalization(epsilon=SAM_NECK_LAYERNORM_EPS, name="norm_1"),
            keras.layers.Conv2D(neck_dim, 3, padding="same", use_bias=False, name="conv_2"),
            keras.layers.LayerNormalization(epsilon=SAM_NECK_LAYERNORM_EPS, name="norm_2"),
        ]
        for i in range(int(math.log2(upsample))):
            if i > 0:
                # Without a nonlinearity between them, successive transposed convs collapse into one linear upsample.
                self.neck.append(keras.layers.Activation("gelu", name=f"up_act_{i}"))
            self.neck.append(keras.layers.Conv2DTranspose(neck_dim, 2, strides=2, name=f"up_{i}"))
            self.neck.append(keras.layers.LayerNormalization(epsilon=SAM_NECK_LAYERNORM_EPS, name=f"up_norm_{i}"))

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Apply SAM's neck and upsample the token grid."""
        for layer in self.neck:
            x = layer(x)
        return x


@keras.saving.register_keras_serializable(package="isonoai")
class PromptEncoder(keras.Model):
    """SAM's prompt encoder, for box prompts only (as in MedSAM).

    A box is encoded as SAM encodes one: its two corners, each a positional encoding plus a learned corner embedding.
    There is no mask prompt, so the dense embedding is SAM's learned "no mask" embedding everywhere.

    Args:
        dim: Embedding width.
        input_image_size: Size of the (square) frame the box coordinates are given in.
        embedding_size: Size of the (square) image embedding grid.
    """

    def __init__(self, dim: int = 256, input_image_size: int = 256, embedding_size: int = 64, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dim = dim
        self.input_image_size = input_image_size
        self.embedding_size = embedding_size

        normal = keras.initializers.RandomNormal(stddev=1.0)
        self.gaussian_matrix = self.add_weight(
            name="gaussian_matrix", shape=(2, dim // 2), initializer=normal, trainable=False
        )
        self.corner_embeds = self.add_weight(name="corner_embeds", shape=(2, dim), initializer=normal)  # [TL, BR]
        self.no_mask_embed = self.add_weight(name="no_mask_embed", shape=(dim,), initializer=normal)
        self.built = True

    def call(self, inputs: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        """Embed the prompts.

        Args:
            inputs: A dictionary with `boxes` (B, 4): (x0, y0, x1, y1) in pixels of the input frame.

        Returns:
            `prompt_sparse_embeddings` (B, 2, dim), `prompt_dense_embeddings` (B, E, E, dim) and
            `prompt_dense_positional_embeddings` (1, E, E, dim).
        """
        boxes = inputs["boxes"]
        corners = keras.ops.reshape(boxes, (-1, 2, 2))  # (B, [top-left, bottom-right], (x, y))
        sparse = _fourier_features((corners + 0.5) / self.input_image_size, self.gaussian_matrix) + self.corner_embeds

        grid = (keras.ops.shape(boxes)[0], self.embedding_size, self.embedding_size, self.dim)
        dense = keras.ops.broadcast_to(self.no_mask_embed, grid)

        # Positional encoding of every embedding grid cell's centre, for the mask decoder.
        centres = (keras.ops.arange(self.embedding_size, dtype="float32") + 0.5) / self.embedding_size
        y, x = keras.ops.meshgrid(centres, centres, indexing="ij")
        dense_pe = _fourier_features(keras.ops.stack([x, y], axis=-1), self.gaussian_matrix)[None]

        return {
            "prompt_sparse_embeddings": sparse,
            "prompt_dense_embeddings": dense,
            "prompt_dense_positional_embeddings": dense_pe,
        }

    def get_config(self) -> dict[str, Any]:
        """Return serialisable model config."""
        return {
            **super().get_config(),
            "dim": self.dim,
            "input_image_size": self.input_image_size,
            "embedding_size": self.embedding_size,
        }


class TwoWayBlock(keras.layers.Layer):
    """One block of SAM's two-way transformer.

    The prompt tokens attend to themselves, then to the image, then pass through an MLP; then the image attends to the
    tokens. Each step is residual and post-normalised. Queries and keys carry positional encodings; values don't.

    Args:
        dim: Token and image embedding width.
        heads: Number of attention heads.
        downsample_rate: Internal width reduction of the token-image cross-attentions.
        skip_first_layer_pe: Whether the token self-attention skips the positional encoding and the residual (SAM's
            first block does).
    """

    def __init__(
        self, dim: int, heads: int, downsample_rate: int, mlp_ratio: int, skip_first_layer_pe: bool, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.skip_first_layer_pe = skip_first_layer_pe
        self.self_attn = keras.layers.MultiHeadAttention(heads, dim // heads)
        self.token_to_image = keras.layers.MultiHeadAttention(heads, dim // heads // downsample_rate)
        self.image_to_token = keras.layers.MultiHeadAttention(heads, dim // heads // downsample_rate)
        self.mlp = Mlp(dim, ratio=mlp_ratio)  # SAM: 256 -> 2048 -> 256
        self.norms = [keras.layers.LayerNormalization(epsilon=TRANSFORMER_LAYERNORM_EPS) for _ in range(4)]

    def call(
        self, queries: tf.Tensor, keys: tf.Tensor, query_pe: tf.Tensor, key_pe: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Update the tokens (`queries`, B x Nq x dim) and the image (`keys`, B x Nk x dim)."""
        if self.skip_first_layer_pe:
            queries = self.self_attn(queries, queries)
        else:
            q = queries + query_pe
            queries = queries + self.self_attn(q, queries, key=q)
        queries = self.norms[0](queries)
        queries = self.norms[1](queries + self.token_to_image(queries + query_pe, keys, key=keys + key_pe))
        queries = self.norms[2](queries + self.mlp(queries))
        keys = self.norms[3](keys + self.image_to_token(keys + key_pe, queries, key=queries + query_pe))
        return queries, keys


@keras.saving.register_keras_serializable(package="isonoai")
class MaskDecoder(keras.Model):  # pylint: disable=too-many-instance-attributes
    """SAM's mask decoder, predicting a single mask and its IoU.

    Its dict inputs and outputs follow keras-hub's `SAMMaskDecoder`, with one mask output instead of four. Unlike SAM,
    its MLPs are our 2-layer GELU `Mlp` (SAM's are ReLU, with 3 layers in the output heads).

    Args:
        dim: Token and image embedding width.
        heads: Number of attention heads.
        depth: Number of two-way transformer blocks.
        mlp_ratio: Hidden-width multiplier of the two-way blocks' MLPs.
        attention_downsample_rate: Internal width reduction of the token-image cross-attentions.
    """

    def __init__(
        self,
        dim: int = 256,
        heads: int = 8,
        depth: int = 2,
        mlp_ratio: int = 2,
        attention_downsample_rate: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.heads = heads
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.attention_downsample_rate = attention_downsample_rate
        self.output_tokens = self.add_weight(  # [IoU token, mask token]
            name="output_tokens", shape=(2, dim), initializer=keras.initializers.RandomNormal(stddev=1.0)
        )
        self.blocks = [
            TwoWayBlock(dim, heads, attention_downsample_rate, mlp_ratio, skip_first_layer_pe=i == 0)
            for i in range(depth)
        ]
        self.final_attn = keras.layers.MultiHeadAttention(heads, dim // heads // attention_downsample_rate)
        self.final_norm = keras.layers.LayerNormalization(epsilon=TRANSFORMER_LAYERNORM_EPS)
        self.upscaling = keras.Sequential(
            [
                keras.layers.Conv2DTranspose(dim // 4, 2, strides=2),
                keras.layers.LayerNormalization(epsilon=CONV_LAYERNORM_EPS),
                keras.layers.Activation("gelu"),
                keras.layers.Conv2DTranspose(dim // 8, 2, strides=2),
                keras.layers.Activation("gelu"),
            ]
        )
        self.hypernetwork = Mlp(dim // 8, ratio=8)  # mask token -> per-pixel weights, hidden width dim
        self.iou_head = Mlp(1, ratio=dim)  # hidden width dim

        # Create the weights now, rather than on the first call: the trainer builds its optimizer before training.
        grid, tokens = keras.ops.zeros((1, 1, 1, dim)), keras.ops.zeros((1, 1, dim))
        self(
            {
                "image_embeddings": grid,
                "prompt_sparse_embeddings": tokens,
                "prompt_dense_embeddings": grid,
                "prompt_dense_positional_embeddings": grid,
            }
        )

    def call(self, inputs: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        """Predict a mask and its IoU.

        Args:
            inputs: `image_embeddings` (B, E, E, dim), plus the prompt encoder's outputs.

        Returns:
            `masks` (B, 1, 4E, 4E) mask logits and `iou` (B, 1).
        """
        # The image, with its dense prompt and positional encoding, as one token per grid cell.
        image = inputs["image_embeddings"] + inputs["prompt_dense_embeddings"]
        shape = keras.ops.shape(image)
        keys = keras.ops.reshape(image, (shape[0], -1, self.dim))
        key_pe = keras.ops.reshape(
            keras.ops.broadcast_to(inputs["prompt_dense_positional_embeddings"], shape), (shape[0], -1, self.dim)
        )
        # The tokens: [IoU token, mask token, prompt tokens...].
        output_tokens = keras.ops.broadcast_to(self.output_tokens, (shape[0], 2, self.dim))
        tokens = keras.ops.concatenate([output_tokens, inputs["prompt_sparse_embeddings"]], axis=1)

        queries = tokens
        for block in self.blocks:
            queries, keys = block(queries, keys, tokens, key_pe)
        queries = self.final_norm(queries + self.final_attn(queries + tokens, keys, key=keys + key_pe))

        # The mask token, through the hypernetwork, weights the upscaled image embedding per pixel.
        upscaled = self.upscaling(keras.ops.reshape(keys, shape))
        masks = keras.ops.einsum("bc,bhwc->bhw", self.hypernetwork(queries[:, 1]), upscaled)
        return {"mask": masks[:, None], "iou": self.iou_head(queries[:, 0])}

    def get_config(self) -> dict[str, Any]:
        """Return serialisable model config."""
        return {
            **super().get_config(),
            "dim": self.dim,
            "heads": self.heads,
            "depth": self.depth,
            "mlp_ratio": self.mlp_ratio,
            "attention_downsample_rate": self.attention_downsample_rate,
        }
