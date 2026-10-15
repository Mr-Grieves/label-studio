"""isonoai's SAM, copied from isonoai/models/sam.py: just what Keras needs to load a trained SAM from MLflow.

Every class keeps its `isonoai` registration name -- the saved model refers to them by it -- and their code is
unchanged, so keep them in sync with isonoai if the architecture changes. Training-only parts (building a SAM from
a backbone run, freezing the backbone) are left out.
"""

from __future__ import annotations

import math
from typing import Any

import keras
import tensorflow as tf

from tsam.utils.model_layers import AddCLSToken, AddPositionalEmbeddings, Mlp, TransformerBlock

# The pretrained ViT's layers aren't registered with Keras, so every load of a saved SAM needs them
VIT_LAYERS = {
    "AddPositionalEmbeddings": AddPositionalEmbeddings,
    "AddCLSToken": AddCLSToken,
    "TransformerBlock": TransformerBlock,
}

SAM_NECK_LAYERNORM_EPS = 1e-6
CONV_LAYERNORM_EPS = 1e-6  # prompt encoder's mask downscaler and mask decoder's upscaling
TRANSFORMER_LAYERNORM_EPS = 1e-5  # mask decoder's two-way transformer


@keras.saving.register_keras_serializable(package="isonoai")
class SAM(keras.Model):
    """The full SAM model: image encoder, prompt encoder and mask decoder, each reachable as an attribute.

    Built as a functional model of the zero-shot pass (`images` and `points` in, `masks` and `iou` out), so it can
    be exported as is. Interactive rounds embed the image once with `image_encoder`, then call `decode` once per round.

    Args:
        image_encoder: Maps images (B, H, W, 1) to embeddings (B, E, E, dim).
        prompt_encoder: A `PromptEncoder`.
        mask_decoder: A `MaskDecoder`.
    """

    MODULES = ("image_encoder", "prompt_encoder", "mask_decoder")

    def __init__(
        self, image_encoder: keras.Model, prompt_encoder: PromptEncoder, mask_decoder: MaskDecoder, **kwargs
    ) -> None:
        # Zero-shot mode: the clicks alone, all foreground, no mask prompt.
        size = prompt_encoder.input_image_size
        inputs = {
            "images": keras.Input((size, size, 1), name="images"),
            "points": keras.Input((None, 2), name="points"),
        }
        labels = keras.ops.ones_like(inputs["points"][:, :, 0])
        embeddings = image_encoder(inputs["images"])
        masks, iou = _decode(prompt_encoder, mask_decoder, embeddings, inputs["points"], labels)
        super().__init__(inputs=inputs, outputs={"mask": masks, "iou": iou}, **kwargs)

        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

    def decode(
        self, embeddings: tf.Tensor, points: tf.Tensor, labels: tf.Tensor, pred_mask: tf.Tensor | None = None
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """One prompt round: predict the mask from the image embeddings, the clicks so far and the last prediction.

        Args:
            embeddings: (B, E, E, dim), from `image_encoder`.
            points: (B, N, 2), the clicks in pixels of the input frame.
            labels: (B, N), 1 for foreground clicks, 0 for background.
            pred_mask: (B, 4E, 4E), the previous round's mask logits, or None for no mask prompt.

        Returns:
            The mask logits (B, 4E, 4E) and its predicted IoU with the ground truth (B,), an unbounded regression.
        """
        return _decode(self.prompt_encoder, self.mask_decoder, embeddings, points, labels, pred_mask)

    def get_config(self) -> dict[str, Any]:
        """Return serialisable model config: the modules, not the functional graph built from them."""
        modules = {name: keras.saving.serialize_keras_object(getattr(self, name)) for name in self.MODULES}
        return {"name": self.name, "trainable": self.trainable, **modules}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SAM:
        """Rebuild the model, and each of its modules, from its config."""
        with keras.saving.custom_object_scope(VIT_LAYERS):
            modules = {name: keras.saving.deserialize_keras_object(config.pop(name)) for name in cls.MODULES}
        return cls(**modules, **config)


def _decode(
    prompt_encoder: PromptEncoder,
    mask_decoder: MaskDecoder,
    embeddings: tf.Tensor,
    points: tf.Tensor,
    labels: tf.Tensor,
    pred_mask: tf.Tensor | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    """`SAM.decode`, also used to build `SAM`'s graph before its modules are attributes. Uses `keras.ops` only, so it
    runs on symbolic inputs as well as tensors."""
    prompts = {
        # SAM pads the clicks with a not-a-point when there are no box prompts.
        "points": keras.ops.concatenate([points, keras.ops.zeros_like(points[:, :1])], axis=1),
        "labels": keras.ops.concatenate([labels, -keras.ops.ones_like(labels[:, :1])], axis=1),
    }
    if pred_mask is not None:
        prompts["masks"] = keras.ops.expand_dims(pred_mask, -1)
    outputs = mask_decoder({"image_embeddings": embeddings, **prompt_encoder(prompts)})
    return outputs["mask"][:, 0], outputs["iou"][:, 0]


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


def _fourier_features(coords: tf.Tensor, gaussian_matrix: tf.Tensor) -> tf.Tensor:
    """SAM's random Fourier positional encoding of (x, y) coordinates in [0, 1]: (..., 2) -> (..., 2 * frequencies)."""
    projected = keras.ops.matmul(2.0 * coords - 1.0, gaussian_matrix) * (2.0 * math.pi)
    return keras.ops.concatenate([keras.ops.sin(projected), keras.ops.cos(projected)], axis=-1)


@keras.saving.register_keras_serializable(package="isonoai")
class PromptEncoder(keras.Model):  # pylint: disable=too-many-instance-attributes
    """SAM's prompt encoder, for point prompts and an optional mask prompt.

    Its dict inputs and outputs follow keras-hub's `SAMPromptEncoder`, without box prompts.

    Args:
        dim: Embedding width.
        input_image_size: Size of the (square) frame the point coordinates are given in.
        embedding_size: Size of the (square) image embedding grid. Mask prompts are 4x this.
        mask_in_channels: Channels inside the mask downscaler.
    """

    def __init__(
        self,
        dim: int = 256,
        input_image_size: int = 256,
        embedding_size: int = 64,
        mask_in_channels: int = 16,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dim = dim
        self.input_image_size = input_image_size
        self.embedding_size = embedding_size
        self.mask_in_channels = mask_in_channels

        normal = keras.initializers.RandomNormal(stddev=1.0)
        self.gaussian_matrix = self.add_weight(
            name="gaussian_matrix", shape=(2, dim // 2), initializer=normal, trainable=False
        )
        self.foreground_embed = self.add_weight(name="foreground_embed", shape=(dim,), initializer=normal)
        self.background_embed = self.add_weight(name="background_embed", shape=(dim,), initializer=normal)
        self.not_a_point_embed = self.add_weight(name="not_a_point_embed", shape=(dim,), initializer=normal)
        self.no_mask_embed = self.add_weight(name="no_mask_embed", shape=(dim,), initializer=normal)

        self.mask_downscaler = keras.Sequential(
            [
                keras.layers.Conv2D(mask_in_channels // 4, 2, strides=2),
                keras.layers.LayerNormalization(epsilon=CONV_LAYERNORM_EPS),
                keras.layers.Activation("gelu"),
                keras.layers.Conv2D(mask_in_channels, 2, strides=2),
                keras.layers.LayerNormalization(epsilon=CONV_LAYERNORM_EPS),
                keras.layers.Activation("gelu"),
                keras.layers.Conv2D(dim, 1),
            ],
            name="mask_downscaler",
        )
        self.mask_downscaler.build((None, 4 * embedding_size, 4 * embedding_size, 1))
        self.built = True

    def call(self, inputs: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        """Embed the prompts.

        Args:
            inputs: A dictionary containing the following keys:
                - `points` (B, N, 2), (x, y) in pixels of the input frame
                - `labels` (B, N): 1 foreground, 0 background, -1 padding
                - `masks` (B, 4E, 4E, 1) mask logits; omit for no mask prompt.

        Returns:
            `prompt_sparse_embeddings` (B, N, dim), `prompt_dense_embeddings` (B, E, E, dim) and
            `prompt_dense_positional_embeddings` (1, E, E, dim).
        """
        points, labels, masks = inputs["points"], inputs["labels"], inputs.get("masks")

        # Points: positional encoding of the pixel centre, plus a foreground or background embedding. Padding points
        # are replaced outright by the not-a-point embedding.
        sparse = _fourier_features((points + 0.5) / self.input_image_size, self.gaussian_matrix)
        labels = keras.ops.expand_dims(labels, -1)
        sparse = keras.ops.where(labels == 0, sparse + self.background_embed, sparse + self.foreground_embed)
        sparse = keras.ops.where(labels == -1, self.not_a_point_embed, sparse)

        # Mask: downscaled to the embedding grid, or a learned "no mask" embedding everywhere.
        if masks is None:
            grid = (keras.ops.shape(points)[0], self.embedding_size, self.embedding_size, self.dim)
            dense = keras.ops.broadcast_to(self.no_mask_embed, grid)
        else:
            dense = self.mask_downscaler(masks)

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
            "mask_in_channels": self.mask_in_channels,
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
