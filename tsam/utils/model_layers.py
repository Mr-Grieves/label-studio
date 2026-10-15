"""Modules for Keras"""

from __future__ import annotations

import logging

import keras
import tensorflow as tf

logger = logging.getLogger(__name__)


LAYERNORM_EPS = 1e-6
MLP_RATIO = 4.0

class _LearnedTokenLayer(keras.layers.Layer):
    """Base for layers holding one learned `(1, 1, dim)` token, tolerant of checkpoints that never saved it.

    Before the token was created with `add_weight`, it was a raw `tf.Variable` that Keras 3 did not track, so
    checkpoints from that era store no variable for these layers. Loading one keeps the freshly initialised token
    instead of failing; the token used when that checkpoint was trained was never saved and cannot be recovered.

    Args:
        dim: Dimensionality of the embeddings.
        weight_name: Name of the token weight.
        init_stddev: Standard deviation of the normal initialiser for the token.
    """

    def __init__(self, dim: int, weight_name: str, init_stddev: float = 0.02, **kwargs) -> None:
        super().__init__(**kwargs)
        self.token = self.add_weight(
            shape=(1, 1, dim),
            initializer=keras.initializers.RandomNormal(stddev=init_stddev),
            trainable=True,
            name=weight_name,
        )

    def load_own_variables(self, store) -> None:
        """Load the token, keeping its fresh initialisation when the checkpoint predates it being tracked."""
        if len(store.keys()) == 0:
            logger.warning(
                f"Checkpoint stores no '{self.token.name}' for layer '{self.name}' (saved before the token was "
                "tracked); keeping a freshly initialised token, so outputs will not match the trained model exactly."
            )
            return
        super().load_own_variables(store)



class AddPositionalEmbeddings(keras.layers.Layer):
    """Add positional embeddings to input tensor."""

    def call(self, x: tf.Tensor, include_cls: bool = False) -> tf.Tensor:
        """Add positional encodings to input tensor.

        Args:
            x: Input tensor. Shape: (batch_size, num_patches, dim)
            include_cls: Whether to include positional encoding for the cls token.
        """
        dim = x.shape[-1]
        num_patches = tf.cast(tf.shape(x)[-2], tf.float32)

        # Initialize positional embeddings
        grid_size = tf.cast(tf.math.pow(num_patches, 0.5), tf.int32)
        pos_embed = get_2d_sincos_pos_embed(dim, grid_size, add_cls_token=include_cls)
        pos_embed = tf.expand_dims(pos_embed, axis=0)

        return x + pos_embed


class AddCLSToken(_LearnedTokenLayer):
    """Prepends a CLS token to patch embeddings.

    NOTE: This method currently ignores any positional embedding for the CLS token, which I deemed unnecessary.

    Args:
        dim: Dimensionality of the embeddings.
    """

    def __init__(self, dim: int, **kwargs) -> None:
        super().__init__(dim, weight_name="cls_token", **kwargs)

    @property
    def cls_token(self) -> keras.Variable:
        """The learned CLS token, shape `(1, 1, dim)`."""
        return self.token

    def call(self, x):
        """Concatenates the cls token to patch embeddings."""
        cls_tokens = tf.tile(self.cls_token, (tf.shape(x)[0], 1, 1))
        return tf.concat([cls_tokens, x], axis=1)


class DropPath(keras.layers.Layer):
    """DropPath layer.

    This layer applies drop-path regularization to the input tensor.
    This is done per-sample, NOT per-batch.

    Args:
        drop_prob: Probability of dropping a path.
    """

    def __init__(self, drop_prob: float, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x):
        """Forward pass."""
        if self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
        random_tensor = keep_prob + tf.random.uniform(shape, dtype=x.dtype)
        random_tensor = tf.floor(random_tensor)  # binarize
        output = x / keep_prob * random_tensor
        return output
    

class Mlp(keras.layers.Layer):
    """Simple Multi-Layer Perceptron (MLP) block.

    This serves as the feed forward network in the transformer block.

    Args:
        output_features: Number of output features.
        ratio: Ratio of hidden to output features.
    """

    def __init__(self, output_features, ratio, **kwargs):
        super().__init__(**kwargs)
        self.fc1 = keras.layers.Dense(int(output_features * ratio), activation=tf.nn.gelu)
        self.fc2 = keras.layers.Dense(output_features)

    def call(self, x):
        """Feed forward logic."""
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class TransformerBlock(keras.layers.Layer):
    """Transformer block for ViT.

    Implemented based off of:
    https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

    Args:
        projection_dim: Dimensionality of the output embeddings.
        heads: Number of attention heads to use.
        mlp_ratio: Ratio of hidden to output features in the MLP.
    """

    def __init__(self, projection_dim: int, heads: int, mlp_ratio: float = MLP_RATIO, dpr: float = 0.0, **kwargs):
        super().__init__(**kwargs)

        # Layer normalization 1.
        self.norm1 = keras.layers.LayerNormalization(epsilon=LAYERNORM_EPS)

        # Multi-head attention layer.
        self.attn = CustomMultiHeadAttention(dim=projection_dim, heads=heads, name="attention")

        # DropPath layer.
        self.drop_path = DropPath(dpr) if dpr > 0.0 else keras.layers.Identity()

        # Layer normalization 2.
        self.norm2 = keras.layers.LayerNormalization(epsilon=LAYERNORM_EPS)

        # MLP block.
        self.mlp = Mlp(output_features=projection_dim, ratio=mlp_ratio)

    def call(self, x):
        """Forward pass."""
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def get_attention(self, x):
        """Get attention map."""
        return self.attn.get_attention(self.norm1(x), apply_weights=False)


class CustomMultiHeadAttention(keras.layers.Layer):
    """Custom multi-head attention Layer.

    This is needed because keras.layers.MultiHeadAttention has >10x as many parameters.
    Implemented following: https://tinyurl.com/33ety6yn

    Args:
        dim: Ouput projection dimension.
        heads: Number of attention heads to use.
    """

    def __init__(self, dim, heads, **kwargs):
        super().__init__(**kwargs)
        assert dim % heads == 0, "Embedding dimension must be divisible by number of heads"
        self.dim = dim  # Dimension of the model
        self.heads = heads  # Number of attention heads to use
        self.head_dim = dim // heads

        self.qkv = keras.layers.Dense(3 * dim, use_bias=True)  # Combined qkv projection
        self.proj = keras.layers.Dense(dim)  # Final linear projection

    def call(self, x):
        """Perform multi-head attention."""
        _, _, C = x.shape
        x = self.get_attention(x)
        x = keras.layers.Permute((2, 1, 3))(x)
        x = keras.layers.Reshape((-1, C))(x)
        x = self.proj(x)
        return x

    def get_attention(self, x, apply_weights=True):
        """Get attention weights."""
        qkv = self.qkv(x)
        qkv = keras.layers.Reshape((-1, 3, self.heads, self.head_dim))(qkv)
        qkv = keras.layers.Permute((2, 3, 1, 4))(qkv)
        q, k, v = keras.ops.unstack(qkv, axis=1)

        attn = keras.ops.matmul(q, keras.ops.transpose(k, axes=[0, 1, 3, 2]))
        attn /= tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        attn = keras.ops.softmax(attn)
        if apply_weights:
            return keras.ops.matmul(attn, v)
        return attn


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int | tf.Tensor, add_cls_token: bool = False) -> tf.Tensor:
    """Create 2D sin/cos positional embeddings.

    Args:
        embed_dim: Embedding dimension.
        grid_size: Grid size.
        add_cls_token: Whether to add a CLS token. Defaults to False.

    Returns:
        2D sin/cos positional embeddings.
    """
    assert embed_dim % 4 == 0, "embed_dim must be multiple of 4"

    # Create grid coordinates based on grid size
    grid_h = tf.range(grid_size, dtype=tf.float32)
    grid_w = tf.range(grid_size, dtype=tf.float32)
    grid = tf.meshgrid(grid_w, grid_h)
    grid = tf.stack(grid, axis=0)
    grid = tf.reshape(grid, [2, 1, grid_size, grid_size])

    # Create 2D sin/cos positional embeddings from grid
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    pos_embed = tf.concat([emb_h, emb_w], axis=1)  # (H*W, D)

    if add_cls_token:
        pos_embed = tf.concat([tf.zeros((1, embed_dim)), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: tf.Tensor) -> tf.Tensor:
    """Create 1D sin/cos positional embeddings from grid.

    Args:
        embed_dim: output dimension for each position
        pos: positions to be encoded. Shape: (M,)

    Returns:
        1D sin/cos positional embeddings. Shape: (M, D)
    """
    assert embed_dim % 2 == 0, "embed_dim/2 must be even"

    omega = tf.range(embed_dim // 2, dtype="float32")
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = tf.reshape(pos, [-1])  # (M,)
    out = tf.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    # Half of the positions get sin pattern, other half get cos pattern
    emb_sin = tf.sin(out)  # (M, D/2)
    emb_cos = tf.cos(out)  # (M, D/2)

    emb = tf.concat([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

