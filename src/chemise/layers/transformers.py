from typing import Callable, Tuple, Any

import flax.linen as nn
import jax
import jax.numpy as jnp

Array = Any
Shape = Tuple[int]
Dtype = Any

class PositionalEncoding(nn.Module):
    d_model : int         # Hidden dimensionality of the input.
    max_len : int = 5000  # Maximum length of a sequence to expect.

    def setup(self):
        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = jnp.zeros((self.max_len, self.d_model))
        position = jnp.arange(0, self.max_len, dtype=jnp.float32)[:,None]
        div_term = jnp.exp(jnp.arange(0, self.d_model, 2) * (-jnp.log(10000.0) / self.d_model))
        pe[:, 0::2] = jnp.sin(position * div_term)
        pe[:, 1::2] = jnp.cos(position * div_term)
        pe = pe[None]
        self.pe = jax.device_put(pe)

    def __call__(self, x):
        x = x + self.pe[:, :x.shape[1]]
        return x


class AddPositionEmbs(nn.Module):
    """Adds learned positional embeddings to the inputs.
    Attributes:
    posemb_init: positional embedding initializer.

    from https://github.com/google-research/vision_transformer/blob/c5f3aaed603bfb90418236db44306d8829eafbe6/vit_jax/models_vit.py#L37
    """

    posemb_init: Callable[[jax.random.PRNGKey, Shape, Dtype], Array]
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs):
        """Applies the AddPositionEmbs module.
        Args:
          inputs: Inputs to the layer.
        Returns:
          Output tensor with shape `(bs, timesteps, in_dim)`.
        """
        # inputs.shape is (batch_size, seq_len, emb_dim).
        assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                                  ' but it is: %d' % inputs.ndim)
        pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
        pe = self.param('pos_embedding', self.posemb_init, pos_emb_shape, self.dtype)
        return inputs + pe

class EncoderBlock(nn.Module):
    input_dims: int
    num_heads: int
    dim_feedforward: int
    dropout_rate: float

    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x, mask=None, train=True):
        # Multi Head Attention block
        atten = nn.LayerNorm(dtype=self.dtype)(x)
        atten = nn.SelfAttention(num_heads=self.num_heads, dtype=self.dtype)(atten, mask=mask, deterministic=not train)
        atten = nn.Dropout(self.dropout_rate)(atten, deterministic=not train)
        x = x + atten # Skip connection

        # MLP block
        ff = nn.LayerNorm(dtype=self.dtype)(x)
        ff = nn.Dense(self.dim_feedforward, dtype=self.dtype)(ff)
        ff = nn.gelu(ff)
        ff = nn.Dropout(self.dropout_rate)(ff, deterministic=not train)
        ff = nn.Dense(self.input_dims, dtype=self.dtype)(ff)
        x = x + ff # Skip connection

        return x


class TransformerEncoder(nn.Module):
    input_dim: int
    dim_feedforward: int

    num_layers: int = 2
    num_heads: int = 4
    dropout_prob: float = 0.05
    skip: bool = True

    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x, mask=None, train=True):
        skip_x = x
        x = nn.LayerNorm(dtype=self.dtype)(x)
        x = EncoderBlock(self.input_dim, self.num_heads, self.dim_feedforward,
                         self.dropout_prob, dtype=self.dtype)(x, mask=mask, train=train)

        for l in range(self.num_layers - 1):
            # Make the encoder block
            encb = EncoderBlock(self.input_dim, self.num_heads, self.dim_feedforward,
                                self.dropout_prob, dtype=self.dtype)
            # Apply it with a skip connection
            # x = skip_x + x if self.skip else x
            # skip_x = x # Save for next cycle

            x = encb(x, mask=mask, train=train)
            x = nn.LayerNorm(dtype=self.dtype)(x)
        return x
