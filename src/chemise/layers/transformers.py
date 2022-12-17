import flax.linen as nn
from chemise.layers.mlp import MLP


class EncoderBlock(nn.Module):
    input_dims: int
    num_heads: int
    dim_feedforward: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x, mask=None, train=True):
        # Multi Head Attention block
        atten = nn.SelfAttention(num_heads=self.num_heads)(x, mask=mask, deterministic=not train)(x)
        atten = nn.Dropout(self.dropout_rate)(atten, deterministic=not train)
        x = nn.LayerNorm()(x + atten)

        # MLP block
        ff = MLP(dropout_rate=self.dropout_rate, width=self.dim_feedforward, depth=1)(x, deterministic=not train)
        ff = nn.Dense(self.input_dims)(ff)
        ff = nn.Dropout(self.dropout_rate)(ff, deterministic=not train)
        x = nn.LayerNorm()(x + ff)
        return x


class TransformerEncoder(nn.Module):
    num_layers: int
    input_dim: int
    num_heads: int
    dim_feedforward: int
    dropout_prob: float
    skip: bool = True


    @nn.compact
    def __call__(self, x, mask=None, train=True):
        x = EncoderBlock(self.input_dim, self.num_heads, self.dim_feedforward, self.dropout_prob)(x, mask=mask,
                                                                                                    train=train)
        for l in range(self.num_layers - 1):
            encb = EncoderBlock(self.input_dim, self.num_heads, self.dim_feedforward, self.dropout_prob)
            enc = encb(x, mask=mask, train=train)
            x = nn.LayerNorm(x + enc) if self.skip else x

        return x
