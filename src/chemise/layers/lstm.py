import functools

import jax.numpy as jnp
from flax import linen as nn


class SimpleLSTM(nn.Module):
    """A simple unidirectional LSTM.
    Example Usage:

    @nn.compact
    def __call__(self, inputs):
        batch_size = inputs.shape[0]
        initial_state = SimpleLSTM.initialize_carry((batch_size,), self.hidden_size)
        _, forward_outputs = SimpleLSTM()(initial_state, inputs)
    """
    cell: nn.OptimizedLSTMCell = None

    @functools.partial(
        nn.transforms.scan,
        variable_broadcast='params',
        in_axes=1, out_axes=1,
        split_rngs={'params': False})
    @nn.compact
    def __call__(self, carry, x):
        carry, x = self.cell(carry, x)
        return carry, x


class AutoregLSTM(nn.Module):
    output_layer: nn.Dense = None
    cell: nn.OptimizedLSTMCell = None

    @functools.partial(
        nn.transforms.scan,
        variable_broadcast='params',
        in_axes=1, out_axes=1,
        split_rngs={'params': False})
    @nn.compact
    def __call__(self, carry_pred, x):
        carry, past_pred = carry_pred
        in_x = jnp.concatenate([x, past_pred], axis=-1)
        carry, y = self.cell(carry, in_x)
        y = self.output_layer(y)
        return (carry, y), y


class FullAutoLSTM(nn.Module):
    output_layer: nn.Dense = None
    cell: nn.OptimizedLSTMCell = None

    @nn.compact
    def __call__(self, initial_state, warmup_input, auto_input):
        carry, warm_lstm = SimpleLSTM(cell=self.cell)(initial_state, warmup_input)
        output_warm = self.output_layer(warm_lstm)

        autoreg = AutoregLSTM(output_layer=self.output_layer, cell=self.cell)
        _, rest_output = autoreg((carry, output_warm[:, -1]), auto_input)

        pred = jnp.concatenate([output_warm, rest_output], axis=-2)
        return pred