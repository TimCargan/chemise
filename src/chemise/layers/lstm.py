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
    """
    Autoreg LSTM, uses the output of the previous time step as a feature the input for the next.
    E.g

    step    |    1     |     2    |   ...   |      n     |
    input   | [F, y0]  |  [F, y1] |   ...   |  [F, yn-1] |
    output  | y1       |  y2      |   ...   |     yn     |


    This implementation combines an LSTM cell with a Module, assumed to be a dense mapping output of the LSTM to
    a prediction that can be passed into the next time step. This is done so the LSTM can have a number of units
    different to the feature width of the prediction target e.g if the lstm cell has 128 units and the target feature
    is width 1 we would use a dense(1) to reduce th feature width.
    We can also concat output of the last time step with other features e.g assuming 9 other features and a target with
    of 1 the shapes are:

    step    |    1     |     2    |   ...   |      n     |
    input   | [9 + 1]  |  [F, y1] |   ...   |  [F, yn-1] |
    lstm    | [128]    |  [128]   |   ...   |    [128]   |
    dense   | [1]      |  [1]     |   ...   |     [1]    |
    output  | y1[1]   |  y2      |   ...   |     yn     |

    """
    output_layer: nn.Module = None
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
    """
    Chain and LSTM and AutoReg LSTM together
    """
    output_layer: nn.Module = None
    cell: nn.OptimizedLSTMCell = None

    @nn.compact
    def __call__(self, initial_state, warmup_input, auto_input):
        carry, warm_lstm = SimpleLSTM(cell=self.cell)(initial_state, warmup_input)
        output_warm = self.output_layer(warm_lstm)

        autoreg = AutoregLSTM(output_layer=self.output_layer, cell=self.cell)
        _, rest_output = autoreg((carry, output_warm[:, -1]), auto_input)

        pred = jnp.concatenate([output_warm, rest_output], axis=-2)
        return pred